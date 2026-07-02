from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pywork.coordinator.worker import (
    CoordinatorWorker,
    WorkerBusyError,
    WorkerExecutionMode,
    WorkerRunRequest,
    WorkerSpec,
    WorkerStatus,
    create_worker,
)
from pywork.subagents.manager import create_default_subagent_manager
from pywork.tasks.task import TaskStatus


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run bash",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Run PowerShell",
        },
    },
]


async def fake_llm(messages, *, tools=None, metadata=None):
    tool_names = [
        tool["function"]["name"]
        for tool in tools or []
    ]

    return {
        "content": (
            f"agent={metadata['agent_name']} "
            f"role={metadata['agent_role']} "
            f"mode={metadata['permission_mode']} "
            f"tools={','.join(tool_names)} "
            f"task={messages[-1]['content']}"
        ),
        "metadata": {},
    }


def make_manager(tmp_path: Path):
    return create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )


@pytest.mark.asyncio
async def test_worker_executes_direct_subtask(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    worker = create_worker(
        worker_id="worker_planner",
        worker_role="planner",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="规划 worker.py 的实现",
            parent_task="实现 Coordinator / Worker 系统",
            parent_messages=[
                {
                    "role": "user",
                    "content": "需要实现 context_modifier、worker、coordinator。",
                }
            ],
        )
    )

    assert result.success is True
    assert result.status == WorkerStatus.COMPLETED
    assert result.agent_name == "planner"
    assert result.worker_role == "planner"
    assert "agent=planner" in result.content
    assert result.context_result is not None
    assert result.context_result.messages[0]["role"] == "system"
    assert worker.status == WorkerStatus.IDLE


@pytest.mark.asyncio
async def test_worker_executes_debugger_context(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    worker = create_worker(
        worker_id="worker_debugger",
        worker_role="debugger",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="分析 pytest 失败原因",
            parent_messages=[
                {
                    "role": "tool",
                    "name": "pytest",
                    "content": "FAILED tests/test_x.py::test_case\nTraceback: ValueError: bad value",
                },
                {
                    "role": "user",
                    "content": "为什么失败？",
                },
            ],
        )
    )

    assert result.success is True
    assert result.agent_name == "debugger"

    context_text = "\n".join(
        message["content"]
        for message in result.context_result.messages
    )

    assert "Traceback" in context_text
    assert "pytest" in context_text


@pytest.mark.asyncio
async def test_worker_executes_task_backed_subtask(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    worker = create_worker(
        worker_id="worker_verifier",
        worker_role="verifier",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="验证 worker.py 的测试",
            execution_mode=WorkerExecutionMode.TASK,
            wait=True,
            max_retries=1,
        )
    )

    assert result.success is True
    assert result.status == WorkerStatus.COMPLETED
    assert result.task_record is not None
    assert result.task_record.status == TaskStatus.SUCCEEDED
    assert result.task_record.agent_id == "verifier"
    assert result.task_record_id == result.task_record.id


@pytest.mark.asyncio
async def test_worker_uses_explicit_agent_name(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    worker = create_worker(
        worker_id="worker_custom",
        worker_role="worker",
        agent_name="reviewer",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="审查 worker.py",
        )
    )

    assert result.success is True
    assert result.agent_name == "reviewer"
    assert "agent=reviewer" in result.content


@pytest.mark.asyncio
async def test_worker_result_to_dict(tmp_path: Path) -> None:
    manager = make_manager(tmp_path)

    worker = create_worker(
        worker_id="worker_dict",
        worker_role="planner",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="规划测试",
        )
    )

    data = result.to_dict()

    assert data["worker_id"] == "worker_dict"
    assert data["worker_role"] == "planner"
    assert data["agent_name"] == "planner"
    assert data["success"] is True
    assert data["context_result"] is not None


@pytest.mark.asyncio
async def test_worker_busy_error(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return {
            "content": "slow done",
            "metadata": {},
        }

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    worker = create_worker(
        worker_id="worker_busy",
        worker_role="planner",
        manager=manager,
        workspace_path=tmp_path,
    )

    task = asyncio.create_task(
        worker.execute(
            WorkerRunRequest(
                task="慢任务",
            )
        )
    )

    for _ in range(20):
        if worker.is_busy:
            break
        await asyncio.sleep(0.01)

    assert worker.is_busy

    with pytest.raises(WorkerBusyError):
        await worker.execute(
            WorkerRunRequest(
                task="另一个任务",
            )
        )

    result = await task

    assert result.success is True


@pytest.mark.asyncio
async def test_worker_handles_subagent_failure(tmp_path: Path) -> None:
    async def failing_llm(messages, *, tools=None, metadata=None):
        raise RuntimeError("llm failed")

    manager = create_default_subagent_manager(
        llm=failing_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    worker = create_worker(
        worker_id="worker_fail",
        worker_role="debugger",
        manager=manager,
        workspace_path=tmp_path,
    )

    result = await worker.execute(
        WorkerRunRequest(
            task="这个任务会失败",
        )
    )

    assert result.success is False
    assert result.status == WorkerStatus.FAILED
    assert result.error is not None
    assert worker.status == WorkerStatus.IDLE


@pytest.mark.asyncio
async def test_worker_cancel_current_task_backed_run(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return {
            "content": "too late",
            "metadata": {},
        }

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    worker = create_worker(
        worker_id="worker_cancel",
        worker_role="debugger",
        manager=manager,
        workspace_path=tmp_path,
    )

    running = asyncio.create_task(
        worker.execute(
            WorkerRunRequest(
                task="慢速调试任务",
                execution_mode=WorkerExecutionMode.TASK,
                wait=True,
            )
        )
    )

    for _ in range(50):
        if worker.current_task_record_id:
            break
        await asyncio.sleep(0.01)

    cancelled = await worker.cancel_current(
        reason="user cancelled",
    )

    result = await running

    assert cancelled is True
    assert result.status == WorkerStatus.CANCELLED
    assert result.task_record is not None
    assert result.task_record.status == TaskStatus.CANCELLED