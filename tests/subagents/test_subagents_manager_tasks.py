from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pywork.subagents.manager import (
    SubAgentTaskRequest,
    create_default_subagent_manager,
)
from pywork.tasks.task import TaskStatus
from pywork.tasks.task_storage import SQLiteTaskStorage


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


@pytest.mark.asyncio
async def test_subagent_manager_creates_agent_task(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    record = await manager.create_agent_task(
        "planner",
        "plan task integration",
        parent_task_id="root_task",
        max_retries=1,
    )

    assert record.id.startswith("task_")
    assert record.status == TaskStatus.PENDING
    assert record.agent_id == "planner"
    assert record.parent_id == "root_task"
    assert record.task_type.value == "subagent"
    assert record.payload["agent_name"] == "planner"
    assert record.payload["task"] == "plan task integration"
    assert record.max_retries == 1


@pytest.mark.asyncio
async def test_subagent_manager_run_agent_task_wait_true(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    record = await manager.run_agent_task(
        "planner",
        "plan task integration",
        wait=True,
    )

    assert record.status == TaskStatus.SUCCEEDED
    assert record.result is not None
    assert record.result.success is True
    assert record.result.value["name"] == "planner"
    assert "agent=planner" in record.result.value["content"]


@pytest.mark.asyncio
async def test_subagent_manager_run_agent_task_wait_false(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return {
            "content": f"agent={metadata['agent_name']} slow done",
            "metadata": {},
        }

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    execution = await manager.run_agent_task(
        "debugger",
        "debug slowly",
        wait=False,
    )

    assert execution.task_id in manager.task_manager.get_active_task_ids()

    record = await execution.wait()

    assert record.status == TaskStatus.SUCCEEDED
    assert record.result is not None
    assert record.result.value["name"] == "debugger"


@pytest.mark.asyncio
async def test_subagent_manager_cancel_agent_task(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return "too late"

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    execution = await manager.run_agent_task(
        "debugger",
        "debug slowly",
        wait=False,
    )

    cancelled = await manager.cancel_agent_task(
        execution.task_id,
        reason="user cancelled",
    )

    assert cancelled.status == TaskStatus.CANCELLED
    assert cancelled.error == "user cancelled"
    assert manager.task_manager.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_subagent_manager_retry_agent_task(tmp_path: Path) -> None:
    attempts = 0

    async def flaky_llm(messages, *, tools=None, metadata=None):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            raise RuntimeError("first failure")

        return {
            "content": "second attempt ok",
            "metadata": {},
        }

    manager = create_default_subagent_manager(
        llm=flaky_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    record = await manager.run_agent_task(
        "verifier",
        "verify flaky task",
        max_retries=1,
        wait=True,
    )

    assert record.status == TaskStatus.FAILED
    assert record.can_retry is True
    assert attempts == 1

    retried = await manager.retry_agent_task(
        record.id,
        wait=True,
    )

    assert retried.status == TaskStatus.SUCCEEDED
    assert retried.result is not None
    assert retried.result.success is True
    assert "second attempt ok" in retried.result.value["content"]
    assert attempts == 2


@pytest.mark.asyncio
async def test_subagent_manager_list_agent_tasks(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    planner = await manager.run_agent_task(
        "planner",
        "plan",
        wait=True,
    )
    reviewer = await manager.run_agent_task(
        "reviewer",
        "review",
        wait=True,
    )

    planner_tasks = await manager.list_agent_tasks(
        agent_id="planner",
    )

    assert [
        task.id
        for task in planner_tasks
    ] == [
        planner.id,
    ]

    succeeded = await manager.list_agent_tasks(
        status="succeeded",
    )

    assert {
        task.id
        for task in succeeded
    } == {
        planner.id,
        reviewer.id,
    }


@pytest.mark.asyncio
async def test_subagent_manager_watch_agent_task(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return {
            "content": "watched ok",
            "metadata": {},
        }

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    execution = await manager.run_agent_task(
        "verifier",
        "watch verify",
        wait=False,
    )

    statuses = []

    async for record in manager.watch_agent_task(
        execution.task_id,
        poll_interval=0.01,
        timeout=1,
    ):
        statuses.append(record.status)

    assert TaskStatus.RUNNING in statuses
    assert statuses[-1] == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_subagent_manager_run_many_agent_tasks(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    records = await manager.run_many_agent_tasks(
        [
            SubAgentTaskRequest(
                agent_name="planner",
                task="plan",
            ),
            SubAgentTaskRequest(
                agent_name="reviewer",
                task="review",
            ),
            SubAgentTaskRequest(
                agent_name="verifier",
                task="verify",
            ),
        ],
        concurrent=True,
        max_concurrency=2,
        wait=True,
    )

    assert [
        record.status
        for record in records
    ] == [
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
        TaskStatus.SUCCEEDED,
    ]

    assert [
        record.agent_id
        for record in records
    ] == [
        "planner",
        "reviewer",
        "verifier",
    ]


@pytest.mark.asyncio
async def test_subagent_manager_tasks_with_sqlite_storage(tmp_path: Path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    from pywork.tasks.task_manager import create_task_manager

    task_manager = create_task_manager(
        storage=storage,
    )

    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
        task_manager=task_manager,
    )

    record = await manager.run_agent_task(
        "planner",
        "plan with storage",
        wait=True,
    )

    loaded = storage.require_task(record.id)

    assert loaded.status == TaskStatus.SUCCEEDED
    assert loaded.agent_id == "planner"
    assert loaded.result is not None
    assert loaded.result.value["name"] == "planner"

    events = storage.list_events(task_id=record.id)

    assert events

    storage.close()