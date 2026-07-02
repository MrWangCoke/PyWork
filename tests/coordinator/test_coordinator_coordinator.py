from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pywork.coordinator.coordinator import (
    CoordinatorEventType,
    CoordinatorPlan,
    CoordinatorPlanStrategy,
    CoordinatorRunRequest,
    CoordinatorStatus,
    CoordinatorTaskStep,
    create_coordinator,
)
from pywork.coordinator.worker import WorkerExecutionMode
from pywork.subagents.manager import create_default_subagent_manager


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


async def agent_llm(messages, *, tools=None, metadata=None):
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


async def planning_llm(messages, *, tools=None, metadata=None):
    return json.dumps(
        {
            "task": "实现 Coordinator 系统",
            "strategy": "sequential",
            "summary": "先规划，再验证。",
            "steps": [
                {
                    "step_id": "step_01",
                    "worker_role": "planner",
                    "agent_name": "planner",
                    "task": "规划 coordinator.py 的实现。",
                    "depends_on": [],
                    "context_profile_name": "planner",
                    "metadata": {
                        "phase": "plan",
                    },
                },
                {
                    "step_id": "step_02",
                    "worker_role": "verifier",
                    "agent_name": "verifier",
                    "task": "验证 coordinator.py 的测试。",
                    "depends_on": [
                        "step_01",
                    ],
                    "context_profile_name": "verifier",
                    "metadata": {
                        "phase": "verify",
                    },
                },
            ],
            "metadata": {
                "source": "test_planner",
            },
        },
        ensure_ascii=False,
    )


def make_manager(tmp_path: Path, llm=agent_llm):
    return create_default_subagent_manager(
        llm=llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )


@pytest.mark.asyncio
async def test_coordinator_creates_llm_plan(tmp_path: Path) -> None:
    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    plan = await coordinator.create_plan(
        CoordinatorRunRequest(
            task="实现 Coordinator 系统",
            workspace_path=tmp_path,
        )
    )

    assert plan.strategy == CoordinatorPlanStrategy.SEQUENTIAL
    assert len(plan.steps) == 2
    assert plan.steps[0].worker_role == "planner"
    assert plan.steps[1].worker_role == "verifier"
    assert plan.metadata["source"] == "test_planner"


@pytest.mark.asyncio
async def test_coordinator_runs_sequential_plan(tmp_path: Path) -> None:
    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="实现 Coordinator 系统",
            workspace_path=tmp_path,
            parent_messages=[
                {
                    "role": "user",
                    "content": "需要实现 context_modifier、worker、coordinator。",
                }
            ],
        )
    )

    assert result.success is True
    assert result.status == CoordinatorStatus.COMPLETED
    assert len(result.worker_results) == 2
    assert [
        item.agent_name
        for item in result.worker_results
    ] == [
        "planner",
        "verifier",
    ]
    assert "Coordinator status: completed" in result.summary


@pytest.mark.asyncio
async def test_coordinator_runs_parallel_steps(tmp_path: Path) -> None:
    plan = CoordinatorPlan(
        task="并行审查和验证",
        strategy=CoordinatorPlanStrategy.PARALLEL,
        summary="review and verify can run independently",
        steps=[
            CoordinatorTaskStep(
                step_id="step_01",
                worker_role="reviewer",
                agent_name="reviewer",
                task="审查 coordinator.py。",
            ),
            CoordinatorTaskStep(
                step_id="step_02",
                worker_role="verifier",
                agent_name="verifier",
                task="验证 coordinator.py。",
            ),
        ],
    )

    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="并行审查和验证",
            workspace_path=tmp_path,
            plan=plan,
        )
    )

    assert result.success is True
    assert result.status == CoordinatorStatus.COMPLETED
    assert {
        item.agent_name
        for item in result.worker_results
    } == {
        "reviewer",
        "verifier",
    }


@pytest.mark.asyncio
async def test_coordinator_uses_fallback_plan_without_planning_llm(tmp_path: Path) -> None:
    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        planning_llm=None,
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="验证 coordinator 的测试",
            workspace_path=tmp_path,
            use_llm_planning=False,
        )
    )

    assert result.success is True
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].worker_role == "verifier"
    assert result.worker_results[0].agent_name == "verifier"


@pytest.mark.asyncio
async def test_coordinator_accepts_caller_steps(tmp_path: Path) -> None:
    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="调用方提供步骤",
            workspace_path=tmp_path,
            steps=[
                {
                    "step_id": "plan",
                    "worker_role": "planner",
                    "task": "规划实现。",
                },
                {
                    "step_id": "review",
                    "worker_role": "reviewer",
                    "task": "审查实现。",
                },
            ],
            strategy="sequential",
        )
    )

    assert result.success is True
    assert [
        item.agent_name
        for item in result.worker_results
    ] == [
        "planner",
        "reviewer",
    ]


@pytest.mark.asyncio
async def test_coordinator_marks_failed_when_worker_fails(tmp_path: Path) -> None:
    async def failing_agent_llm(messages, *, tools=None, metadata=None):
        raise RuntimeError("worker llm failed")

    coordinator = create_coordinator(
        manager=make_manager(tmp_path, llm=failing_agent_llm),
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="这个任务会失败",
            workspace_path=tmp_path,
            steps=[
                {
                    "step_id": "debug",
                    "worker_role": "debugger",
                    "task": "分析失败。",
                }
            ],
            use_llm_planning=False,
        )
    )

    assert result.success is False
    assert result.status == CoordinatorStatus.FAILED
    assert len(result.worker_results) == 1
    assert result.worker_results[0].success is False


@pytest.mark.asyncio
async def test_coordinator_emits_events(tmp_path: Path) -> None:
    events = []

    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    coordinator.add_event_handler(
        lambda event: events.append(event)
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="实现 Coordinator 系统",
            workspace_path=tmp_path,
        )
    )

    assert result.success is True

    event_types = [
        event.event_type
        for event in events
    ]

    assert CoordinatorEventType.STARTED in event_types
    assert CoordinatorEventType.PLAN_CREATED in event_types
    assert CoordinatorEventType.STEP_STARTED in event_types
    assert CoordinatorEventType.STEP_COMPLETED in event_types
    assert CoordinatorEventType.COMPLETED in event_types


@pytest.mark.asyncio
async def test_coordinator_result_to_dict(tmp_path: Path) -> None:
    coordinator = create_coordinator(
        manager=make_manager(tmp_path),
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    result = await coordinator.run(
        CoordinatorRunRequest(
            task="实现 Coordinator 系统",
            workspace_path=tmp_path,
        )
    )

    data = result.to_dict()

    assert data["success"] is True
    assert data["status"] == "completed"
    assert data["plan"]["steps"]
    assert data["worker_results"]


@pytest.mark.asyncio
async def test_coordinator_cancel_current_task_mode(tmp_path: Path) -> None:
    async def slow_agent_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return {
            "content": "too late",
            "metadata": {},
        }

    coordinator = create_coordinator(
        manager=make_manager(tmp_path, llm=slow_agent_llm),
        workspace_path=tmp_path,
    )

    running = asyncio.create_task(
        coordinator.run(
            CoordinatorRunRequest(
                task="慢速调试任务",
                workspace_path=tmp_path,
                steps=[
                    {
                        "step_id": "debug",
                        "worker_role": "debugger",
                        "task": "执行慢速调试。",
                    }
                ],
                use_llm_planning=False,
                worker_execution_mode=WorkerExecutionMode.TASK,
            )
        )
    )

    for _ in range(80):
        if coordinator.active_workers:
            break
        await asyncio.sleep(0.01)

    cancelled_count = await coordinator.cancel_current(
        reason="user cancelled",
    )

    result = await running

    assert cancelled_count >= 1
    assert result.status == CoordinatorStatus.CANCELLED
    assert result.success is False