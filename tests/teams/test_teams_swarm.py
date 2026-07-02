from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pywork.subagents.manager import create_default_subagent_manager
from pywork.teams.swarm import (
    SwarmEventType,
    SwarmPlan,
    SwarmRunRequest,
    SwarmStatus,
    SwarmStrategy,
    SwarmTaskStep,
    create_swarm,
)
from pywork.teams.team import TeamTaskStatus, create_team
from pywork.teams.teammate import TeammateExecutionMode


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
    return {
        "content": f"agent={metadata['agent_name']} task={messages[-1]['content']}",
        "metadata": {},
    }


async def planning_llm(messages, *, tools=None, metadata=None):
    return json.dumps(
        {
            "task": "实现 Team / Swarm 系统",
            "strategy": "sequential",
            "summary": "先规划，再审查，再验证。",
            "steps": [
                {
                    "step_id": "swarm_step_01",
                    "title": "规划 Swarm",
                    "description": "规划 teams/swarm.py 的实现。",
                    "role": "planner",
                    "priority": "high",
                    "depends_on": [],
                    "metadata": {
                        "phase": "plan",
                    },
                },
                {
                    "step_id": "swarm_step_02",
                    "title": "审查 Swarm",
                    "description": "审查 teams/swarm.py 的实现。",
                    "role": "reviewer",
                    "priority": "normal",
                    "depends_on": [
                        "swarm_step_01",
                    ],
                    "metadata": {
                        "phase": "review",
                    },
                },
                {
                    "step_id": "swarm_step_03",
                    "title": "验证 Swarm",
                    "description": "验证 teams/swarm.py 的测试。",
                    "role": "verifier",
                    "priority": "normal",
                    "depends_on": [
                        "swarm_step_02",
                    ],
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


def make_team(tmp_path: Path, llm=agent_llm):
    manager = make_manager(tmp_path, llm=llm)

    team = create_team(
        team_id="team_swarm_test",
        name="Swarm Test Team",
        manager=manager,
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    team.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )
    team.create_teammate(
        teammate_id="verifier_1",
        role="verifier",
    )
    team.create_teammate(
        teammate_id="general_1",
        role="general",
    )

    return team


@pytest.mark.asyncio
async def test_swarm_creates_llm_plan(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        swarm_id="swarm_test",
        team=team,
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    plan = await swarm.create_plan(
        SwarmRunRequest(
            task="实现 Team / Swarm 系统",
            workspace_path=tmp_path,
        )
    )

    assert plan.strategy == SwarmStrategy.SEQUENTIAL
    assert len(plan.steps) == 3
    assert plan.steps[0].role == "planner"
    assert plan.steps[1].role == "reviewer"
    assert plan.steps[2].role == "verifier"
    assert plan.metadata["source"] == "test_planner"


@pytest.mark.asyncio
async def test_swarm_runs_sequential_plan(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        swarm_id="swarm_seq",
        team=team,
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="实现 Team / Swarm 系统",
            workspace_path=tmp_path,
        )
    )

    assert result.success is True
    assert result.status == SwarmStatus.COMPLETED
    assert len(result.executions) == 3
    assert all(execution.success for execution in result.executions)
    assert "Swarm status: completed" in result.summary

    tasks = team.list_shared_tasks()

    assert len(tasks) == 3
    assert all(task.status == TeamTaskStatus.SUCCEEDED for task in tasks)


@pytest.mark.asyncio
async def test_swarm_runs_parallel_plan(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    plan = SwarmPlan(
        task="并行审查和验证",
        strategy=SwarmStrategy.PARALLEL,
        summary="review and verify can run independently",
        steps=[
            SwarmTaskStep(
                step_id="review",
                title="审查",
                description="审查 swarm.py。",
                role="reviewer",
            ),
            SwarmTaskStep(
                step_id="verify",
                title="验证",
                description="验证 swarm.py。",
                role="verifier",
            ),
        ],
    )

    swarm = create_swarm(
        swarm_id="swarm_parallel",
        team=team,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="并行审查和验证",
            workspace_path=tmp_path,
            plan=plan,
        )
    )

    assert result.success is True
    assert result.status == SwarmStatus.COMPLETED
    assert {
        execution.assigned_to
        for execution in result.executions
    } == {
        "reviewer_1",
        "verifier_1",
    }


@pytest.mark.asyncio
async def test_swarm_uses_fallback_plan(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        team=team,
        planning_llm=None,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="验证 teams/swarm.py 的测试",
            workspace_path=tmp_path,
            use_llm_planning=False,
        )
    )

    assert result.success is True
    assert len(result.plan.steps) == 1
    assert result.plan.steps[0].resolved_role == "verifier"
    assert result.executions[0].assigned_to == "verifier_1"


@pytest.mark.asyncio
async def test_swarm_accepts_caller_steps(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        team=team,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="调用方提供 steps",
            workspace_path=tmp_path,
            steps=[
                {
                    "step_id": "plan",
                    "title": "规划",
                    "description": "规划实现。",
                    "role": "planner",
                },
                {
                    "step_id": "review",
                    "title": "审查",
                    "description": "审查实现。",
                    "role": "reviewer",
                },
            ],
            strategy="sequential",
        )
    )

    assert result.success is True
    assert [
        execution.assigned_to
        for execution in result.executions
    ] == [
        "planner_1",
        "reviewer_1",
    ]


@pytest.mark.asyncio
async def test_swarm_marks_failed_when_teammate_fails(tmp_path: Path) -> None:
    async def failing_llm(messages, *, tools=None, metadata=None):
        raise RuntimeError("agent failed")

    team = make_team(tmp_path, llm=failing_llm)

    swarm = create_swarm(
        team=team,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="失败任务",
            workspace_path=tmp_path,
            steps=[
                {
                    "step_id": "debug",
                    "title": "调试失败",
                    "description": "分析失败。",
                    "role": "debugger",
                }
            ],
            use_llm_planning=False,
        )
    )

    assert result.success is False
    assert result.status == SwarmStatus.FAILED
    assert len(result.executions) == 1
    assert result.executions[0].status == TeamTaskStatus.FAILED


@pytest.mark.asyncio
async def test_swarm_emits_events(tmp_path: Path) -> None:
    events = []

    team = make_team(tmp_path)

    swarm = create_swarm(
        swarm_id="swarm_events",
        team=team,
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    swarm.add_event_handler(
        lambda event: events.append(event)
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="实现 Team / Swarm 系统",
            workspace_path=tmp_path,
        )
    )

    assert result.success is True

    event_types = [
        event.event_type
        for event in events
    ]

    assert SwarmEventType.STARTED in event_types
    assert SwarmEventType.PLAN_CREATED in event_types
    assert SwarmEventType.TASK_CREATED in event_types
    assert SwarmEventType.TASK_DISPATCHED in event_types
    assert SwarmEventType.TASK_COMPLETED in event_types
    assert SwarmEventType.COMPLETED in event_types


@pytest.mark.asyncio
async def test_swarm_result_to_dict(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        swarm_id="swarm_dict",
        team=team,
        planning_llm=planning_llm,
        workspace_path=tmp_path,
    )

    result = await swarm.run(
        SwarmRunRequest(
            task="实现 Team / Swarm 系统",
            workspace_path=tmp_path,
        )
    )

    data = result.to_dict()

    assert data["swarm_id"] == "swarm_dict"
    assert data["success"] is True
    assert data["status"] == "completed"
    assert data["plan"]["steps"]
    assert data["executions"]


@pytest.mark.asyncio
async def test_swarm_cancel_current_task_mode(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return {
            "content": "too late",
            "metadata": {},
        }

    team = make_team(tmp_path, llm=slow_llm)

    swarm = create_swarm(
        swarm_id="swarm_cancel",
        team=team,
        workspace_path=tmp_path,
    )

    running = asyncio.create_task(
        swarm.run(
            SwarmRunRequest(
                task="慢速 swarm 任务",
                workspace_path=tmp_path,
                steps=[
                    {
                        "step_id": "debug",
                        "title": "慢速调试",
                        "description": "执行慢速调试。",
                        "role": "debugger",
                    }
                ],
                use_llm_planning=False,
                teammate_execution_mode=TeammateExecutionMode.TASK,
                step_timeout_seconds=20,
            )
        )
    )

    for _ in range(100):
        if swarm.active_task_ids:
            break
        await asyncio.sleep(0.01)

    cancelled_count = await swarm.cancel_current(
        reason="user cancelled",
    )

    result = await running

    assert cancelled_count >= 1
    assert result.status == SwarmStatus.CANCELLED
    assert result.success is False


def test_swarm_to_dict(tmp_path: Path) -> None:
    team = make_team(tmp_path)

    swarm = create_swarm(
        swarm_id="swarm_info",
        team=team,
        workspace_path=tmp_path,
        metadata={
            "source": "test",
        },
    )

    data = swarm.to_dict()

    assert data["swarm_id"] == "swarm_info"
    assert data["team_id"] == team.team_id
    assert data["status"] == "idle"
    assert data["team"]["roster"]["member_count"] == 4
    assert data["metadata"]["source"] == "test"