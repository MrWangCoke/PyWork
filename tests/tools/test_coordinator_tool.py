from __future__ import annotations

import asyncio

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.subagents.manager import create_default_subagent_manager
from pywork.tools.coordinator_tool import CoordinatorTool
from pywork.tools.tool import ToolExecutionContext


@pytest.mark.asyncio
async def test_coordinator_tool_runs_three_workers_in_parallel_task_mode(tmp_path) -> None:
    running_count = 0
    max_running_count = 0

    async def fake_llm(messages, *, tools=None, metadata=None):
        nonlocal running_count
        nonlocal max_running_count

        running_count += 1
        max_running_count = max(max_running_count, running_count)

        await asyncio.sleep(0.05)

        running_count -= 1

        return f"{metadata['agent_name']} finished"

    manager = create_default_subagent_manager(
        llm=fake_llm,
        workspace_path=tmp_path,
    )

    tool = CoordinatorTool()

    call = create_tool_call(
        "coordinator",
        {
            "action": "run",
            "strategy": "parallel",
            "execution_mode": "task",
            "wait": True,
            "max_concurrency": 3,
            "steps": [
                {
                    "agent_name": "planner",
                    "task": "规划实现方案",
                },
                {
                    "agent_name": "reviewer",
                    "task": "审查代码",
                },
                {
                    "agent_name": "verifier",
                    "task": "运行测试",
                },
            ],
        },
    )

    result = await tool.execute(
        call,
        ToolExecutionContext(
            workspace_path=str(tmp_path),
            metadata={
                "subagent_manager": manager,
                "task_manager": manager.task_manager,
            },
        ),
    )

    assert result.success is True
    assert result.data["worker_count"] == 3
    assert result.data["strategy"] == "parallel"
    assert result.data["execution_mode"] == "task"
    assert result.data["task_manager_visible"] is True

    # 至少出现两个 worker 同时运行，证明不是纯顺序。
    assert max_running_count >= 2

    task_records = await manager.list_agent_tasks()

    assert len(task_records) == 3

    agent_ids = {
        record.agent_id
        for record in task_records
    }

    assert {
        "planner",
        "reviewer",
        "verifier",
    }.issubset(agent_ids)


@pytest.mark.asyncio
async def test_coordinator_tool_infers_agents_from_string_steps(tmp_path) -> None:
    async def fake_llm(messages, *, tools=None, metadata=None):
        return f"{metadata['agent_name']} ok"

    manager = create_default_subagent_manager(
        llm=fake_llm,
        workspace_path=tmp_path,
    )

    tool = CoordinatorTool()

    call = create_tool_call(
        "coordinator",
        {
            "action": "run",
            "strategy": "parallel",
            "execution_mode": "task",
            "steps": [
                "规划实现方案",
                "审查代码",
                "运行测试",
            ],
        },
    )

    result = await tool.execute(
        call,
        ToolExecutionContext(
            workspace_path=str(tmp_path),
            metadata={
                "subagent_manager": manager,
            },
        ),
    )

    assert result.success is True

    task_records = await manager.list_agent_tasks()

    agent_ids = [
        record.agent_id
        for record in task_records
    ]

    assert agent_ids == [
        "planner",
        "reviewer",
        "verifier",
    ]