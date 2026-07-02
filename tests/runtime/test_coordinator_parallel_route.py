from __future__ import annotations

import asyncio

import pytest

from pywork.runtime.graph import (
    AgentGraphRunner,
    create_default_agent_graph_state,
    detect_coordinator_parallel_tool_call,
)
from pywork.subagents.manager import create_default_subagent_manager
from pywork.tools.registry import create_default_registry


def test_detect_coordinator_parallel_tool_call() -> None:
    registry = create_default_registry()

    data = create_default_agent_graph_state(
        user_input="""
把这三个任务并行跑：
1. 规划实现方案
2. 审查代码
3. 运行测试
""".strip(),
        registry=registry,
        config={},
        metadata={},
    )

    call = detect_coordinator_parallel_tool_call(data)

    assert call is not None
    assert call.tool_name == "coordinator"
    assert call.arguments["action"] == "run"
    assert call.arguments["strategy"] == "parallel"
    assert call.arguments["execution_mode"] == "task"
    assert call.arguments["wait"] is True
    assert len(call.arguments["steps"]) == 3

    agent_names = [
        step["agent_name"]
        for step in call.arguments["steps"]
    ]

    assert agent_names == [
        "planner",
        "reviewer",
        "verifier",
    ]


@pytest.mark.asyncio
async def test_runtime_routes_parallel_request_to_coordinator_tool(tmp_path) -> None:
    running_count = 0
    max_running_count = 0

    async def fake_llm(messages, *, tools=None, metadata=None):
        nonlocal running_count
        nonlocal max_running_count

        running_count += 1
        max_running_count = max(max_running_count, running_count)

        await asyncio.sleep(0.05)

        running_count -= 1

        return f"{metadata['agent_name']} done"

    registry = create_default_registry()
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=registry.list_definitions(),
        workspace_path=tmp_path,
    )

    runner = AgentGraphRunner(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            },
            "permissions": {
                "mode": "default",
            },
            "agent": {
                "max_iterations": 5,
                "max_context_messages": 20,
            },
        },
        runtime_objects={
            "subagent_manager": manager,
            "task_manager": manager.task_manager,
        },
    )

    state = await runner.arun(
        """
把这三个任务并行跑：
1. 规划实现方案
2. 审查代码
3. 运行测试
""".strip(),
        metadata={
            "subagent_manager": manager,
            "task_manager": manager.task_manager,
        },
    )

    last_message = state.get_last_message()

    assert last_message is not None
    assert "Tool `coordinator` result" in last_message.content
    assert "Coordinator finished 3 worker task(s)" in last_message.content

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