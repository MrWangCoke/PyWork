from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.subagents.manager import create_default_subagent_manager
from pywork.subagents.router import LLMSubAgentRouter
from pywork.tools.agent_tool import AgentTool
from pywork.tools.tool import ToolExecutionContext


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


async def router_llm(messages, *, tools=None, metadata=None):
    return json.dumps(
        {
            "agent_name": "planner",
            "task": "Plan the SubAgent tool implementation.",
            "reason": "The task asks for implementation planning.",
            "confidence": 0.9,
            "kind": "single",
            "pipeline": [],
            "missing_information": [],
            "metadata": {},
        },
        ensure_ascii=False,
    )


def make_tool(tmp_path: Path) -> AgentTool:
    manager = create_default_subagent_manager(
        llm=agent_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )
    router = LLMSubAgentRouter(
        manager=manager,
        llm=router_llm,
    )

    return AgentTool(
        manager=manager,
        router=router,
    )


def make_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
        metadata={
            "parent_messages": [
                {
                    "role": "user",
                    "content": "Parent context from main agent.",
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_agent_tool_lists_agents(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "list_agents",
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success

    names = [
        item["name"]
        for item in result.data["agents"]
    ]

    assert names == [
        "debugger",
        "general",
        "planner",
        "reviewer",
        "verifier",
    ]


@pytest.mark.asyncio
async def test_agent_tool_describes_agent(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "describe_agent",
            "agent_name": "planner",
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success
    assert result.data["agent"]["name"] == "planner"
    assert result.data["instance"]["role"] == "planner"
    assert result.data["instance"]["permission_mode"] == "readonly"


@pytest.mark.asyncio
async def test_agent_tool_runs_specific_agent(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "run",
            "agent_name": "reviewer",
            "task": "Review the agent tool implementation.",
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success
    assert result.data["result"]["name"] == "reviewer"
    assert result.data["result"]["status"] == "completed"
    assert "agent=reviewer" in result.content
    assert "mode=readonly" in result.content


@pytest.mark.asyncio
async def test_agent_tool_routes_task(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "route",
            "task": "帮我规划 agent_tool.py 的实现",
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success
    assert result.data["route"]["agent_name"] == "planner"
    assert result.data["route"]["confidence"] == 0.9


@pytest.mark.asyncio
async def test_agent_tool_route_and_run(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "route_and_run",
            "task": "帮我规划 agent_tool.py 的实现",
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success
    assert result.data["route"]["agent_name"] == "planner"
    assert result.data["result"]["name"] == "planner"
    assert result.data["result"]["status"] == "completed"


@pytest.mark.asyncio
async def test_agent_tool_run_many(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)
    call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "run_many",
            "tasks": [
                {
                    "agent_name": "planner",
                    "task": "Plan the change.",
                },
                {
                    "agent_name": "reviewer",
                    "task": "Review the change.",
                },
                {
                    "agent_name": "verifier",
                    "task": "Verify the change.",
                },
            ],
        },
    )

    result = await tool.run(
        call,
        make_context(tmp_path),
    )

    assert result.success

    names = [
        item["name"]
        for item in result.data["results"]
    ]

    assert names == [
        "planner",
        "reviewer",
        "verifier",
    ]


@pytest.mark.asyncio
async def test_agent_tool_history(tmp_path: Path) -> None:
    tool = make_tool(tmp_path)

    run_call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "run",
            "agent_name": "planner",
            "task": "Plan something.",
        },
    )

    await tool.run(
        run_call,
        make_context(tmp_path),
    )

    history_call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "history",
            "history_limit": 5,
        },
    )

    result = await tool.run(
        history_call,
        make_context(tmp_path),
    )

    assert result.success
    assert len(result.data["history"]) == 1
    assert result.data["history"][0]["agent_name"] == "planner"


@pytest.mark.asyncio
async def test_agent_tool_abort_all(tmp_path: Path) -> None:
    async def slow_agent_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return "finished too late"

    manager = create_default_subagent_manager(
        llm=slow_agent_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )
    tool = AgentTool(
        manager=manager,
    )

    run_call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "run",
            "agent_name": "debugger",
            "task": "Debug slow task.",
        },
    )

    task = asyncio.create_task(
        tool.run(
            run_call,
            make_context(tmp_path),
        )
    )

    for _ in range(20):
        if manager.get_active_runs():
            break
        await asyncio.sleep(0.01)

    assert manager.get_active_runs()

    abort_call = create_tool_call(
        tool_name="agent",
        arguments={
            "action": "abort_all",
            "reason": "user cancelled",
        },
    )

    abort_result = await tool.run(
        abort_call,
        make_context(tmp_path),
    )

    assert abort_result.success
    assert abort_result.data["aborted_count"] == 1

    run_result = await task

    assert run_result.success
    assert run_result.data["result"]["status"] == "aborted"