from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pywork.subagents.base import SubAgentContext, SubAgentStatus
from pywork.subagents.manager import (
    SubAgentManager,
    SubAgentManagerEventType,
    SubAgentNotFoundError,
    SubAgentTaskRequest,
    create_default_subagent_manager,
)


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
            "description": "Find files by glob pattern",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search text in files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_edit",
            "description": "Edit file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run bash command",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Run PowerShell command",
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
        "metadata": {
            "tool_count": len(tool_names),
        },
    }


def test_default_manager_registers_all_agents(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    names = [
        item["name"]
        for item in manager.list_agents()
    ]

    assert names == [
        "debugger",
        "general",
        "planner",
        "reviewer",
        "verifier",
    ]

    assert manager.resolve_agent_name("plan") == "planner"
    assert manager.resolve_agent_name("review") == "reviewer"
    assert manager.resolve_agent_name("debug") == "debugger"
    assert manager.resolve_agent_name("verify") == "verifier"
    assert manager.resolve_agent_name("default") == "general"


def test_unknown_agent_raises(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    with pytest.raises(SubAgentNotFoundError):
        manager.resolve_agent_name("missing")


@pytest.mark.asyncio
async def test_run_single_agent(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    result = await manager.run_agent(
        "planner",
        "plan manager implementation",
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "planner"
    assert "agent=planner" in result.content
    assert "mode=readonly" in result.content
    assert "file_read" in result.content
    assert "bash" not in result.content

    history = manager.get_history()

    assert len(history) == 1
    assert history[0]["agent_name"] == "planner"
    assert history[0]["status"] == "completed"
    assert history[0]["duration_ms"] is not None


@pytest.mark.asyncio
async def test_run_agent_with_alias(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    result = await manager.run_agent(
        "debug",
        "debug failing test",
    )

    assert result.success
    assert result.name == "debugger"
    assert "bash" in result.content
    assert "powershell" in result.content


@pytest.mark.asyncio
async def test_run_agent_with_context_isolation(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    parent_messages = [
        {
            "role": "user",
            "content": "Parent context should be copied.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    context = SubAgentContext(
        task="review context",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await manager.run_agent(
        "reviewer",
        "review context",
        context=context,
    )

    assert result.success
    assert result.name == "reviewer"

    assert parent_messages == [
        {
            "role": "user",
            "content": "Parent context should be copied.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    contents = [
        message.content
        for message in result.state.messages
    ]

    assert "Parent context should be copied." in contents


@pytest.mark.asyncio
async def test_run_sequence(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    results = await manager.run_sequence(
        [
            SubAgentTaskRequest(
                agent_name="planner",
                task="plan feature",
            ),
            SubAgentTaskRequest(
                agent_name="reviewer",
                task="review feature",
            ),
            SubAgentTaskRequest(
                agent_name="verifier",
                task="verify feature",
            ),
        ]
    )

    assert [
        result.name
        for result in results
    ] == [
        "planner",
        "reviewer",
        "verifier",
    ]

    assert all(result.success for result in results)
    assert len(manager.get_history()) == 3


@pytest.mark.asyncio
async def test_run_parallel_preserves_result_order(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    results = await manager.run_parallel(
        [
            SubAgentTaskRequest(
                agent_name="planner",
                task="plan feature",
            ),
            SubAgentTaskRequest(
                agent_name="debugger",
                task="debug feature",
            ),
            SubAgentTaskRequest(
                agent_name="verifier",
                task="verify feature",
            ),
        ],
        max_concurrency=2,
    )

    assert [
        result.name
        for result in results
    ] == [
        "planner",
        "debugger",
        "verifier",
    ]

    assert all(result.success for result in results)
    assert len(manager.get_history()) == 3


@pytest.mark.asyncio
async def test_event_handlers_receive_started_and_finished_events(tmp_path: Path) -> None:
    events = []

    async def handler(event):
        events.append(event)

    manager = create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )
    manager.add_event_handler(handler)

    result = await manager.run_agent(
        "planner",
        "plan events",
    )

    assert result.success

    event_types = [
        event.event_type
        for event in events
    ]

    assert SubAgentManagerEventType.STARTED in event_types
    assert SubAgentManagerEventType.COMPLETED in event_types


@pytest.mark.asyncio
async def test_abort_all_marks_running_agent_aborted(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return "finished too late"

    manager = create_default_subagent_manager(
        llm=slow_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    task = asyncio.create_task(
        manager.run_agent(
            "debugger",
            "debug slow task",
        )
    )

    for _ in range(20):
        if manager.get_active_runs():
            break
        await asyncio.sleep(0.01)

    assert manager.get_active_runs()

    count = manager.abort_all("user cancelled all")
    assert count == 1

    result = await task

    assert result.aborted
    assert result.status == SubAgentStatus.ABORTED
    assert result.error == "user cancelled all"
    assert manager.get_active_runs() == []