from __future__ import annotations

import pytest

from pywork.runtime.events import RuntimeEventBus, RuntimeEventType
from pywork.runtime.controller import RuntimeController
from pywork.runtime.graph import (
    AgentGraphRunner,
    append_observation_node,
    build_llm_messages,
    create_default_agent_graph_state,
    execute_tool_node,
    extract_glob_file_read_paths,
)
from pywork.runtime.state import create_agent_state
from pywork.state.app_state import create_app_state
from pywork.schemas.tool_schema import ToolResult, create_tool_call
from pywork.tools.registry import create_default_registry


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        ("file_read", {"path": "README.md"}),
        ("glob", {"pattern": "src/pywork/tools/*.py"}),
        ("grep", {"pattern": "async def", "path": "src/pywork"}),
    ],
)
async def test_runtime_graph_executes_default_registry_tools(
    tool_name: str,
    arguments: dict[str, str],
) -> None:
    tool_call = create_tool_call(
        tool_name=tool_name,
        arguments=arguments,
    )
    data = create_default_agent_graph_state(
        user_input="test",
        registry=create_default_registry(),
        config={
            "workspace": {
                "path": ".",
                "project_root": ".",
            }
        },
    )
    data["tool_call"] = tool_call
    data["parsed_tool_call"] = tool_call
    data["has_tool_call"] = True

    data = await execute_tool_node(data)
    data = append_observation_node(data)
    state = data["agent_state"]

    assert data["route_reason"] == "tool_result_observed_continue_to_llm"
    assert data["graph_route"] == "continue"
    assert data["awaiting_final_response"] is True
    assert state.tool_results
    assert state.tool_results[-1].tool_name == tool_name
    assert state.tool_results[-1].success is True
    assert state.get_last_message() is not None
    assert f"`{tool_name}`" in state.get_last_message().content
    assert "Now answer the original user request" in state.get_last_message().content


@pytest.mark.asyncio
async def test_agent_graph_runner_emits_tool_runtime_events() -> None:
    event_bus = RuntimeEventBus()
    runner = AgentGraphRunner(
        event_bus=event_bus,
        emit_events=True,
        config={
            "workspace": {
                "path": ".",
                "project_root": ".",
            },
        },
    )

    state = await runner.arun("/tool echo hello")
    event_types = [event.event_type for event in event_bus.history()]

    assert state.status.value == "finished"
    assert state.iteration == 2
    assert state.get_last_message() is not None
    assert state.get_last_message().role == "assistant"
    assert "hello" in state.get_last_message().content
    assert RuntimeEventType.LIFECYCLE in event_types
    assert RuntimeEventType.STATUS in event_types
    assert RuntimeEventType.MESSAGE in event_types
    assert RuntimeEventType.TOOL_CALL in event_types
    assert RuntimeEventType.TOOL_RESULT in event_types
    assert RuntimeEventType.CHECKPOINT in event_types


@pytest.mark.asyncio
async def test_agent_graph_runner_emits_finished_for_plain_response() -> None:
    event_bus = RuntimeEventBus()
    runner = AgentGraphRunner(
        event_bus=event_bus,
        emit_events=True,
    )

    state = await runner.arun("hello")
    lifecycle_events = [
        event
        for event in event_bus.history()
        if event.event_type == RuntimeEventType.LIFECYCLE
    ]

    assert state.status.value == "finished"
    assert lifecycle_events
    assert lifecycle_events[-1].lifecycle == "finished"


@pytest.mark.asyncio
async def test_agent_graph_routes_known_file_read_requests_to_file_read() -> None:
    runner = AgentGraphRunner(
        config={
            "workspace": {
                "path": ".",
                "project_root": ".",
            },
            "agent": {
                "max_iterations": 5,
            },
        },
    )

    state = await runner.arun("读一下 README.md，简单总结它的内容。")

    assert state.status.value == "finished"
    assert state.tool_calls
    assert state.tool_calls[0].tool_name == "file_read"
    assert state.tool_calls[0].arguments["path"] == "README.md"
    assert state.get_last_message() is not None
    assert state.get_last_message().role == "assistant"


@pytest.mark.asyncio
async def test_agent_graph_resolves_filename_with_glob_then_reads_file() -> None:
    runner = AgentGraphRunner(
        config={
            "workspace": {
                "path": ".",
                "project_root": ".",
            },
            "agent": {
                "max_iterations": 8,
            },
        },
    )

    state = await runner.arun("Read messages.py and summarize it.")
    tool_names = [call.tool_name for call in state.tool_calls]

    assert state.status.value == "finished"
    assert tool_names[0] == "glob"
    assert "file_read" in tool_names
    assert any(
        call.arguments.get("path", "").endswith("messages.py")
        for call in state.tool_calls
        if call.tool_name == "file_read"
    )
    assert state.get_last_message() is not None
    assert state.get_last_message().role == "assistant"


@pytest.mark.asyncio
async def test_agent_graph_reads_multiple_files_from_directory_request() -> None:
    runner = AgentGraphRunner(
        config={
            "workspace": {
                "path": ".",
                "project_root": ".",
            },
            "agent": {
                "max_iterations": 12,
            },
        },
    )

    state = await runner.arun("Read files under tests directory and summarize them.")
    file_read_calls = [
        call
        for call in state.tool_calls
        if call.tool_name == "file_read"
    ]

    assert state.status.value == "finished"
    assert state.tool_calls[0].tool_name == "glob"
    assert len(file_read_calls) >= 2
    assert all(
        str(call.arguments.get("path", "")).startswith("tests/")
        for call in file_read_calls
    )
    assert state.get_last_message() is not None
    assert state.get_last_message().role == "assistant"


def test_extract_glob_file_read_paths_does_not_apply_batch_cap() -> None:
    tool_call = create_tool_call(
        tool_name="glob",
        arguments={
            "pattern": "docs/**/*.md",
        },
    )
    result = ToolResult.success_result(
        call=tool_call,
        content="matched files",
        data={
            "matches": [
                {
                    "kind": "file",
                    "relative_path": f"docs/file_{index}.md",
                }
                for index in range(12)
            ],
        },
    )

    paths = extract_glob_file_read_paths(result)

    assert len(paths) == 12
    assert paths[0] == "docs/file_0.md"
    assert paths[-1] == "docs/file_11.md"


@pytest.mark.asyncio
async def test_runtime_controller_stream_yields_tool_events_and_result() -> None:
    controller = RuntimeController(
        app_state=create_app_state(
            config={
                "permissions": {
                    "mode": "default",
                },
                "agent": {
                    "max_iterations": 5,
                    "max_context_messages": 20,
                },
            }
        )
    )

    events = [
        event
        async for event in controller.stream("/tool echo hello from stream")
    ]
    result = controller.get_last_stream_result()

    assert result is not None
    assert result.success is True
    assert result.metadata["run_id"]
    assert {event.run_id for event in events} == {result.metadata["run_id"]}
    assert RuntimeEventType.TOOL_CALL in {event.event_type for event in events}
    assert RuntimeEventType.TOOL_RESULT in {event.event_type for event in events}
    assert events[-1].event_type == RuntimeEventType.LIFECYCLE
    assert events[-1].lifecycle == "finished"


def test_build_llm_messages_keeps_multi_turn_history() -> None:
    state = create_agent_state(
        system_prompt="old prompt",
        max_iterations=5,
    )
    state.add_user_message("first user")
    state.add_assistant_message("first assistant")
    state.add_user_message("second user")

    messages = build_llm_messages(
        {
            "agent_state": state,
            "config": {
                "agent": {
                    "max_context_messages": 2,
                },
                "llm": {
                    "system_prompt": "fresh prompt",
                },
            },
        }
    )

    assert [message.role for message in messages] == [
        "system",
        "assistant",
        "user",
    ]
    assert messages[0].content == "fresh prompt"
    assert messages[1].content == "first assistant"
    assert messages[2].content == "second user"


@pytest.mark.asyncio
async def test_runtime_controller_reuses_agent_state_between_turns() -> None:
    controller = RuntimeController(
        app_state=create_app_state(
            config={
                "permissions": {
                    "mode": "default",
                },
                "agent": {
                    "max_iterations": 5,
                    "max_context_messages": 20,
                },
                "llm": {
                    "fallback_to_mock": True,
                },
            }
        )
    )

    await controller.arun("/tool echo first turn memory")
    await controller.arun("what did I say before?")

    messages = controller.engine.agent_state.messages
    contents = [message.content for message in messages]

    assert any("first turn memory" in content for content in contents)
    assert any("what did I say before?" in content for content in contents)
    assert len(messages) >= 4
