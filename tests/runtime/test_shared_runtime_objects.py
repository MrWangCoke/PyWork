from __future__ import annotations

from pywork.runtime.engine import RuntimeEngine
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    create_graph_tool_context,
)
from pywork.runtime.shared_objects import normalize_subagent_llm_messages
from pywork.schemas.tool_schema import create_tool_call
from pywork.teams.mailbox import AgentMailbox
from pywork.tools.registry import create_default_registry
from pywork.tools.send_message import SendMessageTool


def test_runtime_engine_injects_shared_objects(tmp_path) -> None:
    registry = create_default_registry()
    engine = RuntimeEngine(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        },
    )

    metadata = engine.runtime_metadata

    assert metadata["task_manager"] is engine.task_manager
    assert metadata["subagent_manager"] is engine.subagent_manager
    assert metadata["mailbox"] is engine.mailbox
    assert metadata["team_registry"] is engine.team_registry
    assert metadata["tool_registry"] is registry
    assert metadata["registry"] is registry
    assert isinstance(metadata["mailbox"], AgentMailbox)


def test_agent_tool_uses_shared_subagent_manager(tmp_path) -> None:
    registry = create_default_registry()
    engine = RuntimeEngine(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
            }
        },
    )

    agent_tool = registry.get("agent")

    assert agent_tool is not None
    assert getattr(agent_tool, "manager", None) is engine.subagent_manager
    assert getattr(agent_tool, "_fallback_runtime", None) is None


def test_subagent_manager_uses_runtime_llm_config(tmp_path) -> None:
    engine = RuntimeEngine(
        config={
            "workspace": {
                "path": str(tmp_path),
            },
            "llm": {
                "default_provider": "qwen",
                "providers": {
                    "qwen": {
                        "provider": "qwen",
                        "api_format": "openai_compatible",
                        "model": "qwen3.6-flash",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "DASHSCOPE_API_KEY",
                    }
                },
            },
        },
    )

    assert engine.subagent_manager.llm is not None


def test_subagent_llm_message_normalization_ignores_empty_tool_call_id() -> None:
    messages = normalize_subagent_llm_messages(
        [
            {
                "role": "system",
                "content": "review this file",
                "tool_call_id": None,
            },
            {
                "role": "user",
                "content": "ok",
                "name": None,
                "tool_call_id": None,
            },
        ]
    )

    assert [message.role for message in messages] == ["system", "user"]
    assert messages[0].content == "review this file"


def test_subagent_llm_message_normalization_converts_invalid_tool_message() -> None:
    messages = normalize_subagent_llm_messages(
        [
            {
                "role": "tool",
                "name": "agent",
                "content": "Reviewer finished with status failed.",
            }
        ]
    )

    assert len(messages) == 1
    assert messages[0].role == "user"
    assert "Tool observation from agent" in messages[0].content
    assert "Reviewer finished" in messages[0].content


def test_graph_tool_context_contains_shared_objects(tmp_path) -> None:
    registry = create_default_registry()
    engine = RuntimeEngine(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            },
            "permissions": {
                "mode": "default",
            },
        },
    )

    data = create_default_agent_graph_state(
        user_input="hello",
        registry=registry,
        config=engine.config,
        agent_state=engine.agent_state,
        metadata=engine.runtime_metadata,
    )

    data["tool_registry"] = registry
    data["registry"] = registry
    data["run_id"] = "run_test"
    data["session_id"] = "session_test"
    data["event_bus"] = engine.event_bus
    data["emit_events"] = False
    data["tool_definitions"] = registry.list_definitions()

    context = create_graph_tool_context(data)

    assert context.workspace_path == str(tmp_path.resolve())
    assert context.project_root == str(tmp_path.resolve())
    assert context.permission_mode == "default"
    assert context.session_id == "session_test"

    assert context.metadata["task_manager"] is engine.task_manager
    assert context.metadata["subagent_manager"] is engine.subagent_manager
    assert context.metadata["mailbox"] is engine.mailbox
    assert context.metadata["team_registry"] is engine.team_registry
    assert context.metadata["tool_registry"] is registry
    assert context.metadata["registry"] is registry
    assert context.metadata["agent_state"] is engine.agent_state
    assert context.metadata["run_id"] == "run_test"
    assert context.metadata["session_id"] == "session_test"


def test_send_message_can_use_runtime_injected_mailbox(tmp_path) -> None:
    registry = create_default_registry()
    engine = RuntimeEngine(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
            }
        },
    )

    data = create_default_agent_graph_state(
        user_input="send message",
        registry=registry,
        config=engine.config,
        agent_state=engine.agent_state,
        metadata=engine.runtime_metadata,
    )

    data["tool_registry"] = registry
    data["registry"] = registry
    data["run_id"] = "run_test"
    data["session_id"] = "session_test"
    data["event_bus"] = engine.event_bus
    data["emit_events"] = False
    data["tool_definitions"] = registry.list_definitions()

    context = create_graph_tool_context(data)
    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "send",
            "sender_id": "agent_a",
            "recipient_id": "agent_b",
            "content": "hello",
        },
    )

    import asyncio

    result = asyncio.run(
        tool.execute(
            call,
            context,
        )
    )

    assert result.success is True
    assert len(engine.mailbox.get_inbox("agent_b")) == 1
