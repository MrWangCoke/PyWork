from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.base import (
    BaseSubAgent,
    SubAgentAbortSignal,
    SubAgentContext,
    SubAgentStatus,
    SubAgentToolScope,
    filter_tool_definitions_by_scope,
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
            "name": "file_write",
            "description": "Write file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run bash",
        },
    },
]


class DemoSubAgent(BaseSubAgent):
    name = "demo"
    role = "general"
    description = "Demo subagent"

    default_allowed_tools = frozenset(
        {
            "file_read",
            "grep",
        }
    )
    default_permission_mode = "readonly"


async def fake_llm(messages, *, tools=None, metadata=None):
    tool_names = [
        tool["function"]["name"]
        for tool in tools or []
    ]

    return {
        "content": (
            f"agent={metadata['agent_name']} "
            f"role={metadata['agent_role']} "
            f"tools={','.join(tool_names)} "
            f"task={messages[-1]['content']}"
        ),
        "metadata": {
            "tool_count": len(tool_names),
        },
    }


def test_tool_scope_filters_allowed_tools() -> None:
    scope = SubAgentToolScope(
        allowed_tools=frozenset({"file_read"}),
        denied_tools=frozenset(),
        permission_mode="readonly",
    )

    filtered = filter_tool_definitions_by_scope(
        TOOL_DEFINITIONS,
        scope,
    )

    names = [
        tool["function"]["name"]
        for tool in filtered
    ]

    assert names == ["file_read"]


def test_tool_scope_denied_tools_win_over_allowed_tools() -> None:
    scope = SubAgentToolScope(
        allowed_tools=frozenset({"file_read", "bash"}),
        denied_tools=frozenset({"bash"}),
        permission_mode="readonly",
    )

    filtered = filter_tool_definitions_by_scope(
        TOOL_DEFINITIONS,
        scope,
    )

    names = [
        tool["function"]["name"]
        for tool in filtered
    ]

    assert names == ["file_read"]


@pytest.mark.asyncio
async def test_subagent_runs_with_isolated_agent_state(tmp_path: Path) -> None:
    agent = DemoSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    parent_messages = [
        {
            "role": "user",
            "content": "parent context message",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    context = SubAgentContext(
        task="inspect project",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await agent.run(
        "inspect project",
        context=context,
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "demo"
    assert result.role == "general"

    assert "agent=demo" in result.content
    assert "role=general" in result.content
    assert "file_read" in result.content
    assert "file_write" not in result.content
    assert "bash" not in result.content

    assert result.state.metadata["subagent_name"] == "demo"
    assert result.state.metadata["permission_mode"] == "readonly"
    assert result.state.metadata["workspace_path"] == str(tmp_path.resolve())

    assert result.state.messages[0].role == "system"
    assert result.state.messages[-1].role == "assistant"

    # parent_messages 只被复制，原对象不应该被修改
    assert parent_messages == [
        {
            "role": "user",
            "content": "parent context message",
            "metadata": {
                "source": "parent",
            },
        }
    ]


@pytest.mark.asyncio
async def test_subagent_abort_before_llm_call(tmp_path: Path) -> None:
    called = False

    async def llm_should_not_be_called(messages, *, tools=None, metadata=None):
        nonlocal called
        called = True
        return "should not happen"

    signal = SubAgentAbortSignal()
    signal.abort("user cancelled")

    agent = DemoSubAgent(
        llm=llm_should_not_be_called,
        tool_definitions=TOOL_DEFINITIONS,
    )

    result = await agent.run(
        "inspect project",
        context=SubAgentContext(
            task="inspect project",
            workspace_path=tmp_path,
        ),
        abort_signal=signal,
    )

    assert result.aborted
    assert result.status == SubAgentStatus.ABORTED
    assert result.error == "user cancelled"
    assert called is False


@pytest.mark.asyncio
async def test_subagent_missing_llm_fails_cleanly(tmp_path: Path) -> None:
    agent = DemoSubAgent(
        tool_definitions=TOOL_DEFINITIONS,
    )

    result = await agent.run(
        "inspect project",
        context=SubAgentContext(
            task="inspect project",
            workspace_path=tmp_path,
        ),
    )

    assert not result.success
    assert result.status == SubAgentStatus.FAILED
    assert result.error is not None
    assert "no llm callable" in result.error


def test_subagent_result_to_dict(tmp_path: Path) -> None:
    agent = DemoSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="demo",
        workspace_path=tmp_path,
    )

    state = agent.create_state(
        context=context,
    )

    from pywork.subagents.base import SubAgentRunResult

    result = SubAgentRunResult(
        agent_id=state.metadata["agent_id"],
        name="demo",
        role="general",
        status=SubAgentStatus.COMPLETED,
        content="ok",
        state=state,
        steps=1,
    )

    data = result.to_dict()

    assert data["name"] == "demo"
    assert data["role"] == "general"
    assert data["status"] == "completed"
    assert data["success"] is True
    assert data["aborted"] is False
    assert "state" in data