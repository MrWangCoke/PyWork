from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.base import SubAgentContext, SubAgentStatus
from pywork.subagents.general import GeneralSubAgent


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
            "name": "bash",
            "description": "Run bash command",
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


def test_general_subagent_defaults() -> None:
    agent = GeneralSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    assert agent.name == "general"
    assert agent.role == "general"
    assert agent.description == "General-purpose coding subagent"
    assert agent.tool_scope.permission_mode == "readonly"

    allowed_tool_names = [
        tool["function"]["name"]
        for tool in agent.get_tool_definitions()
    ]

    assert allowed_tool_names == [
        "file_read",
        "glob",
        "grep",
    ]


@pytest.mark.asyncio
async def test_general_subagent_runs_in_readonly_scope(tmp_path: Path) -> None:
    agent = GeneralSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="inspect project structure",
        workspace_path=tmp_path,
        parent_messages=[
            {
                "role": "user",
                "content": "Parent task context",
            }
        ],
    )

    result = await agent.run(
        "inspect project structure",
        context=context,
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "general"
    assert result.role == "general"

    assert "agent=general" in result.content
    assert "role=general" in result.content
    assert "mode=readonly" in result.content

    assert "file_read" in result.content
    assert "glob" in result.content
    assert "grep" in result.content

    assert "file_write" not in result.content
    assert "bash" not in result.content

    assert result.state.metadata["subagent_name"] == "general"
    assert result.state.metadata["subagent_role"] == "general"
    assert result.state.metadata["permission_mode"] == "readonly"
    assert result.state.metadata["workspace_path"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_general_subagent_has_parent_context_isolation(tmp_path: Path) -> None:
    parent_messages = [
        {
            "role": "user",
            "content": "This is parent context",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    agent = GeneralSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="analyze context",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await agent.run(
        "analyze context",
        context=context,
    )

    assert result.success

    # 原 parent_messages 不应该被子 Agent 修改
    assert parent_messages == [
        {
            "role": "user",
            "content": "This is parent context",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    contents = [
        message.content
        for message in result.state.messages
    ]

    assert "This is parent context" in contents
    assert result.state.messages[-1].role == "assistant"