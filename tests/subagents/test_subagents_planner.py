from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.base import SubAgentContext, SubAgentStatus
from pywork.subagents.planner import PlannerSubAgent


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


def test_planner_subagent_defaults() -> None:
    agent = PlannerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    assert agent.name == "planner"
    assert agent.role == "planner"
    assert agent.description == "Task decomposition and implementation planning subagent"
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


def test_planner_prompt_is_planning_focused() -> None:
    agent = PlannerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    prompt = agent.get_system_prompt()

    assert "planning subagent" in prompt
    assert "Do not modify files" in prompt
    assert "Do not run shell commands" in prompt
    assert "Proposed steps" in prompt
    assert "Validation plan" in prompt


@pytest.mark.asyncio
async def test_planner_runs_in_readonly_scope(tmp_path: Path) -> None:
    agent = PlannerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="plan subagent manager implementation",
        workspace_path=tmp_path,
        parent_messages=[
            {
                "role": "user",
                "content": "We are adding multi-agent collaboration.",
            }
        ],
    )

    result = await agent.run(
        "plan subagent manager implementation",
        context=context,
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "planner"
    assert result.role == "planner"

    assert "agent=planner" in result.content
    assert "role=planner" in result.content
    assert "mode=readonly" in result.content

    assert "file_read" in result.content
    assert "glob" in result.content
    assert "grep" in result.content

    assert "file_write" not in result.content
    assert "file_edit" not in result.content
    assert "bash" not in result.content
    assert "powershell" not in result.content

    assert result.state.metadata["subagent_name"] == "planner"
    assert result.state.metadata["subagent_role"] == "planner"
    assert result.state.metadata["permission_mode"] == "readonly"
    assert result.state.metadata["workspace_path"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_planner_keeps_parent_context_isolated(tmp_path: Path) -> None:
    parent_messages = [
        {
            "role": "user",
            "content": "Parent says: implement multi-agent collaboration.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    agent = PlannerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="create an implementation plan",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await agent.run(
        "create an implementation plan",
        context=context,
    )

    assert result.success

    # 子 Agent 可以复制 parent context，但不能修改原对象
    assert parent_messages == [
        {
            "role": "user",
            "content": "Parent says: implement multi-agent collaboration.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    contents = [
        message.content
        for message in result.state.messages
    ]

    assert "Parent says: implement multi-agent collaboration." in contents
    assert result.state.messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_planner_missing_llm_fails_cleanly(tmp_path: Path) -> None:
    agent = PlannerSubAgent(
        tool_definitions=TOOL_DEFINITIONS,
    )

    result = await agent.run(
        "plan something",
        context=SubAgentContext(
            task="plan something",
            workspace_path=tmp_path,
        ),
    )

    assert not result.success
    assert result.status == SubAgentStatus.FAILED
    assert result.error is not None
    assert "no llm callable" in result.error