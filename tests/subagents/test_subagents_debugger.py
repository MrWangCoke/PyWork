from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.base import SubAgentContext, SubAgentStatus
from pywork.subagents.debugger import DebuggerSubAgent


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


def test_debugger_subagent_defaults() -> None:
    agent = DebuggerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    assert agent.name == "debugger"
    assert agent.role == "debugger"
    assert agent.description == "Debugging subagent"
    assert agent.tool_scope.permission_mode == "default"

    allowed_tool_names = [
        tool["function"]["name"]
        for tool in agent.get_tool_definitions()
    ]

    assert allowed_tool_names == [
        "file_read",
        "glob",
        "grep",
        "bash",
        "powershell",
    ]


def test_debugger_prompt_is_debugging_focused() -> None:
    agent = DebuggerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    prompt = agent.get_system_prompt()

    assert "debugging subagent" in prompt
    assert "Analyze errors" in prompt
    assert "Most likely root cause" in prompt
    assert "Minimal fix" in prompt
    assert "Verification command" in prompt
    assert "Do not run destructive commands" in prompt
    assert "Do not bypass PermissionGate" in prompt


@pytest.mark.asyncio
async def test_debugger_runs_with_shell_tools_in_default_mode(tmp_path: Path) -> None:
    agent = DebuggerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="debug failing runtime shell flow test",
        workspace_path=tmp_path,
        parent_messages=[
            {
                "role": "user",
                "content": "pytest failed with exit_code 127 in bash.",
            }
        ],
    )

    result = await agent.run(
        "debug failing runtime shell flow test",
        context=context,
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "debugger"
    assert result.role == "debugger"

    assert "agent=debugger" in result.content
    assert "role=debugger" in result.content
    assert "mode=default" in result.content

    assert "file_read" in result.content
    assert "glob" in result.content
    assert "grep" in result.content
    assert "bash" in result.content
    assert "powershell" in result.content

    assert "file_write" not in result.content
    assert "file_edit" not in result.content

    assert result.state.metadata["subagent_name"] == "debugger"
    assert result.state.metadata["subagent_role"] == "debugger"
    assert result.state.metadata["permission_mode"] == "default"
    assert result.state.metadata["workspace_path"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_debugger_keeps_parent_context_isolated(tmp_path: Path) -> None:
    parent_messages = [
        {
            "role": "user",
            "content": "Parent says: runtime task is stuck after tool_result_observed.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    agent = DebuggerSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="debug stuck runtime",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await agent.run(
        "debug stuck runtime",
        context=context,
    )

    assert result.success

    # 子 Agent 可以复制 parent context，但不能修改原对象
    assert parent_messages == [
        {
            "role": "user",
            "content": "Parent says: runtime task is stuck after tool_result_observed.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    contents = [
        message.content
        for message in result.state.messages
    ]

    assert "Parent says: runtime task is stuck after tool_result_observed." in contents
    assert result.state.messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_debugger_missing_llm_fails_cleanly(tmp_path: Path) -> None:
    agent = DebuggerSubAgent(
        tool_definitions=TOOL_DEFINITIONS,
    )

    result = await agent.run(
        "debug something",
        context=SubAgentContext(
            task="debug something",
            workspace_path=tmp_path,
        ),
    )

    assert not result.success
    assert result.status == SubAgentStatus.FAILED
    assert result.error is not None
    assert "no llm callable" in result.error