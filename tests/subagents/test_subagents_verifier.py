from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.base import SubAgentContext, SubAgentStatus
from pywork.subagents.verifier import VerifierSubAgent


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


def test_verifier_subagent_defaults() -> None:
    agent = VerifierSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    assert agent.name == "verifier"
    assert agent.role == "verifier"
    assert agent.description == "Verification and test-running subagent"
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


def test_verifier_prompt_is_verification_focused() -> None:
    agent = VerifierSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    prompt = agent.get_system_prompt()

    assert "verification subagent" in prompt
    assert "verification commands" in prompt
    assert "stdout" in prompt
    assert "stderr" in prompt
    assert "exit_code" in prompt
    assert "Do not modify files" in prompt
    assert "Do not run destructive commands" in prompt
    assert "Do not bypass PermissionGate" in prompt


@pytest.mark.asyncio
async def test_verifier_runs_with_shell_tools_in_default_mode(tmp_path: Path) -> None:
    agent = VerifierSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="verify runtime shell flow tests",
        workspace_path=tmp_path,
        parent_messages=[
            {
                "role": "user",
                "content": "We changed shell permission handling and need to verify tests.",
            }
        ],
    )

    result = await agent.run(
        "verify runtime shell flow tests",
        context=context,
    )

    assert result.success
    assert result.status == SubAgentStatus.COMPLETED
    assert result.name == "verifier"
    assert result.role == "verifier"

    assert "agent=verifier" in result.content
    assert "role=verifier" in result.content
    assert "mode=default" in result.content

    assert "file_read" in result.content
    assert "glob" in result.content
    assert "grep" in result.content
    assert "bash" in result.content
    assert "powershell" in result.content

    assert "file_write" not in result.content
    assert "file_edit" not in result.content

    assert result.state.metadata["subagent_name"] == "verifier"
    assert result.state.metadata["subagent_role"] == "verifier"
    assert result.state.metadata["permission_mode"] == "default"
    assert result.state.metadata["workspace_path"] == str(tmp_path.resolve())


@pytest.mark.asyncio
async def test_verifier_keeps_parent_context_isolated(tmp_path: Path) -> None:
    parent_messages = [
        {
            "role": "user",
            "content": "Parent says: run focused tests after the file_write change.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    agent = VerifierSubAgent(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
    )

    context = SubAgentContext(
        task="verify file_write behavior",
        workspace_path=tmp_path,
        parent_messages=parent_messages,
    )

    result = await agent.run(
        "verify file_write behavior",
        context=context,
    )

    assert result.success

    # 子 Agent 可以复制 parent context，但不能修改原对象
    assert parent_messages == [
        {
            "role": "user",
            "content": "Parent says: run focused tests after the file_write change.",
            "metadata": {
                "source": "parent",
            },
        }
    ]

    contents = [
        message.content
        for message in result.state.messages
    ]

    assert "Parent says: run focused tests after the file_write change." in contents
    assert result.state.messages[-1].role == "assistant"


@pytest.mark.asyncio
async def test_verifier_missing_llm_fails_cleanly(tmp_path: Path) -> None:
    agent = VerifierSubAgent(
        tool_definitions=TOOL_DEFINITIONS,
    )

    result = await agent.run(
        "verify something",
        context=SubAgentContext(
            task="verify something",
            workspace_path=tmp_path,
        ),
    )

    assert not result.success
    assert result.status == SubAgentStatus.FAILED
    assert result.error is not None
    assert "no llm callable" in result.error