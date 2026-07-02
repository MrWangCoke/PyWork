from __future__ import annotations

from pathlib import Path

import pytest

from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.schemas.tool_schema import create_tool_call


def make_graph_data(
    tmp_path: Path,
) -> dict:
    return create_default_agent_graph_state(
        user_input="",
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            },
            "permissions": {
                "enabled": True,
                "audit_enabled": False,
                "mode": "default",
            },
        },
    )


def attach_tool_call(
    data: dict,
    *,
    tool_name: str,
    arguments: dict,
):
    call = create_tool_call(
        tool_name=tool_name,
        arguments=arguments,
    )

    data["parsed_tool_call"] = call
    data["tool_call"] = call
    data["has_tool_call"] = True

    return call


def get_tool_messages(agent_state) -> list[dict]:
    return [
        message
        for message in agent_state.messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


@pytest.mark.asyncio
async def test_file_read_result_is_appended_to_agent_state(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "hello agent state",
        encoding="utf-8",
    )

    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="file_read",
        arguments={
            "path": "README.md",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    assert output["tool_result"].success

    agent_state = output["agent_state"]
    tool_messages = get_tool_messages(agent_state)

    assert tool_messages

    latest = tool_messages[-1]

    assert latest["role"] == "tool"
    assert latest["name"] == "file_read"
    assert "hello agent state" in latest["content"]


@pytest.mark.asyncio
async def test_permission_block_result_is_appended_to_agent_state(
    tmp_path: Path,
) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf /",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    assert not output["tool_result"].success

    agent_state = output["agent_state"]
    tool_messages = get_tool_messages(agent_state)

    assert tool_messages

    latest = tool_messages[-1]

    assert latest["role"] == "tool"
    assert latest["name"] == "bash"
    assert "Permission denied" in latest["content"]
    assert "was not executed" in latest["content"]