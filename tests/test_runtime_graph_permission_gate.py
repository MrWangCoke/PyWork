from __future__ import annotations

from pathlib import Path

import pytest

from pywork.permission.policy import PermissionDecisionType
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.schemas.tool_schema import create_tool_call


def make_graph_data(
    tmp_path: Path,
    *,
    mode: str = "default",
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
                "audit_enabled": True,
                "mode": mode,
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


def test_permission_check_allows_file_read(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "hello",
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

    output = permission_check_node(data)

    gate_result = output["permission_gate_result"]

    assert gate_result.allowed
    assert gate_result.decision.decision == PermissionDecisionType.ALLOW


@pytest.mark.asyncio
async def test_execute_tool_runs_when_permission_allowed(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "hello runtime gate",
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

    result = output["tool_result"]

    assert result.success
    assert "hello runtime gate" in result.content


@pytest.mark.asyncio
async def test_execute_tool_blocks_file_write_without_approval(tmp_path: Path) -> None:
    data = make_graph_data(
        tmp_path,
        mode="default",
    )

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.should_ask
    assert gate_result.decision.decision == PermissionDecisionType.ASK

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert "Approval required" in result.content
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_execute_tool_runs_file_write_in_accept_edits(tmp_path: Path) -> None:
    data = make_graph_data(
        tmp_path,
        mode="accept_edits",
    )

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.allowed
    assert gate_result.decision.decision == PermissionDecisionType.ALLOW

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert result.success
    assert (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_execute_tool_denies_dangerous_bash_without_running(tmp_path: Path) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf /",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.denied
    assert gate_result.decision.decision == PermissionDecisionType.DENY

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert "Permission denied" in result.content
    assert "rm -rf" not in result.content or "was not executed" in result.content


def test_permission_check_writes_audit_log(tmp_path: Path) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="file_read",
        arguments={
            "path": "README.md",
        },
    )

    permission_check_node(data)

    audit_path = tmp_path / ".pywork" / "audit" / "permissions.jsonl"

    assert audit_path.exists()
    assert "file_read" in audit_path.read_text(encoding="utf-8")