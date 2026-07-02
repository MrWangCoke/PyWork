from __future__ import annotations

from pathlib import Path

import pytest

from pywork.permission.file_permissions import evaluate_file_permission
from pywork.permission.policy import PermissionDecisionType
from pywork.permission.risk import RiskLevel
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.runtime.permission_gate import PermissionGate
from pywork.schemas.tool_schema import create_tool_call


def test_file_write_normal_file_asks(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/utils/helper.py",
        operation="write",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK
    assert result.risk == RiskLevel.HIGH
    assert result.matched_rules == ("operation:write",)


def test_file_write_env_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".env",
        operation="write",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "sensitive_file" in result.matched_rules


def test_file_edit_git_config_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".git/config",
        operation="edit",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "protected_directory" in result.matched_rules


def test_file_read_outside_workspace_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "../outside.txt",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "outside_workspace" in result.matched_rules


def test_file_edit_pyproject_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "pyproject.toml",
        operation="edit",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "important_project_file" in result.matched_rules


def test_file_read_normal_file_allowed(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/utils/helper.py",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ALLOW
    assert result.risk == RiskLevel.LOW
    assert result.matched_rules == ("operation:read",)


def test_permission_gate_uses_file_permission_for_file_write(
    tmp_path: Path,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK
    assert result.decision.risk == RiskLevel.HIGH
    assert result.rule_result is not None
    assert result.rule_result.source == "file_permissions"


def test_permission_gate_accept_edits_allows_normal_file_write(
    tmp_path: Path,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
        mode="accept_edits",
    )

    assert result.decision.decision == PermissionDecisionType.ALLOW
    assert result.decision.risk == RiskLevel.HIGH


def test_permission_gate_accept_edits_does_not_bypass_env(
    tmp_path: Path,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_write",
        arguments={
            "path": ".env",
            "content": "TOKEN=abc\n",
        },
        mode="accept_edits",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.rule_result is not None
    assert result.rule_result.hard_decision is True


def test_permission_gate_accept_edits_does_not_bypass_git_config(
    tmp_path: Path,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_edit",
        arguments={
            "path": ".git/config",
            "old_string": "a",
            "new_string": "b",
        },
        mode="accept_edits",
    )

    assert result.decision.decision == PermissionDecisionType.DENY
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.rule_result is not None
    assert result.rule_result.hard_decision is True


def make_graph_data(
    tmp_path: Path,
    *,
    mode: str = "default",
    approval_handler=None,
) -> dict:
    data = create_default_agent_graph_state(
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

    data["approval_handler"] = approval_handler

    return data


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


@pytest.mark.asyncio
async def test_runtime_blocks_normal_file_write_without_approval(
    tmp_path: Path,
) -> None:
    data = make_graph_data(
        tmp_path,
        mode="default",
        approval_handler=None,
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
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_runtime_runs_normal_file_write_in_accept_edits(
    tmp_path: Path,
) -> None:
    data = make_graph_data(
        tmp_path,
        mode="accept_edits",
        approval_handler=None,
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
async def test_runtime_denies_git_config_edit_before_execution(
    tmp_path: Path,
) -> None:
    data = make_graph_data(
        tmp_path,
        mode="accept_edits",
        approval_handler=None,
    )

    attach_tool_call(
        data,
        tool_name="file_edit",
        arguments={
            "path": ".git/config",
            "old_string": "a",
            "new_string": "b",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.denied
    assert gate_result.decision.decision == PermissionDecisionType.DENY

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success


@pytest.mark.asyncio
async def test_runtime_pyproject_edit_requires_elevated_approval(
    tmp_path: Path,
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        "[project]\nname = 'demo'\n",
        encoding="utf-8",
    )

    data = make_graph_data(
        tmp_path,
        mode="accept_edits",
        approval_handler=None,
    )

    attach_tool_call(
        data,
        tool_name="file_edit",
        arguments={
            "path": "pyproject.toml",
            "old_string": "demo",
            "new_string": "changed",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.requires_elevated_confirmation
    assert gate_result.decision.decision == PermissionDecisionType.ASK_ELEVATED

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert pyproject.read_text(encoding="utf-8") == "[project]\nname = 'demo'\n"