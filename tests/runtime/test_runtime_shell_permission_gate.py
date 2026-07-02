from __future__ import annotations

from pathlib import Path

import pytest

from pywork.permission.bash_permissions import evaluate_bash_permission
from pywork.permission.policy import PermissionDecisionType
from pywork.permission.powershell_permissions import evaluate_powershell_permission
from pywork.permission.risk import RiskLevel
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.runtime.permission_gate import PermissionGate
from pywork.schemas.tool_schema import create_tool_call


def test_bash_rm_rf_build_asks_elevated() -> None:
    result = evaluate_bash_permission("rm -rf build")

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "rm_rf" in result.matched_rules


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf $HOME",
        "rm -rf ${HOME}",
        "rm -rf *",
        "rm -rf .",
        "rm -rf ..",
    ],
)
def test_bash_dangerous_rm_rf_is_denied(command: str) -> None:
    result = evaluate_bash_permission(command)

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "dangerous_rm_rf" in result.matched_rules


def test_bash_pytest_is_allowed() -> None:
    result = evaluate_bash_permission("uv run pytest tests")

    assert result.decision == PermissionDecisionType.ALLOW
    assert result.risk == RiskLevel.LOW


def test_permission_gate_uses_bash_permission_for_rm_rf_build(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "bash",
        arguments={
            "command": "rm -rf build",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.rule_result is not None
    assert result.rule_result.source == "bash_permissions"


def test_permission_gate_uses_bash_permission_for_dangerous_rm_rf(
    tmp_path: Path,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "bash",
        arguments={
            "command": "rm -rf /",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.DENY
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.rule_result is not None
    assert result.rule_result.source == "bash_permissions"


def test_powershell_remove_item_build_asks_elevated() -> None:
    result = evaluate_powershell_permission(
        "Remove-Item build -Recurse -Force",
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL


@pytest.mark.parametrize(
    "command",
    [
        "Remove-Item -Recurse -Force C:\\",
        "Remove-Item -Recurse -Force ~",
        "Remove-Item -Recurse -Force $HOME",
        "Remove-Item -Recurse -Force $env:USERPROFILE",
        "Remove-Item -Recurse -Force *",
    ],
)
def test_powershell_dangerous_remove_item_is_denied(command: str) -> None:
    result = evaluate_powershell_permission(command)

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "dangerous_remove_item" in result.matched_rules


def test_powershell_safe_get_child_item_allowed() -> None:
    result = evaluate_powershell_permission("Get-ChildItem")

    assert result.decision == PermissionDecisionType.ALLOW
    assert result.risk == RiskLevel.LOW


def make_graph_data(
    tmp_path: Path,
    *,
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
                "mode": "default",
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
async def test_runtime_denies_dangerous_bash_before_execution(tmp_path: Path) -> None:
    approval_called = False

    async def approval_handler(gate_result):
        nonlocal approval_called
        approval_called = True
        return None

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

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
    assert approval_called is False


@pytest.mark.asyncio
async def test_runtime_asks_elevated_for_rm_rf_build_before_execution(
    tmp_path: Path,
) -> None:
    approval_called = False

    async def approval_handler(gate_result):
        nonlocal approval_called
        approval_called = True
        return None

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf build",
        },
    )

    data = permission_check_node(data)

    gate_result = data["permission_gate_result"]

    assert gate_result.requires_elevated_confirmation
    assert gate_result.decision.decision == PermissionDecisionType.ASK_ELEVATED

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert approval_called is True