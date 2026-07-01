from __future__ import annotations

from pathlib import Path

import pytest

from pywork.permission.policy import PermissionDecisionType
from pywork.permission.risk import RiskLevel
from pywork.runtime.permission_gate import PermissionGate


def test_file_write_normal_file_requires_approval(tmp_path: Path) -> None:
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
    assert result.should_ask
    assert result.rule_result is not None
    assert result.rule_result.source == "file_permissions"


def test_file_write_accept_edits_allows_normal_file(tmp_path: Path) -> None:
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
    assert result.allowed


def test_file_write_env_requires_elevated_approval(tmp_path: Path) -> None:
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
    assert result.requires_elevated_confirmation


def test_file_edit_git_config_is_denied(tmp_path: Path) -> None:
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
    assert result.denied


def test_file_read_outside_workspace_is_denied(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_read",
        arguments={
            "path": "../outside.txt",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.DENY
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.denied


def test_file_edit_pyproject_requires_elevated_approval(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_edit",
        arguments={
            "path": "pyproject.toml",
            "old_string": "demo",
            "new_string": "changed",
        },
        mode="accept_edits",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.requires_elevated_confirmation


def test_bash_rm_rf_build_requires_elevated_approval(tmp_path: Path) -> None:
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
    assert result.requires_elevated_confirmation
    assert result.rule_result is not None
    assert result.rule_result.source == "bash_permissions"


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf /",
        "rm -rf ~",
        "rm -rf $HOME",
        "rm -rf ${HOME}",
        "rm -rf *",
    ],
)
def test_bash_dangerous_rm_rf_is_denied(
    tmp_path: Path,
    command: str,
) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "bash",
        arguments={
            "command": command,
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.DENY
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.denied


def test_powershell_remove_item_requires_elevated_approval(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "powershell",
        arguments={
            "command": "Remove-Item build -Recurse -Force",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.requires_elevated_confirmation
    assert result.rule_result is not None
    assert result.rule_result.source == "powershell_permissions"