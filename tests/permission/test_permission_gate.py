from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pywork.permission.policy import PermissionDecisionType
from pywork.permission.risk import RiskLevel
from pywork.runtime.permission_gate import (
    PermissionGate,
    render_permission_gate_result,
)


@dataclass
class FakeToolCall:
    tool_name: str
    arguments: dict[str, object]
    call_id: str | None = None


def test_file_read_allowed(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_read",
        arguments={
            "path": "README.md",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ALLOW
    assert result.decision.risk == RiskLevel.LOW
    assert result.allowed


def test_file_write_default_asks(tmp_path: Path) -> None:
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


def test_file_edit_important_project_file_asks_elevated(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "file_edit",
        arguments={
            "path": "pyproject.toml",
            "old_string": "a",
            "new_string": "b",
        },
        mode="accept_edits",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.requires_elevated_confirmation


def test_file_read_outside_workspace_denied(tmp_path: Path) -> None:
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


def test_bash_safe_pytest_command_allowed(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "bash",
        arguments={
            "command": "uv run pytest tests",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ALLOW
    assert result.decision.risk == RiskLevel.LOW
    assert result.allowed
    assert result.rule_result is not None
    assert result.rule_result.source == "bash_permissions"


def test_bash_unknown_command_asks(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "bash",
        arguments={
            "command": "custom-tool --flag",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK
    assert result.decision.risk == RiskLevel.MEDIUM
    assert result.should_ask


def test_bash_rm_workspace_target_asks_elevated(tmp_path: Path) -> None:
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


def test_bash_dangerous_rm_rf_denied(tmp_path: Path) -> None:
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
    assert result.denied


def test_powershell_safe_command_allowed(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "powershell",
        arguments={
            "command": "Get-ChildItem",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ALLOW
    assert result.decision.risk == RiskLevel.LOW
    assert result.allowed
    assert result.rule_result is not None
    assert result.rule_result.source == "powershell_permissions"


def test_powershell_remove_item_asks_elevated(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    result = gate.check_tool(
        "powershell",
        arguments={
            "command": "Remove-Item demo.txt",
        },
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.decision.risk == RiskLevel.CRITICAL
    assert result.requires_elevated_confirmation


def test_plan_mode_denies_file_write_even_if_normal_path(tmp_path: Path) -> None:
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
        mode="plan",
    )

    assert result.decision.decision == PermissionDecisionType.DENY
    assert result.decision.risk == RiskLevel.HIGH
    assert result.denied


def test_check_tool_call_like_object(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=False,
    )

    call = FakeToolCall(
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
        call_id="call_123",
    )

    result = gate.check(
        call,
        mode="default",
    )

    assert result.decision.decision == PermissionDecisionType.ASK
    assert result.decision.request.call_id == "call_123"


def test_audit_record_created(tmp_path: Path) -> None:
    gate = PermissionGate(
        tmp_path,
        audit_enabled=True,
        session_id="session_1",
    )

    result = gate.check_tool(
        "file_read",
        arguments={
            "path": "README.md",
        },
        mode="default",
        call_id="call_audit",
    )

    assert result.audit_record is not None
    assert result.audit_record.tool_name == "file_read"
    assert result.audit_record.call_id == "call_audit"
    assert result.audit_record.session_id == "session_1"

    assert (tmp_path / ".pywork" / "audit" / "permissions.jsonl").exists()


def test_render_permission_gate_result(tmp_path: Path) -> None:
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

    rendered = render_permission_gate_result(result)

    assert "deny" in rendered
    assert "tool=bash" in rendered
    assert "risk=critical" in rendered