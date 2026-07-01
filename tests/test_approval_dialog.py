from __future__ import annotations

from pywork.permission.audit import PermissionAuditUserAction
from pywork.permission.policy import PermissionDecisionType, evaluate_permission
from pywork.tui.components.approval_dialog import (
    ApprovalChoice,
    approval_result_from_choice,
    build_arguments_renderable,
    build_decision_table,
    choice_to_user_action,
    format_arguments_json,
    risk_style,
)


def test_choice_to_user_action() -> None:
    assert choice_to_user_action(ApprovalChoice.ALLOW) == PermissionAuditUserAction.ALLOW
    assert choice_to_user_action(ApprovalChoice.DENY) == PermissionAuditUserAction.DENY
    assert (
        choice_to_user_action(ApprovalChoice.ALWAYS_ALLOW)
        == PermissionAuditUserAction.ALWAYS_ALLOW
    )


def test_approval_result_from_choice_allow() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
        },
    )

    result = approval_result_from_choice(
        decision,
        ApprovalChoice.ALLOW,
    )

    assert result.choice == ApprovalChoice.ALLOW
    assert result.user_action == PermissionAuditUserAction.ALLOW
    assert result.allowed
    assert not result.always_allow
    assert result.decision is decision


def test_approval_result_from_choice_deny() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
    )

    result = approval_result_from_choice(
        decision,
        ApprovalChoice.DENY,
    )

    assert result.choice == ApprovalChoice.DENY
    assert result.user_action == PermissionAuditUserAction.DENY
    assert not result.allowed
    assert not result.always_allow


def test_approval_result_from_choice_always_allow() -> None:
    decision = evaluate_permission(
        "bash",
        mode="default",
    )

    result = approval_result_from_choice(
        decision,
        ApprovalChoice.ALWAYS_ALLOW,
    )

    assert result.choice == ApprovalChoice.ALWAYS_ALLOW
    assert result.user_action == PermissionAuditUserAction.ALWAYS_ALLOW
    assert result.allowed
    assert result.always_allow


def test_format_arguments_json_redacts_sensitive_values() -> None:
    text = format_arguments_json(
        {
            "path": "demo.txt",
            "api_key": "sk-secret-value",
            "content": "hello",
        }
    )

    assert '"path": "demo.txt"' in text
    assert '"api_key": "[REDACTED]"' in text
    assert "sk-secret-value" not in text


def test_format_arguments_json_truncates() -> None:
    text = format_arguments_json(
        {
            "content": "x" * 100,
        },
        max_chars=20,
    )

    assert "truncated" in text


def test_risk_style() -> None:
    assert risk_style("critical") == "bold red"
    assert risk_style("high") == "bold yellow"
    assert risk_style("medium") == "yellow"
    assert risk_style("low") == "green"


def test_build_decision_table() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
        },
        call_id="call_1",
    )

    table = build_decision_table(decision)

    assert table is not None


def test_build_arguments_renderable_empty() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
    )

    renderable = build_arguments_renderable(decision)

    assert renderable is not None


def test_decision_for_elevated_command() -> None:
    decision = evaluate_permission(
        "powershell",
        mode="default",
        arguments={
            "command": "Remove-Item demo.txt",
        },
    )

    assert decision.decision == PermissionDecisionType.ASK_ELEVATED

    result = approval_result_from_choice(
        decision,
        ApprovalChoice.DENY,
    )

    assert not result.allowed