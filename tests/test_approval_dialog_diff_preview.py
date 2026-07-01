from __future__ import annotations

from pywork.permission.policy import PermissionPolicy
from pywork.tui.components.approval_dialog import (
    ApprovalChoice,
    ApprovalDialog,
    approval_result_from_choice,
    build_warning_text,
    is_file_change_decision,
)


DEMO_DIFF = """--- a/src/utils/helper.py
+++ b/src/utils/helper.py
@@ -1 +1 @@
-old
+new
"""


def make_file_write_decision():
    return PermissionPolicy().evaluate_tool(
        "file_write",
        action="write",
        mode="default",
        risk="high",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
        call_id="call_file_write",
    )


def make_bash_decision():
    return PermissionPolicy().evaluate_tool(
        "bash",
        action="execute",
        mode="default",
        risk="critical",
        arguments={
            "command": "rm -rf build",
        },
        call_id="call_bash",
    )


def test_file_change_decision_detected() -> None:
    decision = make_file_write_decision()

    assert is_file_change_decision(decision)


def test_file_change_dialog_uses_accept_reject_labels() -> None:
    decision = make_file_write_decision()

    dialog = ApprovalDialog(
        decision,
        diff_text=DEMO_DIFF,
    )

    assert dialog.is_file_change
    assert dialog.has_diff_preview
    assert dialog.allow_label == "Accept"
    assert dialog.deny_label == "Reject"


def test_approval_dialog_does_not_expose_always_allow() -> None:
    decision = make_file_write_decision()
    dialog = ApprovalDialog(
        decision,
        show_always_allow=True,
    )

    actions = {binding[1] for binding in dialog.BINDINGS}

    assert dialog.show_always_allow is False
    assert "always_allow" not in actions


def test_normal_tool_dialog_uses_allow_deny_labels() -> None:
    decision = make_bash_decision()

    dialog = ApprovalDialog(decision)

    assert not dialog.is_file_change
    assert not dialog.has_diff_preview
    assert dialog.allow_label == "Allow"
    assert dialog.deny_label == "Deny"


def test_file_change_dialog_can_exist_without_diff_text() -> None:
    decision = make_file_write_decision()

    dialog = ApprovalDialog(decision)

    assert dialog.is_file_change
    assert not dialog.has_diff_preview
    assert dialog.allow_label == "Accept"
    assert dialog.deny_label == "Reject"


def test_approval_result_values_stay_runtime_compatible() -> None:
    decision = make_file_write_decision()

    allow_result = approval_result_from_choice(
        decision,
        ApprovalChoice.ALLOW,
    )
    deny_result = approval_result_from_choice(
        decision,
        ApprovalChoice.DENY,
    )
    always_allow_result = approval_result_from_choice(
        decision,
        ApprovalChoice.ALWAYS_ALLOW,
    )

    assert allow_result.allowed is True
    assert allow_result.always_allow is False
    assert allow_result.user_action.value == "allow"

    assert deny_result.allowed is False
    assert deny_result.always_allow is False
    assert deny_result.user_action.value == "deny"

    assert always_allow_result.allowed is True
    assert always_allow_result.always_allow is True
    assert always_allow_result.user_action.value == "always_allow"


def test_file_preview_warning_mentions_diff() -> None:
    decision = make_file_write_decision()

    warning = build_warning_text(
        decision,
        is_file_preview=True,
    )

    assert "diff preview" in warning.plain
    assert "Reject" in warning.plain
