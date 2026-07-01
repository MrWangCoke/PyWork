from __future__ import annotations

import json
from pathlib import Path

from pywork.permission.audit import (
    PermissionAuditEventType,
    PermissionAuditLog,
    PermissionAuditUserAction,
    create_policy_decision_record,
    create_user_decision_record,
    record_permission_decision,
    render_permission_audit_record,
    sanitize_arguments,
)
from pywork.permission.policy import PermissionDecisionType, evaluate_permission


def test_sanitize_arguments_redacts_secret_keys() -> None:
    data = sanitize_arguments(
        {
            "path": "demo.txt",
            "api_key": "sk-secret-value",
            "password": "123456",
            "nested": {
                "token": "abc",
                "normal": "ok",
            },
        }
    )

    assert data["path"] == "demo.txt"
    assert data["api_key"] == "[REDACTED]"
    assert data["password"] == "[REDACTED]"
    assert data["nested"]["token"] == "[REDACTED]"
    assert data["nested"]["normal"] == "ok"


def test_sanitize_arguments_redacts_secret_text() -> None:
    data = sanitize_arguments(
        {
            "command": "echo sk-abcdefghijklmnopqrstuvwxyz",
        }
    )

    assert "[REDACTED_SECRET]" in data["command"]
    assert "sk-abcdefghijklmnopqrstuvwxyz" not in data["command"]


def test_create_policy_decision_record() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
            "content": "hello",
        },
        call_id="call_1",
    )

    record = create_policy_decision_record(
        decision,
        session_id="session_1",
    )

    assert record.event_type == PermissionAuditEventType.POLICY_DECISION
    assert record.tool_name == "file_write"
    assert record.mode == "default"
    assert record.risk == "high"
    assert record.decision == "ask"
    assert record.allowed is False
    assert record.call_id == "call_1"
    assert record.session_id == "session_1"
    assert record.arguments["path"] == "demo.txt"


def test_create_user_decision_record_allow() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
        },
        call_id="call_1",
    )

    record = create_user_decision_record(
        decision,
        user_action=PermissionAuditUserAction.ALLOW,
        session_id="session_1",
    )

    assert record.event_type == PermissionAuditEventType.USER_DECISION
    assert record.user_action == "allow"
    assert record.allowed is True
    assert record.tool_name == "file_write"


def test_create_user_decision_record_deny() -> None:
    decision = evaluate_permission(
        "bash",
        mode="default",
        arguments={
            "command": "rm demo.txt",
        },
        call_id="call_2",
    )

    record = create_user_decision_record(
        decision,
        user_action=PermissionAuditUserAction.DENY,
    )

    assert record.user_action == "deny"
    assert record.allowed is False
    assert record.risk == "critical"


def test_audit_log_append_and_list(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    decision = evaluate_permission(
        "file_read",
        mode="default",
        arguments={
            "path": "README.md",
        },
        call_id="call_1",
    )

    written = audit_log.record_policy_decision(
        decision,
        session_id="session_1",
    )

    assert audit_log.path.exists()

    records = audit_log.list_records()

    assert len(records) == 1
    assert records[0].audit_id == written.audit_id
    assert records[0].tool_name == "file_read"
    assert records[0].decision == "allow"


def test_audit_log_records_jsonl(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
        },
        call_id="call_json",
    )

    audit_log.record_policy_decision(decision)

    lines = audit_log.path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["tool_name"] == "file_write"
    assert data["decision"] == "ask"
    assert data["risk"] == "high"


def test_audit_log_filter_by_call_id(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    first = evaluate_permission(
        "file_read",
        mode="default",
        call_id="call_1",
    )
    second = evaluate_permission(
        "bash",
        mode="default",
        call_id="call_2",
    )

    audit_log.record_policy_decision(first)
    audit_log.record_policy_decision(second)

    records = audit_log.find_by_call_id("call_2")

    assert len(records) == 1
    assert records[0].tool_name == "bash"
    assert records[0].decision == "ask_elevated"


def test_audit_log_filter_by_event_type(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    decision = evaluate_permission(
        "file_write",
        mode="default",
        call_id="call_1",
    )

    audit_log.record_policy_decision(decision)
    audit_log.record_user_decision(
        decision,
        user_action=PermissionAuditUserAction.ALLOW,
    )

    policy_records = audit_log.list_records(
        event_type=PermissionAuditEventType.POLICY_DECISION,
    )
    user_records = audit_log.list_records(
        event_type=PermissionAuditEventType.USER_DECISION,
    )

    assert len(policy_records) == 1
    assert len(user_records) == 1
    assert user_records[0].user_action == "allow"


def test_audit_log_reverse_limit(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    for index in range(3):
        decision = evaluate_permission(
            "file_read",
            mode="default",
            call_id=f"call_{index}",
        )
        audit_log.record_policy_decision(decision)

    records = audit_log.list_records(
        reverse=True,
        limit=2,
    )

    assert len(records) == 2
    assert records[0].call_id == "call_2"
    assert records[1].call_id == "call_1"


def test_record_permission_decision_shortcut(tmp_path: Path) -> None:
    decision = evaluate_permission(
        "file_read",
        mode="default",
        call_id="call_shortcut",
    )

    record = record_permission_decision(
        decision,
        workspace_path=tmp_path,
        session_id="session_x",
    )

    assert record.tool_name == "file_read"
    assert record.session_id == "session_x"

    audit_log = PermissionAuditLog(tmp_path)

    records = audit_log.list_records()

    assert len(records) == 1
    assert records[0].call_id == "call_shortcut"


def test_render_permission_audit_record() -> None:
    decision = evaluate_permission(
        "bash",
        mode="default",
        arguments={
            "command": "rm demo.txt",
        },
        call_id="call_render",
    )

    record = create_policy_decision_record(
        decision,
        session_id="session_render",
    )

    rendered = render_permission_audit_record(record)

    assert "event=policy_decision" in rendered
    assert "tool=bash" in rendered
    assert "mode=default" in rendered
    assert "risk=critical" in rendered
    assert "decision=ask_elevated" in rendered
    assert "call_id=call_render" in rendered


def test_clear_audit_log(tmp_path: Path) -> None:
    audit_log = PermissionAuditLog(tmp_path)

    decision = evaluate_permission(
        "file_read",
        mode="default",
    )

    audit_log.record_policy_decision(decision)

    assert audit_log.path.exists()

    audit_log.clear()

    assert not audit_log.path.exists()