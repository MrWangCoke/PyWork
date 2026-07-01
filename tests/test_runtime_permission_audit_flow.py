from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pywork.permission.audit import (
    PermissionAuditEventType,
    PermissionAuditLog,
    PermissionAuditUserAction,
)
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.schemas.tool_schema import create_tool_call


@dataclass
class FakeApprovalResult:
    user_action: PermissionAuditUserAction
    allowed: bool
    always_allow: bool = False


def make_graph_data(
    tmp_path: Path,
    *,
    mode: str = "default",
    approval_handler=None,
    session_id: str = "audit_session",
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

    data["session_id"] = session_id
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


def read_audit_records(tmp_path: Path):
    return PermissionAuditLog(tmp_path).list_records()


def event_types(records):
    return [record.event_type for record in records]


@pytest.mark.asyncio
async def test_auto_allow_records_policy_and_execution_result(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "hello audit",
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

    records = read_audit_records(tmp_path)

    assert event_types(records) == [
        PermissionAuditEventType.POLICY_DECISION,
        PermissionAuditEventType.EXECUTION_RESULT,
    ]

    policy_record = records[0]
    execution_record = records[1]

    assert policy_record.tool_name == "file_read"
    assert policy_record.decision == "allow"
    assert policy_record.allowed is True

    assert execution_record.tool_name == "file_read"
    assert execution_record.event_type == PermissionAuditEventType.EXECUTION_RESULT
    assert execution_record.allowed is True
    assert execution_record.metadata["executed"] is True
    assert execution_record.metadata["success"] is True


@pytest.mark.asyncio
async def test_approval_allow_records_policy_user_and_execution_result(
    tmp_path: Path,
) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALLOW,
            allowed=True,
        )

    data = make_graph_data(
        tmp_path,
        mode="default",
        approval_handler=approval_handler,
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
    output = await execute_tool_node(data)

    assert output["tool_result"].success
    assert (tmp_path / "src" / "utils" / "helper.py").exists()

    records = read_audit_records(tmp_path)

    assert event_types(records) == [
        PermissionAuditEventType.POLICY_DECISION,
        PermissionAuditEventType.USER_DECISION,
        PermissionAuditEventType.EXECUTION_RESULT,
    ]

    assert records[0].decision == "ask"

    assert records[1].event_type == PermissionAuditEventType.USER_DECISION
    assert records[1].user_action == "allow"
    assert records[1].allowed is True

    assert records[2].event_type == PermissionAuditEventType.EXECUTION_RESULT
    assert records[2].allowed is True
    assert records[2].metadata["executed"] is True
    assert records[2].metadata["success"] is True


@pytest.mark.asyncio
async def test_approval_deny_records_policy_user_and_execution_result(
    tmp_path: Path,
) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.DENY,
            allowed=False,
        )

    data = make_graph_data(
        tmp_path,
        mode="default",
        approval_handler=approval_handler,
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
    output = await execute_tool_node(data)

    assert not output["tool_result"].success
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()

    records = read_audit_records(tmp_path)

    assert event_types(records) == [
        PermissionAuditEventType.POLICY_DECISION,
        PermissionAuditEventType.USER_DECISION,
        PermissionAuditEventType.EXECUTION_RESULT,
    ]

    assert records[0].decision == "ask"

    assert records[1].event_type == PermissionAuditEventType.USER_DECISION
    assert records[1].user_action == "deny"
    assert records[1].allowed is False

    assert records[2].event_type == PermissionAuditEventType.EXECUTION_RESULT
    assert records[2].allowed is False
    assert records[2].metadata["executed"] is False
    assert records[2].metadata["success"] is False


@pytest.mark.asyncio
async def test_missing_approval_handler_records_user_deny(
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
    output = await execute_tool_node(data)

    assert not output["tool_result"].success
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()

    records = read_audit_records(tmp_path)

    assert event_types(records) == [
        PermissionAuditEventType.POLICY_DECISION,
        PermissionAuditEventType.USER_DECISION,
        PermissionAuditEventType.EXECUTION_RESULT,
    ]

    assert records[1].user_action == "deny"
    assert records[1].allowed is False
    assert records[1].metadata["approval_result_present"] is False

    assert records[2].metadata["executed"] is False


@pytest.mark.asyncio
async def test_policy_deny_records_policy_and_execution_result_only(
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

    records = read_audit_records(tmp_path)

    assert event_types(records) == [
        PermissionAuditEventType.POLICY_DECISION,
        PermissionAuditEventType.EXECUTION_RESULT,
    ]

    assert records[0].tool_name == "bash"
    assert records[0].decision == "deny"

    assert records[1].event_type == PermissionAuditEventType.EXECUTION_RESULT
    assert records[1].allowed is False
    assert records[1].metadata["executed"] is False


@pytest.mark.asyncio
async def test_always_allow_records_user_decision_and_later_policy_allow(
    tmp_path: Path,
) -> None:
    call_count = 0

    async def approval_handler(gate_result):
        nonlocal call_count
        call_count += 1
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALWAYS_ALLOW,
            allowed=True,
            always_allow=True,
        )

    data = make_graph_data(
        tmp_path,
        mode="default",
        approval_handler=approval_handler,
    )

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('one')\n",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    assert output["tool_result"].success
    assert call_count == 1

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('two')\n",
            "overwrite": True,
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    assert output["tool_result"].success
    assert call_count == 1

    records = read_audit_records(tmp_path)

    user_records = [
        record
        for record in records
        if record.event_type == PermissionAuditEventType.USER_DECISION
    ]

    assert len(user_records) == 1
    assert user_records[0].user_action == "always_allow"

    execution_records = [
        record
        for record in records
        if record.event_type == PermissionAuditEventType.EXECUTION_RESULT
    ]

    assert len(execution_records) == 2
    assert all(record.metadata["executed"] for record in execution_records)