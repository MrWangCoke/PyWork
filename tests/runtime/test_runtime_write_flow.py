from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pywork.permission.audit import PermissionAuditUserAction
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
    approval_handler=None,
    mode: str = "default",
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
async def test_file_write_allow_creates_file(tmp_path: Path) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALLOW,
            allowed=True,
        )

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

    target = tmp_path / "src" / "utils" / "helper.py"

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "def helper():\n    return 'ok'\n",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert result.success
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "def helper():\n    return 'ok'\n"


@pytest.mark.asyncio
async def test_file_write_deny_does_not_create_file(tmp_path: Path) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.DENY,
            allowed=False,
        )

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

    target = tmp_path / "src" / "utils" / "helper.py"

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "def helper():\n    return 'ok'\n",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert not target.exists()


@pytest.mark.asyncio
async def test_file_edit_allow_modifies_file(tmp_path: Path) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALLOW,
            allowed=True,
        )

    target = tmp_path / "helper.py"
    target.write_text(
        "value = 'old'\n",
        encoding="utf-8",
    )

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

    attach_tool_call(
        data,
        tool_name="file_edit",
        arguments={
            "path": "helper.py",
            "old_string": "old",
            "new_string": "new",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert result.success
    assert target.read_text(encoding="utf-8") == "value = 'new'\n"


@pytest.mark.asyncio
async def test_file_edit_reject_keeps_file_unchanged(tmp_path: Path) -> None:
    async def approval_handler(gate_result):
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.DENY,
            allowed=False,
        )

    target = tmp_path / "helper.py"
    target.write_text(
        "value = 'old'\n",
        encoding="utf-8",
    )

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
    )

    attach_tool_call(
        data,
        tool_name="file_edit",
        arguments={
            "path": "helper.py",
            "old_string": "old",
            "new_string": "new",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert target.read_text(encoding="utf-8") == "value = 'old'\n"


@pytest.mark.asyncio
async def test_file_write_accept_edits_mode_creates_without_approval(
    tmp_path: Path,
) -> None:
    called = False

    async def approval_handler(gate_result):
        nonlocal called
        called = True
        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALLOW,
            allowed=True,
        )

    data = make_graph_data(
        tmp_path,
        approval_handler=approval_handler,
        mode="accept_edits",
    )

    target = tmp_path / "src" / "utils" / "helper.py"

    attach_tool_call(
        data,
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "x = 1\n",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    assert output["tool_result"].success
    assert target.exists()
    assert called is False