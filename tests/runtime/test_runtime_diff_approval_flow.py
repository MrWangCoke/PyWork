from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pywork.permission.audit import PermissionAuditUserAction
from pywork.runtime.file_change_preview import (
    build_file_change_preview_for_gate_result,
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
async def test_file_write_accept_flow_has_diff_preview_and_creates_file(
    tmp_path: Path,
) -> None:
    seen_diff = ""

    async def approval_handler(gate_result):
        nonlocal seen_diff

        preview = build_file_change_preview_for_gate_result(
            gate_result,
            workspace_path=tmp_path,
        )

        assert preview is not None
        assert preview.operation == "write"
        assert preview.has_changes
        assert not (tmp_path / "src" / "utils" / "helper.py").exists()

        seen_diff = preview.diff_text

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

    assert output["tool_result"].success
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "def helper():\n    return 'ok'\n"

    assert "--- /dev/null" in seen_diff
    assert "+++ b/src/utils/helper.py" in seen_diff
    assert "+def helper():" in seen_diff


@pytest.mark.asyncio
async def test_file_write_reject_flow_has_diff_preview_and_does_not_create_file(
    tmp_path: Path,
) -> None:
    seen_diff = ""

    async def approval_handler(gate_result):
        nonlocal seen_diff

        preview = build_file_change_preview_for_gate_result(
            gate_result,
            workspace_path=tmp_path,
        )

        assert preview is not None
        seen_diff = preview.diff_text

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

    assert not output["tool_result"].success
    assert not target.exists()
    assert "+def helper():" in seen_diff


@pytest.mark.asyncio
async def test_file_edit_accept_flow_has_diff_preview_and_modifies_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "helper.py"
    target.write_text(
        "value = 'old'\n",
        encoding="utf-8",
    )

    seen_diff = ""

    async def approval_handler(gate_result):
        nonlocal seen_diff

        preview = build_file_change_preview_for_gate_result(
            gate_result,
            workspace_path=tmp_path,
        )

        assert preview is not None
        assert preview.operation == "edit"
        assert target.read_text(encoding="utf-8") == "value = 'old'\n"

        seen_diff = preview.diff_text

        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.ALLOW,
            allowed=True,
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

    assert output["tool_result"].success
    assert target.read_text(encoding="utf-8") == "value = 'new'\n"

    assert "--- a/helper.py" in seen_diff
    assert "+++ b/helper.py" in seen_diff
    assert "-value = 'old'" in seen_diff
    assert "+value = 'new'" in seen_diff


@pytest.mark.asyncio
async def test_file_edit_reject_flow_has_diff_preview_and_keeps_file_unchanged(
    tmp_path: Path,
) -> None:
    target = tmp_path / "helper.py"
    target.write_text(
        "value = 'old'\n",
        encoding="utf-8",
    )

    seen_diff = ""

    async def approval_handler(gate_result):
        nonlocal seen_diff

        preview = build_file_change_preview_for_gate_result(
            gate_result,
            workspace_path=tmp_path,
        )

        assert preview is not None
        seen_diff = preview.diff_text

        return FakeApprovalResult(
            user_action=PermissionAuditUserAction.DENY,
            allowed=False,
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

    assert not output["tool_result"].success
    assert target.read_text(encoding="utf-8") == "value = 'old'\n"

    assert "-value = 'old'" in seen_diff
    assert "+value = 'new'" in seen_diff