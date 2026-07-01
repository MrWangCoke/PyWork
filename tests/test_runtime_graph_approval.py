from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from pywork.permission.audit import PermissionAuditUserAction
from pywork.permission.policy import PermissionDecisionType
from pywork.runtime.graph import (
    append_observation_node,
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.runtime.state import AgentStatus
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
async def test_file_write_runs_after_user_allow(tmp_path: Path) -> None:
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

    assert data["permission_gate_result"].should_ask

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert result.success
    assert (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_file_write_does_not_run_after_user_deny(tmp_path: Path) -> None:
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

    result = output["tool_result"]

    assert not result.success
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_user_denied_file_write_finishes_without_runtime_error(
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
    data = await execute_tool_node(data)
    data = append_observation_node(data)

    state = data["agent_state"]
    last_message = state.get_last_message()

    assert state.status == AgentStatus.FINISHED
    assert state.last_error is None
    assert data["graph_route"] == "stop"
    assert data["route_reason"] == "permission_blocked_direct_finish"
    assert last_message is not None
    assert "没有运行" in last_message.content
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_file_write_without_approval_handler_stays_blocked(tmp_path: Path) -> None:
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

    result = output["tool_result"]

    assert not result.success
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


@pytest.mark.asyncio
async def test_dangerous_bash_deny_does_not_call_approval_handler(tmp_path: Path) -> None:
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
    )

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf /",
        },
    )

    data = permission_check_node(data)
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert not called


@pytest.mark.asyncio
async def test_always_allow_applies_to_same_later_operation(tmp_path: Path) -> None:
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

    # 第二次相同 tool + path + risk 应该走 session override
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
