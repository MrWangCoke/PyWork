from __future__ import annotations

from pathlib import Path

import pytest

import subprocess

from pywork.permission.policy import PermissionDecisionType
from pywork.runtime.graph import (
    create_default_agent_graph_state,
    execute_tool_node,
    permission_check_node,
)
from pywork.schemas.tool_schema import create_tool_call
from pywork.utils.shell import which


def bash_has_python() -> bool:
    bash_path = which("bash")

    if bash_path is None:
        return False

    try:
        completed = subprocess.run(
            [
                bash_path,
                "-lc",
                "command -v python >/dev/null 2>&1",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            check=False,
        )

        return completed.returncode == 0
    except Exception:
        return False

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


def get_tool_messages(agent_state) -> list[dict]:
    return [
        message
        for message in agent_state.messages
        if isinstance(message, dict) and message.get("role") == "tool"
    ]


@pytest.mark.asyncio
async def test_bash_pytest_returns_stdout_stderr_exit_code(tmp_path: Path) -> None:
    if which("bash") is None:
        pytest.skip("bash executable is not available on this machine")
    if not bash_has_python():
        pytest.skip("python executable is not available inside bash")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()

    sample_test = tests_dir / "test_sample.py"
    sample_test.write_text(
        "def test_sample():\n"
        "    assert 1 + 1 == 2\n",
        encoding="utf-8",
    )

    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "python -m pytest tests/test_sample.py -q",
            "timeout": 30,
        },
    )

    data = permission_check_node(data)

    assert data["permission_gate_result"].allowed

    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert result.success
    assert isinstance(result.data, dict)

    assert "stdout" in result.data
    assert "stderr" in result.data
    assert "exit_code" in result.data
    assert "timed_out" in result.data
    assert "duration_ms" in result.data

    assert result.data["exit_code"] == 0
    assert result.data["timed_out"] is False

    agent_state = output["agent_state"]
    tool_messages = get_tool_messages(agent_state)

    assert tool_messages

    latest = tool_messages[-1]

    assert latest["role"] == "tool"
    assert latest["name"] == "bash"
    assert "exit_code: 0" in latest["content"]
    assert "STDOUT:" in latest["content"]
    assert "STDERR:" in latest["content"]


def test_bash_rm_rf_build_requires_elevated_approval(tmp_path: Path) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf build",
        },
    )

    output = permission_check_node(data)
    gate_result = output["permission_gate_result"]

    assert gate_result.requires_elevated_confirmation
    assert gate_result.decision.decision == PermissionDecisionType.ASK_ELEVATED


def test_bash_rm_rf_root_is_denied(tmp_path: Path) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="bash",
        arguments={
            "command": "rm -rf /",
        },
    )

    output = permission_check_node(data)
    gate_result = output["permission_gate_result"]

    assert gate_result.denied
    assert gate_result.decision.decision == PermissionDecisionType.DENY


@pytest.mark.asyncio
async def test_runtime_does_not_execute_denied_bash(tmp_path: Path) -> None:
    called = False

    async def approval_handler(gate_result):
        nonlocal called
        called = True
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
    output = await execute_tool_node(data)

    result = output["tool_result"]

    assert not result.success
    assert called is False
    assert "Permission denied" in result.content


def test_powershell_remove_item_requires_elevated_approval(tmp_path: Path) -> None:
    data = make_graph_data(tmp_path)

    attach_tool_call(
        data,
        tool_name="powershell",
        arguments={
            "command": "Remove-Item build -Recurse -Force",
        },
    )

    output = permission_check_node(data)
    gate_result = output["permission_gate_result"]

    assert gate_result.requires_elevated_confirmation
    assert gate_result.decision.decision == PermissionDecisionType.ASK_ELEVATED