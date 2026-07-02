from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pywork.runtime.tool_result_payload import (
    append_tool_result_to_agent_state,
    build_tool_result_agent_content,
    tool_result_to_agent_message,
)
from pywork.schemas.tool_schema import ToolResult, create_tool_call


@dataclass
class FakeAgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)


def make_shell_result() -> ToolResult:
    call = create_tool_call(
        tool_name="bash",
        arguments={
            "command": "uv run pytest tests",
        },
    )

    return ToolResult.success_result(
        call=call,
        content="pytest finished with exit code 1",
        data={
            "command": "uv run pytest tests",
            "cwd": "E:/MrWang/Desktop/pywork",
            "exit_code": 1,
            "stdout": "tests/test_demo.py::test_demo FAILED\n",
            "stderr": "AssertionError: expected 1 == 2\n",
            "timed_out": False,
            "duration_ms": 1234,
            "command_success": False,
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
    )


def test_build_shell_tool_result_agent_content_contains_command_fields() -> None:
    result = make_shell_result()

    content = build_tool_result_agent_content(result)

    assert "tool_name: bash" in content
    assert "command: uv run pytest tests" in content
    assert "exit_code: 1" in content
    assert "timed_out: False" in content
    assert "duration_ms: 1234" in content
    assert "STDOUT:" in content
    assert "tests/test_demo.py::test_demo FAILED" in content
    assert "STDERR:" in content
    assert "AssertionError" in content


def test_tool_result_to_agent_message_is_tool_role() -> None:
    result = make_shell_result()

    message = tool_result_to_agent_message(result)

    assert message["role"] == "tool"
    assert message["name"] == "bash"
    assert "content" in message
    assert "exit_code: 1" in message["content"]
    assert "tool_call_id" in message


def test_append_tool_result_to_agent_state_appends_message() -> None:
    result = make_shell_result()
    state = FakeAgentState()

    append_tool_result_to_agent_state(
        state,
        result,
    )

    assert len(state.messages) == 1

    message = state.messages[0]

    assert message["role"] == "tool"
    assert message["name"] == "bash"
    assert "stdout" in message["content"].lower()
    assert "stderr" in message["content"].lower()
    assert "exit_code: 1" in message["content"]


def test_generic_tool_result_content_contains_data() -> None:
    call = create_tool_call(
        tool_name="file_read",
        arguments={
            "path": "README.md",
        },
    )

    result = ToolResult.success_result(
        call=call,
        content="hello",
        data={
            "path": "README.md",
            "size": 5,
        },
    )

    content = build_tool_result_agent_content(result)

    assert "tool_name: file_read" in content
    assert "hello" in content
    assert '"path": "README.md"' in content