from __future__ import annotations

import json
from typing import Any


DEFAULT_MAX_STDOUT_CHARS = 20_000
DEFAULT_MAX_STDERR_CHARS = 12_000
DEFAULT_MAX_CONTENT_CHARS = 20_000
DEFAULT_MAX_DATA_CHARS = 12_000


SHELL_RESULT_KEYS = {
    "stdout",
    "stderr",
    "exit_code",
    "timed_out",
    "duration_ms",
    "command",
    "cwd",
    "command_success",
    "stdout_truncated",
    "stderr_truncated",
}


def safe_getattr(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    return getattr(value, name, default)


def truncate_text(
    text: Any,
    *,
    max_chars: int,
    label: str = "content",
) -> str:
    value = "" if text is None else str(text)

    if max_chars <= 0:
        return value

    if len(value) <= max_chars:
        return value

    suffix = f"\n... {label} truncated, original length={len(value)} chars ..."

    return value[: max(0, max_chars - len(suffix))] + suffix


def safe_json_dumps(
    value: Any,
    *,
    max_chars: int = DEFAULT_MAX_DATA_CHARS,
) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        text = str(value)

    return truncate_text(
        text,
        max_chars=max_chars,
        label="json data",
    )


def get_tool_result_data(result: Any) -> dict[str, Any]:
    data = safe_getattr(
        result,
        "data",
        None,
    )

    if isinstance(data, dict):
        return dict(data)

    return {}


def get_tool_result_metadata(result: Any) -> dict[str, Any]:
    metadata = safe_getattr(
        result,
        "metadata",
        None,
    )

    if isinstance(metadata, dict):
        return dict(metadata)

    return {}


def get_tool_result_call_id(result: Any) -> str | None:
    for attr in (
        "call_id",
        "tool_call_id",
    ):
        value = safe_getattr(
            result,
            attr,
            None,
        )

        if value:
            return str(value)

    call = safe_getattr(
        result,
        "call",
        None,
    )

    if call is not None:
        value = (
            safe_getattr(call, "call_id", None)
            or safe_getattr(call, "id", None)
        )

        if value:
            return str(value)

    return None


def get_tool_result_name(result: Any) -> str:
    value = (
        safe_getattr(result, "tool_name", None)
        or safe_getattr(result, "name", None)
    )

    if value:
        return str(value)

    call = safe_getattr(
        result,
        "call",
        None,
    )

    if call is not None:
        value = (
            safe_getattr(call, "tool_name", None)
            or safe_getattr(call, "name", None)
        )

        if value:
            return str(value)

    return "unknown_tool"


def extract_nested_shell_result(data: dict[str, Any]) -> dict[str, Any]:
    nested = data.get("shell_result")

    if isinstance(nested, dict):
        return dict(nested)

    return {}


def get_first_existing(
    *values: Any,
    default: Any = None,
) -> Any:
    for value in values:
        if value is not None:
            return value

    return default


def extract_shell_payload(result: Any) -> dict[str, Any]:
    """
    从 ToolResult.data 中提取 shell 信息。

    兼容两种格式：

    1. 直接在 data 里：
       data["stdout"], data["stderr"], data["exit_code"]

    2. 包在 data["shell_result"] 里：
       data["shell_result"]["stdout"]
    """
    data = get_tool_result_data(result)
    nested = extract_nested_shell_result(data)

    payload = {
        "command": get_first_existing(
            data.get("command"),
            nested.get("command"),
        ),
        "cwd": get_first_existing(
            data.get("cwd"),
            nested.get("cwd"),
        ),
        "exit_code": get_first_existing(
            data.get("exit_code"),
            nested.get("exit_code"),
        ),
        "stdout": get_first_existing(
            data.get("stdout"),
            nested.get("stdout"),
            default="",
        ),
        "stderr": get_first_existing(
            data.get("stderr"),
            nested.get("stderr"),
            default="",
        ),
        "timed_out": get_first_existing(
            data.get("timed_out"),
            nested.get("timed_out"),
        ),
        "duration_ms": get_first_existing(
            data.get("duration_ms"),
            nested.get("duration_ms"),
        ),
        "command_success": get_first_existing(
            data.get("command_success"),
            data.get("success"),
            nested.get("command_success"),
            nested.get("success"),
        ),
        "stdout_truncated": get_first_existing(
            data.get("stdout_truncated"),
            nested.get("stdout_truncated"),
        ),
        "stderr_truncated": get_first_existing(
            data.get("stderr_truncated"),
            nested.get("stderr_truncated"),
        ),
    }

    return payload


def result_has_shell_payload(result: Any) -> bool:
    data = get_tool_result_data(result)
    nested = extract_nested_shell_result(data)

    return bool(
        SHELL_RESULT_KEYS.intersection(data.keys())
        or SHELL_RESULT_KEYS.intersection(nested.keys())
    )


def build_shell_tool_result_content(
    result: Any,
    *,
    max_stdout_chars: int = DEFAULT_MAX_STDOUT_CHARS,
    max_stderr_chars: int = DEFAULT_MAX_STDERR_CHARS,
) -> str:
    payload = extract_shell_payload(result)

    tool_name = get_tool_result_name(result)
    success = bool(safe_getattr(result, "success", False))
    error = safe_getattr(result, "error", None)

    lines: list[str] = [
        "Tool execution result:",
        f"- tool_name: {tool_name}",
        f"- success: {success}",
    ]

    if payload["command"] is not None:
        lines.append(f"- command: {payload['command']}")

    if payload["cwd"] is not None:
        lines.append(f"- cwd: {payload['cwd']}")

    if payload["exit_code"] is not None:
        lines.append(f"- exit_code: {payload['exit_code']}")

    if payload["timed_out"] is not None:
        lines.append(f"- timed_out: {payload['timed_out']}")

    if payload["duration_ms"] is not None:
        lines.append(f"- duration_ms: {payload['duration_ms']}")

    if payload["command_success"] is not None:
        lines.append(f"- command_success: {payload['command_success']}")

    if payload["stdout_truncated"] is not None:
        lines.append(f"- stdout_truncated: {payload['stdout_truncated']}")

    if payload["stderr_truncated"] is not None:
        lines.append(f"- stderr_truncated: {payload['stderr_truncated']}")

    if error:
        lines.extend(
            [
                "",
                "Error:",
                truncate_text(
                    error,
                    max_chars=max_stderr_chars,
                    label="error",
                ),
            ]
        )

    stdout = truncate_text(
        payload["stdout"],
        max_chars=max_stdout_chars,
        label="stdout",
    )

    stderr = truncate_text(
        payload["stderr"],
        max_chars=max_stderr_chars,
        label="stderr",
    )

    lines.extend(
        [
            "",
            "STDOUT:",
            "```text",
            stdout,
            "```",
            "",
            "STDERR:",
            "```text",
            stderr,
            "```",
        ]
    )

    return "\n".join(lines)


def build_generic_tool_result_content(
    result: Any,
    *,
    max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    max_data_chars: int = DEFAULT_MAX_DATA_CHARS,
) -> str:
    tool_name = get_tool_result_name(result)
    success = bool(safe_getattr(result, "success", False))
    content = safe_getattr(result, "content", "") or ""
    error = safe_getattr(result, "error", None)
    data = get_tool_result_data(result)

    lines: list[str] = [
        "Tool execution result:",
        f"- tool_name: {tool_name}",
        f"- success: {success}",
    ]

    if error:
        lines.extend(
            [
                "",
                "Error:",
                truncate_text(
                    error,
                    max_chars=max_content_chars,
                    label="error",
                ),
            ]
        )

    if content:
        lines.extend(
            [
                "",
                "Content:",
                truncate_text(
                    content,
                    max_chars=max_content_chars,
                    label="content",
                ),
            ]
        )

    if data:
        lines.extend(
            [
                "",
                "Data:",
                "```json",
                safe_json_dumps(
                    data,
                    max_chars=max_data_chars,
                ),
                "```",
            ]
        )

    return "\n".join(lines)


def build_tool_result_agent_content(
    result: Any,
) -> str:
    """
    把 ToolResult 转成 LLM 下一轮能读懂的文本。

    Shell 工具重点保留：
    - stdout
    - stderr
    - exit_code
    - timed_out
    - duration_ms

    普通工具保留：
    - content
    - error
    - data
    """
    if result_has_shell_payload(result):
        return build_shell_tool_result_content(result)

    return build_generic_tool_result_content(result)


def tool_result_to_agent_message(
    result: Any,
) -> dict[str, Any]:
    """
    生成标准 tool message。

    这个 message 会进入 AgentState.messages，
    后续 build_llm_messages() 会把它发给 LLM。
    """
    message: dict[str, Any] = {
        "role": "tool",
        "name": get_tool_result_name(result),
        "content": build_tool_result_agent_content(result),
    }

    call_id = get_tool_result_call_id(result)

    if call_id:
        message["tool_call_id"] = call_id

    return message


def append_tool_result_to_agent_state(
    agent_state: Any,
    result: Any,
) -> Any:
    """
    把 ToolResult 追加进 AgentState.messages。

    这里优先直接 append 标准 dict message，原因是：
    - tool message 需要保留 role/name/tool_call_id/content
    - 有些 AgentState.add_message() 会把 dict 当成普通 content
    - 直接 append dict 最稳定，build_llm_messages 后面也更容易识别
    """
    message = tool_result_to_agent_message(result)

    messages = getattr(
        agent_state,
        "messages",
        None,
    )

    if isinstance(messages, list):
        messages.append(message)
        return agent_state

    add_message = getattr(
        agent_state,
        "add_message",
        None,
    )

    if callable(add_message):
        try:
            add_message(
                role=message["role"],
                content=message["content"],
                name=message.get("name"),
                tool_call_id=message.get("tool_call_id"),
            )
            return agent_state
        except TypeError:
            add_message(message)
            return agent_state

    raise TypeError(
        "agent_state does not support messages.append(...) or add_message(...)"
    )
