from __future__ import annotations

import json
from typing import Any, Iterable

from pywork.schemas.message_schema import (
    AnyMessage,
    AssistantMessage,
    ErrorMessage,
    MessageRole,
    SystemMessage,
    ToolMessage,
    UserMessage,
    create_assistant_message,
    create_error_message,
    create_system_message,
    create_tool_message,
    create_user_message,
)
from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel


class MessageConversionError(Exception):
    """
    婵炴垵鐗婃导鍛村冀閻撳海纭€閺夌儐鍓氬畷鎻掝嚕閸屾氨鍩楅柕?
    """

    pass


def get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    """
    闁告艾鏈鍌炲礂閻撳寒鍟?dict 闁?SDK 閺夆晜鏌ㄥú鏍偓鐢殿攰閽栧嫰濡?

    濞撴艾顑呴々?OpenAI SDK 闁告瑯鍨甸崗妯绘交閺傛寧绀€閻庣數顢婇挅鍕晬?
        tool_call.function.name

    濞戞梻鍠庤ぐ鏌ユ嚄閸婄噥娼堕柟瀛樺灣濠婃垶娼浣稿簥闁?dict闁?
        tool_call["function"]["name"]
    """
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def role_value(message: AnyMessage) -> str:
    role = message.role

    if isinstance(role, MessageRole):
        return role.value

    return str(role)


def safe_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )


def safe_json_loads(value: str) -> Any:
    if not value:
        return {}

    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {
            "input": value,
        }


def normalize_message_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    return str(content)


def ensure_openai_message_content(message: dict[str, Any]) -> dict[str, Any]:
    if "content" not in message or message["content"] is None:
        message["content"] = ""
    else:
        message["content"] = normalize_message_content(message["content"])

    return message


def tool_call_to_openai(tool_call: ToolCall) -> dict[str, Any]:
    """
    PyWork ToolCall -> OpenAI tool_call 闁哄秶鍘х槐锟犲Υ?
    """
    return {
        "id": tool_call.call_id,
        "type": "function",
        "function": {
            "name": tool_call.tool_name,
            "arguments": safe_json_dumps(tool_call.arguments),
        },
    }


def openai_tool_call_to_tool_call(raw_tool_call: Any) -> ToolCall:
    """
    OpenAI tool_call -> PyWork ToolCall闁?

    闁稿繒鍘ч?dict 闁?SDK object闁?
    """
    call_id = str(
        get_attr_or_key(
            raw_tool_call,
            "id",
            "",
        )
    )

    function = get_attr_or_key(raw_tool_call, "function", {}) or {}

    name = str(
        get_attr_or_key(
            function,
            "name",
            "",
        )
    ).strip()

    raw_arguments = get_attr_or_key(
        function,
        "arguments",
        "{}",
    )

    if not name:
        raise MessageConversionError("OpenAI tool call missing function.name")

    if isinstance(raw_arguments, str):
        arguments = safe_json_loads(raw_arguments)
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = {
            "input": raw_arguments,
        }

    if not isinstance(arguments, dict):
        arguments = {
            "input": arguments,
        }

    return ToolCall(
        call_id=call_id or None,  # type: ignore[arg-type]
        tool_name=name,
        arguments=arguments,
        risk_level=ToolRiskLevel.LOW,
        metadata={
            "source": "openai_tool_call",
        },
    )



def tool_definition_to_openai(tool_definition: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool_definition["name"],
            "description": tool_definition.get("description", ""),
            "parameters": tool_definition.get(
                "input_schema",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        },
    }


def tool_definitions_to_openai(
    tool_definitions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        tool_definition_to_openai(definition)
        for definition in tool_definitions
    ]


def message_to_openai(message: AnyMessage) -> dict[str, Any] | None:
    """Convert a PyWork message into an OpenAI-compatible message."""
    role = role_value(message)

    if role == "system":
        return ensure_openai_message_content(
            {
                "role": "system",
                "content": normalize_message_content(
                    getattr(message, "content", "")
                ),
            }
        )

    if role == "user":
        return ensure_openai_message_content(
            {
                "role": "user",
                "content": normalize_message_content(
                    getattr(message, "content", "")
                ),
            }
        )

    if role == "assistant":
        payload: dict[str, Any] = {
            "role": "assistant",
            "content": normalize_message_content(
                getattr(message, "content", "")
            ),
        }

        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            payload["tool_calls"] = [
                tool_call_to_openai(call)
                for call in tool_calls
            ]

        return ensure_openai_message_content(payload)

    if role == "tool":
        return ensure_openai_message_content(
            {
                "role": "tool",
                "tool_call_id": getattr(message, "tool_call_id", ""),
                "content": normalize_message_content(
                    getattr(message, "content", "")
                ),
            }
        )

    if role == "error":
        return ensure_openai_message_content(
            {
                "role": "system",
                "content": "Runtime error: "
                + normalize_message_content(getattr(message, "content", "")),
            }
        )

    return None


def messages_to_openai(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for message in messages:
        item = message_to_openai(message)

        if item is not None:
            converted.append(
                ensure_openai_message_content(item)
            )

    return converted

def openai_message_to_pywork(raw_message: Any) -> AnyMessage:
    """
    OpenAI message -> PyWork Message闁?

    闁稿繒鍘ч?dict 闁?SDK object闁?
    """
    role = str(
        get_attr_or_key(
            raw_message,
            "role",
            "assistant",
        )
    )

    content = get_attr_or_key(
        raw_message,
        "content",
        "",
    )

    if content is None:
        content = ""

    if role == "system":
        return create_system_message(str(content))

    if role == "user":
        return create_user_message(str(content))

    if role == "assistant":
        raw_tool_calls = get_attr_or_key(
            raw_message,
            "tool_calls",
            None,
        )

        tool_calls: list[ToolCall] = []

        if raw_tool_calls:
            tool_calls = [
                openai_tool_call_to_tool_call(raw_call)
                for raw_call in raw_tool_calls
            ]

        return create_assistant_message(
            str(content),
            tool_calls=tool_calls,
            metadata={
                "source": "openai_message",
            },
        )

    if role == "tool":
        tool_call_id = str(
            get_attr_or_key(
                raw_message,
                "tool_call_id",
                "",
            )
        )

        name = str(
            get_attr_or_key(
                raw_message,
                "name",
                "unknown_tool",
            )
        )

        call = ToolCall(
            call_id=tool_call_id or None,  # type: ignore[arg-type]
            tool_name=name,
            arguments={},
            risk_level=ToolRiskLevel.LOW,
            metadata={
                "source": "openai_tool_message",
            },
        )

        result = ToolResult.success_result(
            call=call,
            content=str(content),
            data={
                "content": content,
            },
        )

        return create_tool_message(
            tool_result=result,
        )

    return create_error_message(
        f"Unsupported OpenAI message role: {role}",
        error_type="MessageConversionError",
    )


def openai_messages_to_pywork(raw_messages: Iterable[Any]) -> list[AnyMessage]:
    return [
        openai_message_to_pywork(message)
        for message in raw_messages
    ]


def tool_call_to_anthropic_block(tool_call: ToolCall) -> dict[str, Any]:
    """
    PyWork ToolCall -> Anthropic tool_use block闁?
    """
    return {
        "type": "tool_use",
        "id": tool_call.call_id,
        "name": tool_call.tool_name,
        "input": tool_call.arguments,
    }


def tool_definition_to_anthropic(tool_definition: dict[str, Any]) -> dict[str, Any]:
    """
    PyWork tool definition -> Anthropic tool schema闁?
    """
    return {
        "name": tool_definition["name"],
        "description": tool_definition.get("description", ""),
        "input_schema": tool_definition.get(
            "input_schema",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    }


def tool_definitions_to_anthropic(
    tool_definitions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        tool_definition_to_anthropic(definition)
        for definition in tool_definitions
    ]


def messages_to_anthropic_system(messages: list[AnyMessage]) -> str | None:
    """
    Anthropic Messages API 闁?system 闁哄嫷鍨板畷鐔兼偑椤掆偓閻⊙冣枔绾板绀夊☉鎾崇Т濠€?messages 闁轰焦澹嗙划宥夋煂鐏炵儵鍋?
    """
    system_parts = [
        message.content
        for message in messages
        if role_value(message) == "system" and message.content
    ]

    if not system_parts:
        return None

    return "\n\n".join(system_parts)


def message_to_anthropic(message: AnyMessage) -> dict[str, Any] | None:
    """
    PyWork Message -> Anthropic message闁?

    Anthropic 闁告瑯浜濈敮鎾矗?user / assistant 濞戞挶鍊楃悮?message闁?
    tool_result 閻熸洑鐒﹂弬渚€宕?user message content block 闂佹彃琚埀?
    """
    role = role_value(message)

    if role == "system":
        return None

    if role == "user":
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": message.content,
                }
            ],
        }

    if role == "assistant":
        assistant = message
        content_blocks: list[dict[str, Any]] = []

        if assistant.content:
            content_blocks.append(
                {
                    "type": "text",
                    "text": assistant.content,
                }
            )

        tool_calls = getattr(assistant, "tool_calls", [])

        for call in tool_calls:
            content_blocks.append(
                tool_call_to_anthropic_block(call)
            )

        return {
            "role": "assistant",
            "content": content_blocks,
        }

    if role == "tool":
        tool_call_id = getattr(message, "tool_call_id", "")

        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": message.content,
                }
            ],
        }

    if role == "error":
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Runtime error: {message.content}",
                }
            ],
        }

    return None


def messages_to_anthropic(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []

    for message in messages:
        item = message_to_anthropic(message)

        if item is not None:
            converted.append(item)

    return converted


def messages_to_anthropic_payload(
    messages: list[AnyMessage],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    PyWork messages -> Anthropic request payload闁?
    """
    payload: dict[str, Any] = {
        "messages": messages_to_anthropic(messages),
    }

    system = messages_to_anthropic_system(messages)

    if system:
        payload["system"] = system

    if model:
        payload["model"] = model

    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    if tools is not None:
        payload["tools"] = tools

    return payload


def anthropic_block_to_text(block: Any) -> str:
    block_type = get_attr_or_key(block, "type", "")

    if block_type == "text":
        return str(
            get_attr_or_key(
                block,
                "text",
                "",
            )
        )

    return ""


def anthropic_tool_use_block_to_tool_call(block: Any) -> ToolCall:
    block_id = str(
        get_attr_or_key(
            block,
            "id",
            "",
        )
    )

    name = str(
        get_attr_or_key(
            block,
            "name",
            "",
        )
    )

    input_data = get_attr_or_key(
        block,
        "input",
        {},
    )

    if not isinstance(input_data, dict):
        input_data = {
            "input": input_data,
        }

    if not name:
        raise MessageConversionError("Anthropic tool_use block missing name")

    return ToolCall(
        call_id=block_id or None,  # type: ignore[arg-type]
        tool_name=name,
        arguments=input_data,
        risk_level=ToolRiskLevel.LOW,
        metadata={
            "source": "anthropic_tool_use",
        },
    )


def anthropic_message_to_pywork(raw_message: Any) -> AssistantMessage:
    """
    Anthropic response message -> PyWork AssistantMessage闁?

    Anthropic assistant response content 闁?blocks闁?
    - text
    - tool_use
    """
    content_blocks = get_attr_or_key(
        raw_message,
        "content",
        [],
    )

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in content_blocks:
        block_type = get_attr_or_key(block, "type", "")

        if block_type == "text":
            text = anthropic_block_to_text(block)

            if text:
                text_parts.append(text)

        elif block_type == "tool_use":
            tool_calls.append(
                anthropic_tool_use_block_to_tool_call(block)
            )

    return create_assistant_message(
        "\n".join(text_parts),
        tool_calls=tool_calls,
        metadata={
            "source": "anthropic_message",
        },
    )


def to_provider_messages(
    messages: list[AnyMessage],
    *,
    provider_format: str,
) -> Any:
    """
    缂備胶鍠嶇粩鎾礂閵夈儱缍撻柨娑欑搷yWork messages -> provider messages闁?

    provider_format:
    - openai
    - openai_compatible
    - anthropic
    """
    normalized = provider_format.strip().lower()

    if normalized in {"openai", "openai_compatible", "deepseek", "qwen"}:
        return messages_to_openai(messages)

    if normalized == "anthropic":
        return messages_to_anthropic_payload(messages)

    raise MessageConversionError(f"Unsupported provider format: {provider_format!r}")


def from_provider_message(
    raw_message: Any,
    *,
    provider_format: str,
) -> AnyMessage:
    """
    缂備胶鍠嶇粩鎾礂閵夈儱缍撻柨娑欘劒rovider message -> PyWork message闁?
    """
    normalized = provider_format.strip().lower()

    if normalized in {"openai", "openai_compatible", "deepseek", "qwen"}:
        return openai_message_to_pywork(raw_message)

    if normalized == "anthropic":
        return anthropic_message_to_pywork(raw_message)

    raise MessageConversionError(f"Unsupported provider format: {provider_format!r}")


def main() -> int:
    from pywork.schemas.message_schema import (
        messages_to_json,
    )
    from pywork.schemas.tool_schema import create_tool_call

    system = create_system_message("You are PyWork, a coding agent.")
    user = create_user_message("Please call the echo tool.")

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello",
        },
    )

    assistant = create_assistant_message(
        "I will call the echo tool.",
        tool_calls=[call],
    )

    result = ToolResult.success_result(
        call=call,
        content="hello",
        data={
            "text": "hello",
        },
    )

    tool_message = create_tool_message(
        tool_result=result,
    )

    messages: list[AnyMessage] = [
        system,
        user,
        assistant,
        tool_message,
    ]

    print("PyWork messages:")
    print(messages_to_json(messages, indent=2))

    print("\nOpenAI messages:")
    print(
        json.dumps(
            messages_to_openai(messages),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nAnthropic system:")
    print(messages_to_anthropic_system(messages))

    print("\nAnthropic messages:")
    print(
        json.dumps(
            messages_to_anthropic(messages),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nOpenAI tools:")
    tool_definitions = [
        {
            "name": "echo",
            "description": "Echo input text.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                    }
                },
                "required": ["text"],
            },
        }
    ]

    print(
        json.dumps(
            tool_definitions_to_openai(tool_definitions),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nAnthropic tools:")
    print(
        json.dumps(
            tool_definitions_to_anthropic(tool_definitions),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
