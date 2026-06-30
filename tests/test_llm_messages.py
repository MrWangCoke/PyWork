from __future__ import annotations

from pywork.llm.messages import (
    messages_to_openai,
    openai_message_to_pywork,
)
from pywork.schemas.message_schema import create_assistant_message, create_tool_message
from pywork.schemas.tool_schema import ToolResult, create_tool_call


def test_openai_assistant_tool_call_message_always_has_string_content() -> None:
    call = create_tool_call(
        tool_name="file_read",
        arguments={
            "path": "README.md",
        },
    )
    assistant = create_assistant_message(
        "",
        tool_calls=[call],
    )

    payload = messages_to_openai([assistant])[0]

    assert payload["role"] == "assistant"
    assert payload["content"] == ""
    assert isinstance(payload["content"], str)
    assert payload["tool_calls"][0]["id"] == call.call_id


def test_openai_tool_message_always_has_string_content_and_call_id() -> None:
    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello",
        },
    )
    result = ToolResult.success_result(
        call=call,
        content="hello",
    )
    message = create_tool_message(
        tool_result=result,
        content=None,
    )

    payload = messages_to_openai([message])[0]

    assert payload == {
        "role": "tool",
        "tool_call_id": call.call_id,
        "content": "hello",
    }


def test_openai_assistant_response_with_null_content_becomes_empty_string() -> None:
    message = openai_message_to_pywork(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "file_read",
                        "arguments": '{"path": "README.md"}',
                    },
                }
            ],
        }
    )

    assert message.content == ""
    assert message.tool_calls
    assert message.tool_calls[0].tool_name == "file_read"
