from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pywork.schemas.tool_schema import ToolCall, ToolResult


class MessageRole(str, Enum):
    """
    PyWork 统一消息角色。
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ERROR = "error"


class MessageStatus(str, Enum):
    """
    消息状态。

    后面流式输出时会用到：
    - streaming：正在输出
    - completed：输出完成
    - error：输出失败
    """

    PENDING = "pending"
    STREAMING = "streaming"
    COMPLETED = "completed"
    ERROR = "error"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_message_id() -> str:
    return f"message_{uuid4().hex}"


class BaseMessage(BaseModel):
    """
    PyWork 消息基类。

    不直接实例化，一般使用：
    - SystemMessage
    - UserMessage
    - AssistantMessage
    - ToolMessage
    - ErrorMessage
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    message_id: str = Field(default_factory=new_message_id)
    role: MessageRole
    content: str = ""

    status: MessageStatus = MessageStatus.COMPLETED

    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    name: str | None = None
    token_estimate: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message_id")
    @classmethod
    def validate_message_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("message_id cannot be empty")

        return value

    @field_validator("content")
    @classmethod
    def normalize_content(cls, value: str) -> str:
        return value or ""

    @field_validator("token_estimate")
    @classmethod
    def validate_token_estimate(cls, value: int) -> int:
        if value < 0:
            raise ValueError("token_estimate cannot be negative")

        return value

    def touch(self) -> None:
        self.updated_at = utc_now()

    def append_content(self, delta: str) -> None:
        self.content += delta
        self.updated_at = utc_now()

    def set_status(self, status: MessageStatus | str) -> None:
        self.status = MessageStatus(status)
        self.updated_at = utc_now()

    def to_chat_payload(self) -> dict[str, Any]:
        """
        转成通用 LLM chat message 格式。

        后面不同 Provider 可以基于这个再适配：
        - OpenAI
        - Anthropic
        - DeepSeek / OpenAI-compatible
        """
        payload: dict[str, Any] = {
            "role": self.role,
            "content": self.content,
        }

        if self.name:
            payload["name"] = self.name

        return payload

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "name": self.name,
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


class SystemMessage(BaseMessage):
    """
    系统消息。

    一般用于：
    - system prompt
    - runtime 启动说明
    - 状态提示
    """

    role: Literal[MessageRole.SYSTEM] = MessageRole.SYSTEM


class UserMessage(BaseMessage):
    """
    用户消息。
    """

    role: Literal[MessageRole.USER] = MessageRole.USER


class AssistantMessage(BaseMessage):
    """
    助手消息。

    可以携带 tool_calls。
    """

    role: Literal[MessageRole.ASSISTANT] = MessageRole.ASSISTANT
    tool_calls: list[ToolCall] = Field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    def add_tool_call(self, call: ToolCall) -> None:
        self.tool_calls.append(call)
        self.updated_at = utc_now()

    def to_chat_payload(self) -> dict[str, Any]:
        payload = super().to_chat_payload()

        if self.tool_calls:
            payload["tool_calls"] = [
                call.model_dump(mode="json")
                for call in self.tool_calls
            ]

        return payload

    def to_log_dict(self) -> dict[str, Any]:
        data = super().to_log_dict()
        data["tool_calls"] = [
            call.model_dump(mode="json")
            for call in self.tool_calls
        ]
        return data


class ToolMessage(BaseMessage):
    """
    工具观察消息。

    它通常对应一次 ToolResult。
    """

    role: Literal[MessageRole.TOOL] = MessageRole.TOOL

    tool_call_id: str
    tool_name: str
    tool_result: ToolResult | None = None

    @field_validator("tool_call_id", "tool_name")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    def to_chat_payload(self) -> dict[str, Any]:
        payload = super().to_chat_payload()
        payload["tool_call_id"] = self.tool_call_id
        payload["name"] = self.tool_name
        return payload

    def to_log_dict(self) -> dict[str, Any]:
        data = super().to_log_dict()
        data["tool_call_id"] = self.tool_call_id
        data["tool_name"] = self.tool_name
        data["tool_result"] = (
            self.tool_result.model_dump(mode="json")
            if self.tool_result is not None
            else None
        )
        return data


class ErrorMessage(BaseMessage):
    """
    错误消息。
    """

    role: Literal[MessageRole.ERROR] = MessageRole.ERROR
    status: Literal[MessageStatus.ERROR] = MessageStatus.ERROR

    error_type: str | None = None

    def to_log_dict(self) -> dict[str, Any]:
        data = super().to_log_dict()
        data["error_type"] = self.error_type
        return data


AnyMessage: TypeAlias = (
    SystemMessage
    | UserMessage
    | AssistantMessage
    | ToolMessage
    | ErrorMessage
)


def create_system_message(
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> SystemMessage:
    return SystemMessage(
        content=content,
        metadata=metadata or {},
    )


def create_user_message(
    content: str,
    *,
    token_estimate: int = 0,
    metadata: dict[str, Any] | None = None,
) -> UserMessage:
    return UserMessage(
        content=content,
        token_estimate=token_estimate,
        metadata=metadata or {},
    )


def create_assistant_message(
    content: str = "",
    *,
    tool_calls: list[ToolCall] | None = None,
    status: MessageStatus | str = MessageStatus.COMPLETED,
    token_estimate: int = 0,
    metadata: dict[str, Any] | None = None,
) -> AssistantMessage:
    return AssistantMessage(
        content=content,
        tool_calls=tool_calls or [],
        status=MessageStatus(status),
        token_estimate=token_estimate,
        metadata=metadata or {},
    )


def create_tool_message(
    *,
    tool_result: ToolResult,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolMessage:
    return ToolMessage(
        content=content if content is not None else tool_result.content,
        tool_call_id=tool_result.call_id,
        tool_name=tool_result.tool_name,
        tool_result=tool_result,
        metadata=metadata or {},
    )


def create_error_message(
    content: str,
    *,
    error_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ErrorMessage:
    return ErrorMessage(
        content=content,
        error_type=error_type,
        metadata=metadata or {},
    )


def message_from_dict(data: dict[str, Any]) -> AnyMessage:
    """
    根据 role 反序列化消息。
    """
    role = data.get("role")

    if role == MessageRole.SYSTEM or role == MessageRole.SYSTEM.value:
        return SystemMessage.model_validate(data)

    if role == MessageRole.USER or role == MessageRole.USER.value:
        return UserMessage.model_validate(data)

    if role == MessageRole.ASSISTANT or role == MessageRole.ASSISTANT.value:
        return AssistantMessage.model_validate(data)

    if role == MessageRole.TOOL or role == MessageRole.TOOL.value:
        return ToolMessage.model_validate(data)

    if role == MessageRole.ERROR or role == MessageRole.ERROR.value:
        return ErrorMessage.model_validate(data)

    raise ValueError(f"Unknown message role: {role!r}")


def messages_to_chat_payload(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    """
    转成通用 chat payload。
    """
    return [
        message.to_chat_payload()
        for message in messages
    ]


def messages_to_log_dict(messages: list[AnyMessage]) -> list[dict[str, Any]]:
    return [
        message.to_log_dict()
        for message in messages
    ]


def messages_to_json(
    messages: list[AnyMessage],
    *,
    indent: int = 2,
) -> str:
    return json.dumps(
        messages_to_log_dict(messages),
        ensure_ascii=False,
        indent=indent,
        default=str,
    )


def main() -> int:
    from pywork.schemas.tool_schema import ToolRiskLevel, ToolResult, create_tool_call

    system_message = create_system_message(
        "You are PyWork, a coding agent."
    )

    user_message = create_user_message(
        "请执行 echo 工具。",
        token_estimate=4,
    )

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello",
        },
        risk_level=ToolRiskLevel.SAFE,
    )

    assistant_message = create_assistant_message(
        "我将调用 echo 工具。",
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

    error_message = create_error_message(
        "demo error",
        error_type="DemoError",
    )

    messages: list[AnyMessage] = [
        system_message,
        user_message,
        assistant_message,
        tool_message,
        error_message,
    ]

    print("Messages:")
    print(messages_to_json(messages, indent=2))

    print("\nChat payload:")
    print(
        json.dumps(
            messages_to_chat_payload(messages),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    print("\nRestore first message:")
    restored = message_from_dict(system_message.model_dump(mode="json"))
    print(restored.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())