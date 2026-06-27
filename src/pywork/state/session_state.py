from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pywork.schemas.tool_schema import ToolCall, ToolResult


MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
    "error",
]


class SessionStatus(str, Enum):
    """
    会话状态。
    """

    ACTIVE = "active"
    IDLE = "idle"
    THINKING = "thinking"
    RUNNING_TOOL = "running_tool"
    ERROR = "error"
    CLOSED = "closed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_session_id() -> str:
    return f"session_{uuid4().hex}"


def new_message_id() -> str:
    return f"msg_{uuid4().hex}"


@dataclass
class SessionTokenUsage:
    """
    会话级 Token 用量。

    这里先记录估算值。
    后面接入 LLM Provider 后，再使用真实 usage 覆盖。
    """

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.input_tokens = max(0, self.input_tokens + input_tokens)
        self.output_tokens = max(0, self.output_tokens + output_tokens)

    def reset(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class SessionMessage:
    """
    会话消息。

    注意：
    这里是状态层消息，不依赖 TUI。
    TUI 的 ChatPanel 可以根据这些消息渲染界面。
    """

    role: MessageRole
    content: str
    message_id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)
    token_estimate: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "token_estimate": self.token_estimate,
            "metadata": self.metadata,
        }


@dataclass
class SessionState:
    """
    会话级状态。

    一个 SessionState 对应一次 PyWork 会话。

    负责保存：
    - 消息列表
    - token 用量
    - 工具调用
    - 工具结果
    - 当前会话状态
    """

    session_id: str = field(default_factory=new_session_id)

    workspace_path: str = "."
    project_root: str = "."
    title: str = "New Session"

    status: SessionStatus = SessionStatus.ACTIVE

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    closed_at: datetime | None = None

    messages: list[SessionMessage] = field(default_factory=list)

    token_usage: SessionTokenUsage = field(default_factory=SessionTokenUsage)

    tool_calls: dict[str, ToolCall] = field(default_factory=dict)
    tool_results: dict[str, ToolResult] = field(default_factory=dict)
    current_tool_call_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def set_status(self, status: SessionStatus | str) -> None:
        self.status = SessionStatus(status)
        self.touch()

    def set_title(self, title: str) -> None:
        title = title.strip()

        if not title:
            title = "New Session"

        self.title = title
        self.touch()

    def close(self) -> None:
        self.status = SessionStatus.CLOSED
        self.closed_at = utc_now()
        self.touch()

    @property
    def is_closed(self) -> bool:
        return self.status == SessionStatus.CLOSED

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def tool_call_count(self) -> int:
        return len(self.tool_calls)

    @property
    def tool_result_count(self) -> int:
        return len(self.tool_results)

    def add_message(
        self,
        role: MessageRole,
        content: str,
        *,
        token_estimate: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = SessionMessage(
            role=role,
            content=content,
            token_estimate=max(0, token_estimate),
            metadata=metadata or {},
        )

        self.messages.append(message)
        self.touch()

        return message

    def add_system_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        return self.add_message(
            "system",
            content,
            metadata=metadata,
        )

    def add_user_message(
        self,
        content: str,
        *,
        token_estimate: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        return self.add_message(
            "user",
            content,
            token_estimate=token_estimate,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        content: str,
        *,
        token_estimate: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        return self.add_message(
            "assistant",
            content,
            token_estimate=token_estimate,
            metadata=metadata,
        )

    def add_tool_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        return self.add_message(
            "tool",
            content,
            metadata=metadata,
        )

    def add_error_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        self.set_status(SessionStatus.ERROR)

        return self.add_message(
            "error",
            content,
            metadata=metadata,
        )

    def get_last_message(self) -> SessionMessage | None:
        if not self.messages:
            return None

        return self.messages[-1]

    def get_messages_by_role(self, role: MessageRole) -> list[SessionMessage]:
        return [
            message
            for message in self.messages
            if message.role == role
        ]

    def clear_messages(self) -> None:
        self.messages.clear()
        self.touch()

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.token_usage.add(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.touch()

    def reset_token_usage(self) -> None:
        self.token_usage.reset()
        self.touch()

    def add_tool_call(self, call: ToolCall) -> ToolCall:
        self.tool_calls[call.call_id] = call
        self.current_tool_call_id = call.call_id
        self.set_status(SessionStatus.RUNNING_TOOL)
        self.touch()

        return call

    def add_tool_result(self, result: ToolResult) -> ToolResult:
        self.tool_results[result.call_id] = result

        if self.current_tool_call_id == result.call_id:
            self.current_tool_call_id = None

        if result.success:
            self.set_status(SessionStatus.IDLE)
        else:
            self.set_status(SessionStatus.ERROR)

        self.touch()

        return result

    def get_tool_call(self, call_id: str) -> ToolCall | None:
        return self.tool_calls.get(call_id)

    def get_tool_result(self, call_id: str) -> ToolResult | None:
        return self.tool_results.get(call_id)

    def has_pending_tool_call(self) -> bool:
        return bool(self.get_pending_tool_call_ids())

    def get_pending_tool_call_ids(self) -> list[str]:
        return [
            call_id
            for call_id in self.tool_calls
            if call_id not in self.tool_results
        ]

    def get_current_tool_call(self) -> ToolCall | None:
        if self.current_tool_call_id is None:
            return None

        return self.get_tool_call(self.current_tool_call_id)

    def clear_tools(self) -> None:
        self.tool_calls.clear()
        self.tool_results.clear()
        self.current_tool_call_id = None
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_path": self.workspace_path,
            "project_root": self.project_root,
            "title": self.title,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "tool_result_count": self.tool_result_count,
            "current_tool_call_id": self.current_tool_call_id,
            "pending_tool_call_ids": self.get_pending_tool_call_ids(),
            "token_usage": self.token_usage.to_dict(),
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "tool_calls": [
                call.model_dump(mode="json")
                for call in self.tool_calls.values()
            ],
            "tool_results": [
                result.model_dump(mode="json")
                for result in self.tool_results.values()
            ],
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "status": self.status.value,
            "message_count": self.message_count,
            "tool_call_count": self.tool_call_count,
            "tool_result_count": self.tool_result_count,
            "pending_tool_call_count": len(self.get_pending_tool_call_ids()),
            "token_usage": self.token_usage.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


def create_session_state(
    *,
    workspace_path: str = ".",
    project_root: str = ".",
    title: str = "New Session",
    metadata: dict[str, Any] | None = None,
) -> SessionState:
    return SessionState(
        workspace_path=workspace_path,
        project_root=project_root,
        title=title,
        metadata=metadata or {},
    )


def main() -> int:
    from pywork.schemas.tool_schema import ToolRiskLevel, create_tool_call

    session = create_session_state(
        workspace_path=".",
        project_root=".",
        title="Demo Session",
    )

    session.add_system_message("PyWork session started.")
    session.add_user_message(
        "hello",
        token_estimate=2,
    )
    session.add_assistant_message(
        "你好，我是 PyWork。",
        token_estimate=5,
    )

    session.add_token_usage(
        input_tokens=2,
        output_tokens=5,
    )

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello",
        },
        risk_level=ToolRiskLevel.SAFE,
    )

    session.add_tool_call(call)

    result = ToolResult.success_result(
        call=call,
        content="hello",
        data={
            "text": "hello",
        },
    )

    session.add_tool_result(result)

    print("Session summary:")
    print(json.dumps(session.summary(), ensure_ascii=False, indent=2))

    print("\nSession full state:")
    print(session.to_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())