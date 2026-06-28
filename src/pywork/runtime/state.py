from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pywork.schemas.tool_schema import ToolCall, ToolResult, create_tool_call


AgentMessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
    "error",
]


class AgentStatus(str, Enum):
    """
    Agent 当前运行状态。
    """

    IDLE = "idle"
    THINKING = "thinking"
    RUNNING_TOOL = "running_tool"
    WAITING_PERMISSION = "waiting_permission"
    FINISHED = "finished"
    ERROR = "error"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_message_id() -> str:
    return f"agent_msg_{uuid4().hex}"


def new_checkpoint_id() -> str:
    return f"checkpoint_{uuid4().hex}"


@dataclass
class AgentMessage:
    """
    Agent 内部消息。

    这个消息结构先保持轻量，不直接绑定 LangChain / LangGraph。
    后面如果接 LangGraph，可以再转换成对应的 message 格式。
    """

    role: AgentMessageRole
    content: str
    message_id: str = field(default_factory=new_message_id)
    created_at: datetime = field(default_factory=utc_now)

    name: str | None = None
    tool_call_id: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "name": self.name,
            "tool_call_id": self.tool_call_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentMessage:
        created_at_raw = data.get("created_at")

        if isinstance(created_at_raw, str):
            created_at = datetime.fromisoformat(created_at_raw)
        elif isinstance(created_at_raw, datetime):
            created_at = created_at_raw
        else:
            created_at = utc_now()

        return cls(
            role=data["role"],
            content=data.get("content", ""),
            message_id=data.get("message_id", new_message_id()),
            created_at=created_at,
            name=data.get("name"),
            tool_call_id=data.get("tool_call_id"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class AgentState:
    """
    Agent 运行状态。

    这是 Runtime 层的核心状态对象。

    核心字段：
    - messages：当前 Agent 上下文消息
    - tool_calls：模型请求执行的工具调用
    - tool_results：工具执行结果
    - status：当前运行状态
    - iteration：当前 Agent 循环次数
    - checkpoint_id：当前检查点 ID
    """

    messages: list[AgentMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    status: AgentStatus = AgentStatus.IDLE

    iteration: int = 0
    max_iterations: int = 20

    checkpoint_id: str = field(default_factory=new_checkpoint_id)

    current_tool_call_id: str | None = None
    last_error: str | None = None

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def set_status(self, status: AgentStatus | str) -> None:
        self.status = AgentStatus(status)
        self.touch()

    def set_idle(self) -> None:
        self.set_status(AgentStatus.IDLE)

    def set_thinking(self) -> None:
        self.set_status(AgentStatus.THINKING)

    def set_running_tool(self, tool_call_id: str | None = None) -> None:
        self.current_tool_call_id = tool_call_id
        self.set_status(AgentStatus.RUNNING_TOOL)

    def set_waiting_permission(self, tool_call_id: str | None = None) -> None:
        self.current_tool_call_id = tool_call_id
        self.set_status(AgentStatus.WAITING_PERMISSION)

    def set_finished(self) -> None:
        self.current_tool_call_id = None
        self.set_status(AgentStatus.FINISHED)

    def set_error(self, error: str) -> None:
        self.last_error = error
        self.set_status(AgentStatus.ERROR)

    def set_cancelled(self, reason: str = "cancelled") -> None:
        self.last_error = reason
        self.current_tool_call_id = None
        self.set_status(AgentStatus.CANCELLED)

    def next_iteration(self) -> int:
        self.iteration += 1
        self.touch()
        return self.iteration

    def reset_iteration(self) -> None:
        self.iteration = 0
        self.touch()

    def can_continue(self) -> bool:
        return self.iteration < self.max_iterations and self.status not in {
            AgentStatus.FINISHED,
            AgentStatus.ERROR,
            AgentStatus.CANCELLED,
        }

    def new_checkpoint(self) -> str:
        self.checkpoint_id = new_checkpoint_id()
        self.touch()
        return self.checkpoint_id

    def add_message(
        self,
        role: AgentMessageRole,
        content: str,
        *,
        name: str | None = None,
        tool_call_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        message = AgentMessage(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
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
    ) -> AgentMessage:
        return self.add_message(
            "system",
            content,
            metadata=metadata,
        )

    def add_user_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        return self.add_message(
            "user",
            content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        return self.add_message(
            "assistant",
            content,
            metadata=metadata,
        )

    def add_tool_message(
        self,
        content: str,
        *,
        tool_call_id: str | None = None,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        return self.add_message(
            "tool",
            content,
            name=name,
            tool_call_id=tool_call_id,
            metadata=metadata,
        )

    def add_error_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> AgentMessage:
        self.set_error(content)

        return self.add_message(
            "error",
            content,
            metadata=metadata,
        )

    def clear_messages(self) -> None:
        self.messages.clear()
        self.touch()

    def get_last_message(self) -> AgentMessage | None:
        if not self.messages:
            return None

        return self.messages[-1]

    def get_messages_by_role(
        self,
        role: AgentMessageRole,
    ) -> list[AgentMessage]:
        return [
            message
            for message in self.messages
            if message.role == role
        ]

    def add_tool_call(self, call: ToolCall) -> ToolCall:
        self.tool_calls.append(call)
        self.current_tool_call_id = call.call_id
        self.set_running_tool(call.call_id)
        self.touch()

        return call

    def add_tool_result(self, result: ToolResult) -> ToolResult:
        self.tool_results.append(result)

        if self.current_tool_call_id == result.call_id:
            self.current_tool_call_id = None

        if result.success:
            self.set_idle()
        else:
            self.set_error(result.error or "tool failed")

        self.add_tool_message(
            result.content or result.error or "",
            tool_call_id=result.call_id,
            name=result.tool_name,
            metadata={
                "success": result.success,
                "status": result.status,
                "duration_ms": result.duration_ms,
            },
        )

        self.touch()

        return result

    def get_tool_call(self, call_id: str) -> ToolCall | None:
        for call in self.tool_calls:
            if call.call_id == call_id:
                return call

        return None

    def get_tool_result(self, call_id: str) -> ToolResult | None:
        for result in self.tool_results:
            if result.call_id == call_id:
                return result

        return None

    def get_pending_tool_calls(self) -> list[ToolCall]:
        finished_call_ids = {
            result.call_id
            for result in self.tool_results
        }

        return [
            call
            for call in self.tool_calls
            if call.call_id not in finished_call_ids
        ]

    def has_pending_tool_calls(self) -> bool:
        return bool(self.get_pending_tool_calls())

    def get_current_tool_call(self) -> ToolCall | None:
        if self.current_tool_call_id is None:
            return None

        return self.get_tool_call(self.current_tool_call_id)

    def clear_tools(self) -> None:
        self.tool_calls.clear()
        self.tool_results.clear()
        self.current_tool_call_id = None
        self.touch()

    def reset_runtime(self) -> None:
        """
        重置运行过程状态，但不清空 messages。
        """
        self.tool_calls.clear()
        self.tool_results.clear()
        self.current_tool_call_id = None
        self.last_error = None
        self.iteration = 0
        self.status = AgentStatus.IDLE
        self.new_checkpoint()
        self.touch()

    def to_messages_payload(self) -> list[dict[str, Any]]:
        """
        转成给模型层使用的消息格式。

        当前先返回通用 dict。
        后面接 LangChain / OpenAI / Anthropic 时再做适配。
        """
        return [
            {
                "role": message.role,
                "content": message.content,
            }
            for message in self.messages
            if message.role in {"system", "user", "assistant", "tool"}
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "checkpoint_id": self.checkpoint_id,
            "message_count": len(self.messages),
            "tool_call_count": len(self.tool_calls),
            "tool_result_count": len(self.tool_results),
            "pending_tool_call_count": len(self.get_pending_tool_calls()),
            "current_tool_call_id": self.current_tool_call_id,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "tool_calls": [
                call.model_dump(mode="json")
                for call in self.tool_calls
            ],
            "tool_results": [
                result.model_dump(mode="json")
                for result in self.tool_results
            ],
            "status": self.status.value,
            "iteration": self.iteration,
            "max_iterations": self.max_iterations,
            "checkpoint_id": self.checkpoint_id,
            "current_tool_call_id": self.current_tool_call_id,
            "last_error": self.last_error,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentState:
        state = cls(
            messages=[
                AgentMessage.from_dict(message)
                for message in data.get("messages", [])
            ],
            tool_calls=[
                ToolCall.model_validate(call)
                for call in data.get("tool_calls", [])
            ],
            tool_results=[
                ToolResult.model_validate(result)
                for result in data.get("tool_results", [])
            ],
            status=AgentStatus(data.get("status", AgentStatus.IDLE)),
            iteration=data.get("iteration", 0),
            max_iterations=data.get("max_iterations", 20),
            checkpoint_id=data.get("checkpoint_id", new_checkpoint_id()),
            current_tool_call_id=data.get("current_tool_call_id"),
            last_error=data.get("last_error"),
            metadata=data.get("metadata", {}),
        )

        created_at_raw = data.get("created_at")
        updated_at_raw = data.get("updated_at")

        if created_at_raw:
            state.created_at = datetime.fromisoformat(created_at_raw)

        if updated_at_raw:
            state.updated_at = datetime.fromisoformat(updated_at_raw)

        return state


def create_agent_state(
    *,
    system_prompt: str | None = None,
    max_iterations: int = 20,
    checkpoint_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentState:
    state = AgentState(
        max_iterations=max_iterations,
        checkpoint_id=checkpoint_id or new_checkpoint_id(),
        metadata=metadata or {},
    )

    if system_prompt:
        state.add_system_message(system_prompt)

    return state


def main() -> int:
    state = create_agent_state(
        system_prompt="You are PyWork, a coding agent.",
        max_iterations=5,
        metadata={
            "demo": True,
        },
    )

    state.add_user_message("请查看当前项目状态。")

    state.next_iteration()
    state.set_thinking()

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "demo tool call",
        },
    )

    state.add_tool_call(call)

    result = ToolResult.success_result(
        call=call,
        content="demo tool result",
        data={
            "text": "demo tool call",
        },
    )

    state.add_tool_result(result)

    state.add_assistant_message("当前 runtime/state.py 工作正常。")
    state.set_finished()

    print("AgentState summary:")
    print(json.dumps(state.summary(), ensure_ascii=False, indent=2))

    print("\nAgentState full state:")
    print(state.to_json(indent=2))

    restored = AgentState.from_dict(state.to_dict())

    print("\nRestored AgentState summary:")
    print(json.dumps(restored.summary(), ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())