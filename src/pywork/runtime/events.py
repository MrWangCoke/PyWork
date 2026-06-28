from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any, Literal
from uuid import uuid4

from pywork.schemas.tool_schema import ToolCall, ToolResult, create_tool_call


MessageRole = Literal[
    "system",
    "user",
    "assistant",
    "tool",
    "error",
]


class RuntimeEventType(str, Enum):
    """
    Runtime 事件类型。

    核心事件：
    - MESSAGE：完整消息
    - MESSAGE_DELTA：流式消息片段
    - TOOL_CALL：工具调用
    - TOOL_RESULT：工具结果
    - ERROR：错误
    - CHECKPOINT：检查点
    - STATUS：状态变化
    - LIFECYCLE：生命周期事件
    """

    MESSAGE = "message"
    MESSAGE_DELTA = "message_delta"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    STATUS = "status"
    LIFECYCLE = "lifecycle"


class RuntimeEventSource(str, Enum):
    """
    Runtime 事件来源。
    """

    USER = "user"
    TUI = "tui"
    CONTROLLER = "controller"
    ENGINE = "engine"
    GRAPH = "graph"
    LLM = "llm"
    TOOL = "tool"
    SYSTEM = "system"


class RuntimeLifecycleEvent(str, Enum):
    STARTED = "started"
    PAUSED = "paused"
    RESUMED = "resumed"
    FINISHED = "finished"
    ABORT_REQUESTED = "abort_requested"
    ABORTED = "aborted"
    ERROR = "error"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_event_id() -> str:
    return f"event_{uuid4().hex}"


def new_run_id() -> str:
    return f"run_{uuid4().hex}"


def new_checkpoint_id() -> str:
    return f"checkpoint_{uuid4().hex}"


@dataclass(frozen=True)
class RuntimeEvent:
    """
    PyWork Runtime 统一事件。

    它可以表示：
    - assistant 消息
    - 流式 token 片段
    - 工具调用
    - 工具结果
    - 错误
    - checkpoint
    - status / lifecycle
    """

    event_type: RuntimeEventType
    source: RuntimeEventSource = RuntimeEventSource.SYSTEM

    event_id: str = field(default_factory=new_event_id)
    run_id: str | None = None
    session_id: str | None = None
    checkpoint_id: str | None = None

    created_at: datetime = field(default_factory=utc_now)

    role: MessageRole | None = None
    content: str = ""
    delta: str = ""

    status: str | None = None
    lifecycle: RuntimeLifecycleEvent | str | None = None

    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None

    error: str | None = None
    error_type: str | None = None

    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def message(
        cls,
        *,
        role: MessageRole,
        content: str,
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.MESSAGE,
            source=RuntimeEventSource(source),
            role=role,
            content=content,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def message_delta(
        cls,
        *,
        delta: str,
        role: MessageRole = "assistant",
        source: RuntimeEventSource | str = RuntimeEventSource.LLM,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.MESSAGE_DELTA,
            source=RuntimeEventSource(source),
            role=role,
            delta=delta,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def tool_call_event(
        cls,
        *,
        tool_call: ToolCall,
        source: RuntimeEventSource | str = RuntimeEventSource.GRAPH,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.TOOL_CALL,
            source=RuntimeEventSource(source),
            tool_call=tool_call,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def tool_result_event(
        cls,
        *,
        tool_result: ToolResult,
        source: RuntimeEventSource | str = RuntimeEventSource.TOOL,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.TOOL_RESULT,
            source=RuntimeEventSource(source),
            tool_result=tool_result,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def error_event(
        cls,
        *,
        error: str,
        error_type: str | None = None,
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.ERROR,
            source=RuntimeEventSource(source),
            error=error,
            error_type=error_type,
            content=error,
            role="error",
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def checkpoint_event(
        cls,
        *,
        checkpoint_id: str,
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
        run_id: str | None = None,
        session_id: str | None = None,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.CHECKPOINT,
            source=RuntimeEventSource(source),
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            session_id=session_id,
            data=data or {},
            metadata=metadata or {},
        )

    @classmethod
    def status_event(
        cls,
        *,
        status: str,
        content: str = "",
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.STATUS,
            source=RuntimeEventSource(source),
            status=status,
            content=content,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    @classmethod
    def lifecycle_event(
        cls,
        *,
        lifecycle: RuntimeLifecycleEvent | str,
        content: str = "",
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return cls(
            event_type=RuntimeEventType.LIFECYCLE,
            source=RuntimeEventSource(source),
            lifecycle=lifecycle,
            content=content,
            run_id=run_id,
            session_id=session_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "source": self.source.value,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "checkpoint_id": self.checkpoint_id,
            "created_at": self.created_at.isoformat(),
            "role": self.role,
            "content": self.content,
            "delta": self.delta,
            "status": self.status,
            "lifecycle": (
                self.lifecycle.value
                if isinstance(self.lifecycle, RuntimeLifecycleEvent)
                else self.lifecycle
            ),
            "tool_call": (
                self.tool_call.model_dump(mode="json")
                if self.tool_call is not None
                else None
            ),
            "tool_result": (
                self.tool_result.model_dump(mode="json")
                if self.tool_result is not None
                else None
            ),
            "error": self.error,
            "error_type": self.error_type,
            "data": self.data,
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    def compact_text(self) -> str:
        """
        用于日志 / 调试的简洁文本。
        """
        if self.event_type == RuntimeEventType.MESSAGE:
            return f"[message:{self.role}] {self.content}"

        if self.event_type == RuntimeEventType.MESSAGE_DELTA:
            return f"[delta:{self.role}] {self.delta}"

        if self.event_type == RuntimeEventType.TOOL_CALL and self.tool_call:
            return f"[tool_call] {self.tool_call.tool_name} {self.tool_call.arguments}"

        if self.event_type == RuntimeEventType.TOOL_RESULT and self.tool_result:
            return (
                f"[tool_result] {self.tool_result.tool_name} "
                f"success={self.tool_result.success}"
            )

        if self.event_type == RuntimeEventType.ERROR:
            return f"[error] {self.error}"

        if self.event_type == RuntimeEventType.CHECKPOINT:
            return f"[checkpoint] {self.checkpoint_id}"

        if self.event_type == RuntimeEventType.STATUS:
            return f"[status] {self.status}: {self.content}"

        if self.event_type == RuntimeEventType.LIFECYCLE:
            return f"[lifecycle] {self.lifecycle}: {self.content}"

        return f"[{self.event_type.value}]"


RuntimeEventHandler = Callable[[RuntimeEvent], None | Awaitable[None]]


class RuntimeEventBus:
    """
    Runtime 事件总线。

    支持：
    - subscribe(handler)
    - unsubscribe(handler)
    - emit(event)
    - await emit_async(event)
    - history()
    """

    def __init__(
        self,
        *,
        keep_history: bool = True,
        max_history: int = 1000,
    ) -> None:
        self.keep_history = keep_history
        self.max_history = max_history

        self._handlers: list[RuntimeEventHandler] = []
        self._history: list[RuntimeEvent] = []
        self._lock = RLock()

    def subscribe(self, handler: RuntimeEventHandler) -> Callable[[], None]:
        """
        订阅事件。

        返回一个 unsubscribe 函数。
        """
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

        def unsubscribe() -> None:
            self.unsubscribe(handler)

        return unsubscribe

    def unsubscribe(self, handler: RuntimeEventHandler) -> None:
        with self._lock:
            if handler in self._handlers:
                self._handlers.remove(handler)

    def clear_subscribers(self) -> None:
        with self._lock:
            self._handlers.clear()

    def _record_history(self, event: RuntimeEvent) -> None:
        if not self.keep_history:
            return

        self._history.append(event)

        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history :]

    def emit(self, event: RuntimeEvent) -> RuntimeEvent:
        """
        同步发布事件。

        如果 handler 是 async 函数：
        - 有运行中的 event loop：create_task
        - 没有 event loop：asyncio.run
        """
        with self._lock:
            handlers = list(self._handlers)

        self._record_history(event)

        for handler in handlers:
            result = handler(event)

            if inspect.isawaitable(result):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(result)
                else:
                    loop.create_task(result)

        return event

    async def emit_async(self, event: RuntimeEvent) -> RuntimeEvent:
        """
        异步发布事件。

        会等待 async handler 执行完成。
        """
        with self._lock:
            handlers = list(self._handlers)

        self._record_history(event)

        awaitables: list[Awaitable[None]] = []

        for handler in handlers:
            result = handler(event)

            if inspect.isawaitable(result):
                awaitables.append(result)

        if awaitables:
            await asyncio.gather(*awaitables)

        return event

    def history(self) -> list[RuntimeEvent]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()

    def filter_history(
        self,
        *,
        event_type: RuntimeEventType | str | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
    ) -> list[RuntimeEvent]:
        events = self.history()

        if event_type is not None:
            target_type = RuntimeEventType(event_type)
            events = [
                event
                for event in events
                if event.event_type == target_type
            ]

        if run_id is not None:
            events = [
                event
                for event in events
                if event.run_id == run_id
            ]

        if session_id is not None:
            events = [
                event
                for event in events
                if event.session_id == session_id
            ]

        return events

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscriber_count": len(self._handlers),
            "history_count": len(self._history),
            "keep_history": self.keep_history,
            "max_history": self.max_history,
            "history": [
                event.to_dict()
                for event in self._history
            ],
        }


_DEFAULT_EVENT_BUS: RuntimeEventBus | None = None


def get_default_event_bus() -> RuntimeEventBus:
    global _DEFAULT_EVENT_BUS

    if _DEFAULT_EVENT_BUS is None:
        _DEFAULT_EVENT_BUS = RuntimeEventBus()

    return _DEFAULT_EVENT_BUS


def reset_default_event_bus() -> RuntimeEventBus:
    global _DEFAULT_EVENT_BUS

    _DEFAULT_EVENT_BUS = RuntimeEventBus()
    return _DEFAULT_EVENT_BUS


async def demo() -> None:
    bus = RuntimeEventBus()
    run_id = new_run_id()
    checkpoint_id = new_checkpoint_id()

    def print_event(event: RuntimeEvent) -> None:
        print(event.compact_text())

    async def async_print_event(event: RuntimeEvent) -> None:
        await asyncio.sleep(0)
        print(f"async handled: {event.event_type.value}")

    bus.subscribe(print_event)
    bus.subscribe(async_print_event)

    await bus.emit_async(
        RuntimeEvent.lifecycle_event(
            lifecycle=RuntimeLifecycleEvent.STARTED,
            content="runtime started",
            run_id=run_id,
        )
    )

    await bus.emit_async(
        RuntimeEvent.message(
            role="user",
            content="hello",
            source=RuntimeEventSource.USER,
            run_id=run_id,
        )
    )

    await bus.emit_async(
        RuntimeEvent.message_delta(
            delta="你好",
            run_id=run_id,
        )
    )

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello",
        },
    )

    await bus.emit_async(
        RuntimeEvent.tool_call_event(
            tool_call=call,
            run_id=run_id,
        )
    )

    result = ToolResult.success_result(
        call=call,
        content="hello",
        data={
            "text": "hello",
        },
    )

    await bus.emit_async(
        RuntimeEvent.tool_result_event(
            tool_result=result,
            run_id=run_id,
        )
    )

    await bus.emit_async(
        RuntimeEvent.checkpoint_event(
            checkpoint_id=checkpoint_id,
            run_id=run_id,
            data={
                "iteration": 1,
            },
        )
    )

    await bus.emit_async(
        RuntimeEvent.error_event(
            error="demo error",
            error_type="DemoError",
            run_id=run_id,
        )
    )

    print("\nHistory count:")
    print(len(bus.history()))

    print("\nEventBus JSON:")
    print(json.dumps(bus.to_dict(), ensure_ascii=False, indent=2, default=str))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())