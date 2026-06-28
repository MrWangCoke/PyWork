from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pywork.runtime.events import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeEventSource,
    RuntimeEventType,
    RuntimeLifecycleEvent,
    get_default_event_bus,
    new_checkpoint_id,
    new_run_id,
)
from pywork.schemas.tool_schema import ToolCall, ToolResult, create_tool_call


class RuntimeStreamCloseReason(str, Enum):
    NORMAL = "normal"
    FINISHED = "finished"
    ABORTED = "aborted"
    ERROR = "error"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_stream_id() -> str:
    return f"stream_{uuid4().hex}"


@dataclass(frozen=True)
class RuntimeStreamClosed:
    reason: RuntimeStreamCloseReason = RuntimeStreamCloseReason.NORMAL
    message: str = ""
    closed_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason.value,
            "message": self.message,
            "closed_at": self.closed_at.isoformat(),
        }


@dataclass
class RuntimeStreamConfig:
    """
    Runtime 事件流配置。
    """

    run_id: str | None = None
    session_id: str | None = None
    event_types: set[RuntimeEventType] | None = None

    include_history: bool = False
    auto_close_on_terminal: bool = True
    close_on_error_event: bool = True

    queue_max_size: int = 1000


def normalize_event_types(
    event_types: Iterable[RuntimeEventType | str] | None,
) -> set[RuntimeEventType] | None:
    if event_types is None:
        return None

    return {
        item if isinstance(item, RuntimeEventType) else RuntimeEventType(item)
        for item in event_types
    }


def is_terminal_event(event: RuntimeEvent) -> tuple[bool, RuntimeStreamCloseReason, str]:
    """
    判断一个事件是否代表流应该结束。
    """
    if event.event_type == RuntimeEventType.LIFECYCLE:
        lifecycle = event.lifecycle

        if isinstance(lifecycle, RuntimeLifecycleEvent):
            lifecycle_value = lifecycle.value
        else:
            lifecycle_value = str(lifecycle)

        if lifecycle_value == RuntimeLifecycleEvent.FINISHED.value:
            return True, RuntimeStreamCloseReason.FINISHED, event.content

        if lifecycle_value == RuntimeLifecycleEvent.ABORTED.value:
            return True, RuntimeStreamCloseReason.ABORTED, event.content

        if lifecycle_value == RuntimeLifecycleEvent.ERROR.value:
            return True, RuntimeStreamCloseReason.ERROR, event.content

    return False, RuntimeStreamCloseReason.NORMAL, ""


class RuntimeEventStream:
    """
    RuntimeEventBus → Async Iterator。

    用法：

        stream = RuntimeEventStream(bus)
        async for event in stream:
            ...

    它会订阅 RuntimeEventBus，把事件放进 asyncio.Queue，
    然后通过 async for 逐个吐出。
    """

    def __init__(
        self,
        bus: RuntimeEventBus | None = None,
        *,
        config: RuntimeStreamConfig | None = None,
    ) -> None:
        self.bus = bus or get_default_event_bus()
        self.config = config or RuntimeStreamConfig()

        self.stream_id = new_stream_id()

        self._queue: asyncio.Queue[RuntimeEvent | RuntimeStreamClosed] = asyncio.Queue(
            maxsize=max(1, self.config.queue_max_size)
        )

        self._unsubscribe: Any = None
        self._started = False
        self._closed = False
        self._close_reason: RuntimeStreamClosed | None = None

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def is_closed(self) -> bool:
        return self._closed

    def matches(self, event: RuntimeEvent) -> bool:
        if self.config.run_id is not None and event.run_id != self.config.run_id:
            return False

        if self.config.session_id is not None and event.session_id != self.config.session_id:
            return False

        if self.config.event_types is not None and event.event_type not in self.config.event_types:
            return False

        return True

    def start(self) -> None:
        if self._started:
            return

        self._started = True

        if self.config.include_history:
            for event in self.bus.history():
                if self.matches(event):
                    self._put_nowait(event)

        self._unsubscribe = self.bus.subscribe(self._on_event)

    def close(
        self,
        *,
        reason: RuntimeStreamCloseReason = RuntimeStreamCloseReason.NORMAL,
        message: str = "",
    ) -> None:
        if self._closed:
            return

        self._closed = True

        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

        closed = RuntimeStreamClosed(
            reason=reason,
            message=message,
        )

        self._close_reason = closed
        self._put_nowait(closed)

    async def aclose(
        self,
        *,
        reason: RuntimeStreamCloseReason = RuntimeStreamCloseReason.NORMAL,
        message: str = "",
    ) -> None:
        self.close(reason=reason, message=message)

    def _put_nowait(self, item: RuntimeEvent | RuntimeStreamClosed) -> None:
        """
        放入队列。

        如果队列满了，丢弃最旧事件，保证最新事件能进入。
        """
        try:
            self._queue.put_nowait(item)
            return
        except asyncio.QueueFull:
            pass

        try:
            self._queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            pass

    def _on_event(self, event: RuntimeEvent) -> None:
        if self._closed:
            return

        if not self.matches(event):
            return

        self._put_nowait(event)

        if self.config.close_on_error_event and event.event_type == RuntimeEventType.ERROR:
            self.close(
                reason=RuntimeStreamCloseReason.ERROR,
                message=event.error or event.content,
            )
            return

        if self.config.auto_close_on_terminal:
            should_close, reason, message = is_terminal_event(event)

            if should_close:
                self.close(
                    reason=reason,
                    message=message,
                )

    def __aiter__(self) -> RuntimeEventStream:
        self.start()
        return self

    async def __anext__(self) -> RuntimeEvent:
        self.start()

        item = await self._queue.get()

        if isinstance(item, RuntimeStreamClosed):
            raise StopAsyncIteration

        return item

    async def events(self) -> AsyncGenerator[RuntimeEvent, None]:
        """
        显式 AsyncGenerator 入口。
        """
        async for event in self:
            yield event

    async def collect(self) -> list[RuntimeEvent]:
        """
        收集直到流结束。
        """
        collected: list[RuntimeEvent] = []

        async for event in self:
            collected.append(event)

        return collected

    async def __aenter__(self) -> RuntimeEventStream:
        self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose(
            reason=RuntimeStreamCloseReason.ERROR if exc else RuntimeStreamCloseReason.NORMAL,
            message=str(exc) if exc else "",
        )


class RuntimeStreamer:
    """
    Runtime 事件推送器。

    它是 RuntimeEngine / RuntimeController 后面最常用的辅助对象。

    用法：
        streamer = RuntimeStreamer()
        await streamer.emit_message_delta("hello")
        await streamer.emit_tool_call(call)
    """

    def __init__(
        self,
        *,
        bus: RuntimeEventBus | None = None,
        run_id: str | None = None,
        session_id: str | None = None,
        checkpoint_id: str | None = None,
        source: RuntimeEventSource | str = RuntimeEventSource.ENGINE,
    ) -> None:
        self.bus = bus or get_default_event_bus()
        self.run_id = run_id or new_run_id()
        self.session_id = session_id
        self.checkpoint_id = checkpoint_id or new_checkpoint_id()
        self.source = RuntimeEventSource(source)

    def create_stream(
        self,
        *,
        event_types: Iterable[RuntimeEventType | str] | None = None,
        include_history: bool = False,
        auto_close_on_terminal: bool = True,
    ) -> RuntimeEventStream:
        return RuntimeEventStream(
            self.bus,
            config=RuntimeStreamConfig(
                run_id=self.run_id,
                session_id=self.session_id,
                event_types=normalize_event_types(event_types),
                include_history=include_history,
                auto_close_on_terminal=auto_close_on_terminal,
            ),
        )

    async def emit(self, event: RuntimeEvent) -> RuntimeEvent:
        return await self.bus.emit_async(event)

    async def emit_lifecycle(
        self,
        lifecycle: RuntimeLifecycleEvent | str,
        *,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.lifecycle_event(
                lifecycle=lifecycle,
                content=content,
                source=self.source,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_status(
        self,
        status: str,
        *,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.status_event(
                status=status,
                content=content,
                source=self.source,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_message(
        self,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.message(
                role=role,  # type: ignore[arg-type]
                content=content,
                source=self.source,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_message_delta(
        self,
        delta: str,
        *,
        role: str = "assistant",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.message_delta(
                delta=delta,
                role=role,  # type: ignore[arg-type]
                source=RuntimeEventSource.LLM,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_tool_call(
        self,
        tool_call: ToolCall,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.tool_call_event(
                tool_call=tool_call,
                source=RuntimeEventSource.GRAPH,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_tool_result(
        self,
        tool_result: ToolResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.tool_result_event(
                tool_result=tool_result,
                source=RuntimeEventSource.TOOL,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_error(
        self,
        error: str,
        *,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        return await self.emit(
            RuntimeEvent.error_event(
                error=error,
                error_type=error_type,
                source=self.source,
                run_id=self.run_id,
                session_id=self.session_id,
                checkpoint_id=self.checkpoint_id,
                metadata=metadata,
            )
        )

    async def emit_checkpoint(
        self,
        *,
        checkpoint_id: str | None = None,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        if checkpoint_id is not None:
            self.checkpoint_id = checkpoint_id

        return await self.emit(
            RuntimeEvent.checkpoint_event(
                checkpoint_id=self.checkpoint_id,
                source=self.source,
                run_id=self.run_id,
                session_id=self.session_id,
                data=data or {},
                metadata=metadata,
            )
        )


async def stream_events(
    bus: RuntimeEventBus | None = None,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    event_types: Iterable[RuntimeEventType | str] | None = None,
    include_history: bool = False,
    auto_close_on_terminal: bool = True,
) -> AsyncGenerator[RuntimeEvent, None]:
    """
    函数式 AsyncGenerator 入口。

    用法：
        async for event in stream_events(run_id=...):
            ...
    """
    stream = RuntimeEventStream(
        bus or get_default_event_bus(),
        config=RuntimeStreamConfig(
            run_id=run_id,
            session_id=session_id,
            event_types=normalize_event_types(event_types),
            include_history=include_history,
            auto_close_on_terminal=auto_close_on_terminal,
        ),
    )

    async with stream:
        async for event in stream.events():
            yield event


async def demo_producer(streamer: RuntimeStreamer) -> None:
    await streamer.emit_lifecycle(
        RuntimeLifecycleEvent.STARTED,
        content="stream started",
    )

    await streamer.emit_status(
        "thinking",
        content="calling model",
    )

    for chunk in ["你好", "，", "这里是 ", "PyWork", " 的流式输出。"]:
        await asyncio.sleep(0.05)
        await streamer.emit_message_delta(chunk)

    await streamer.emit_message(
        "assistant",
        "你好，这里是 PyWork 的流式输出。",
    )

    call = create_tool_call(
        tool_name="echo",
        arguments={
            "text": "hello streaming",
        },
    )

    await streamer.emit_tool_call(call)

    await asyncio.sleep(0.05)

    result = ToolResult.success_result(
        call=call,
        content="hello streaming",
        data={
            "text": "hello streaming",
        },
    )

    await streamer.emit_tool_result(result)

    await streamer.emit_checkpoint(
        data={
            "iteration": 1,
        }
    )

    await streamer.emit_status(
        "idle",
        content="done",
    )

    await streamer.emit_lifecycle(
        RuntimeLifecycleEvent.FINISHED,
        content="stream finished",
    )


async def demo_consumer(stream: RuntimeEventStream) -> list[RuntimeEvent]:
    events: list[RuntimeEvent] = []

    async for event in stream:
        events.append(event)
        print(event.compact_text())

    return events


async def demo() -> None:
    bus = RuntimeEventBus()
    streamer = RuntimeStreamer(
        bus=bus,
        source=RuntimeEventSource.ENGINE,
    )

    stream = streamer.create_stream()

    consumer_task = asyncio.create_task(
        demo_consumer(stream)
    )

    await demo_producer(streamer)

    events = await consumer_task

    print("\nStream collected:")
    print(len(events))

    print("\nLast event:")
    if events:
        print(events[-1].to_json(indent=2))

    print("\nBus history:")
    print(json.dumps(bus.to_dict(), ensure_ascii=False, indent=2, default=str))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())