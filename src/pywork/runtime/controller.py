from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pywork.runtime.engine import (
    RuntimeEngine,
    RuntimeRunResult,
    RuntimeStatus,
)
from pywork.state.app_state import AppState, create_app_state
from pywork.state.session_state import SessionStatus
from pywork.tools.registry import ToolRegistry


class RuntimeControllerStatus(str, Enum):
    """
    Controller 调度状态。
    """

    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FINISHED = "finished"
    ERROR = "error"


class RuntimeControllerEventType(str, Enum):
    """
    Controller 事件类型。
    """

    INPUT_RECEIVED = "input_received"
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    PAUSED = "paused"
    RESUMED = "resumed"
    ABORT_REQUESTED = "abort_requested"
    ABORTED = "aborted"
    ERROR = "error"
    STATE_SYNCED = "state_synced"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_controller_event_id() -> str:
    return f"controller_event_{uuid4().hex}"


def estimate_tokens(text: str) -> int:
    text = text.strip()

    if not text:
        return 0

    return max(1, len(text) // 2)


@dataclass(frozen=True)
class RuntimeControllerEvent:
    event_type: RuntimeControllerEventType
    status: RuntimeControllerStatus
    message: str = ""
    event_id: str = field(default_factory=new_controller_event_id)
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "status": self.status.value,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RuntimeControllerRunResult:
    """
    Controller 一次用户输入调度结果。
    """

    success: bool
    status: RuntimeControllerStatus
    runtime_result: RuntimeRunResult | None = None

    input_text: str = ""
    output: str = ""
    error: str | None = None

    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    duration_ms: int = 0

    events: list[RuntimeControllerEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "input_text": self.input_text,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_ms": self.duration_ms,
            "runtime_result": (
                self.runtime_result.to_dict()
                if self.runtime_result is not None
                else None
            ),
            "events": [
                event.to_dict()
                for event in self.events
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


@dataclass
class RuntimeControllerConfig:
    """
    Controller 配置。
    """

    add_user_message_to_session: bool = True
    add_assistant_message_to_session: bool = True
    sync_tool_calls_to_session: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeControllerError(Exception):
    pass


class RuntimeControllerAlreadyRunningError(RuntimeControllerError):
    pass


class RuntimeController:
    """
    PyWork Runtime Controller。

    职责：
    1. 接收用户输入
    2. 更新 AppState / UIState
    3. 调用 RuntimeEngine
    4. 把 RuntimeResult 同步回 SessionState
    5. 管理 pause / resume / abort
    """

    def __init__(
        self,
        *,
        app_state: AppState | None = None,
        engine: RuntimeEngine | None = None,
        registry: ToolRegistry | None = None,
        config: dict[str, Any] | None = None,
        controller_config: RuntimeControllerConfig | None = None,
    ) -> None:
        self.app_state = app_state or create_app_state(
            config=config or {},
        )

        self.controller_config = controller_config or RuntimeControllerConfig()

        self.engine = engine or RuntimeEngine(
            registry=registry or self.app_state.tool_registry,
            config=config or self.app_state.config,
            agent_state=None,
        )

        self.status: RuntimeControllerStatus = RuntimeControllerStatus.READY
        self._events: list[RuntimeControllerEvent] = []

        self._current_task: asyncio.Task[RuntimeControllerRunResult] | None = None

        self._synced_tool_call_ids: set[str] = set()
        self._synced_tool_result_ids: set[str] = set()

    @property
    def is_running(self) -> bool:
        return self.status == RuntimeControllerStatus.RUNNING

    @property
    def is_paused(self) -> bool:
        return self.status == RuntimeControllerStatus.PAUSED

    def get_events(self) -> list[RuntimeControllerEvent]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()

    def get_app_state(self) -> AppState:
        return self.app_state

    def get_engine(self) -> RuntimeEngine:
        return self.engine

    def _emit(
        self,
        event_type: RuntimeControllerEventType,
        *,
        status: RuntimeControllerStatus | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeControllerEvent:
        event = RuntimeControllerEvent(
            event_type=event_type,
            status=status or self.status,
            message=message,
            metadata=metadata or {},
        )

        self._events.append(event)
        return event

    def _set_status(
        self,
        status: RuntimeControllerStatus,
        *,
        message: str = "",
        event_type: RuntimeControllerEventType | None = None,
    ) -> None:
        self.status = status

        if event_type is not None:
            self._emit(
                event_type,
                status=status,
                message=message,
            )

    def _make_result(
        self,
        *,
        success: bool,
        status: RuntimeControllerStatus,
        started_at: datetime,
        finished_at: datetime,
        input_text: str,
        runtime_result: RuntimeRunResult | None = None,
        output: str = "",
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeControllerRunResult:
        duration_ms = max(
            0,
            int((finished_at - started_at).total_seconds() * 1000),
        )

        return RuntimeControllerRunResult(
            success=success,
            status=status,
            runtime_result=runtime_result,
            input_text=input_text,
            output=output,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            events=self.get_events(),
            metadata=metadata or {},
        )

    def _prepare_user_input(self, user_input: str) -> str:
        return user_input.strip()

    def _record_user_input(self, user_input: str) -> None:
        if not self.controller_config.add_user_message_to_session:
            return

        token_estimate = estimate_tokens(user_input)

        self.app_state.add_user_message(
            user_input,
            token_estimate=token_estimate,
            metadata={
                "source": "runtime_controller",
            },
        )

    def _sync_tool_events_from_runtime_result(
        self,
        result: RuntimeRunResult,
    ) -> None:
        if not self.controller_config.sync_tool_calls_to_session:
            return

        agent_state = result.agent_state

        for call in agent_state.tool_calls:
            if call.call_id in self._synced_tool_call_ids:
                continue

            self.app_state.add_tool_call(call)
            self._synced_tool_call_ids.add(call.call_id)

        for tool_result in agent_state.tool_results:
            if tool_result.result_id in self._synced_tool_result_ids:
                continue

            self.app_state.add_tool_result(tool_result)
            self._synced_tool_result_ids.add(tool_result.result_id)

    def _sync_runtime_result_to_app_state(
        self,
        result: RuntimeRunResult,
    ) -> None:
        """
        把 RuntimeResult 同步回 AppState。

        注意：
        AgentState 内部也有 messages。
        SessionState 是给 UI / 历史记录使用的状态。
        所以这里会把最终 output 作为 assistant message 写入 session。
        """
        self._sync_tool_events_from_runtime_result(result)

        if result.success:
            if (
                result.output
                and self.controller_config.add_assistant_message_to_session
            ):
                self.app_state.add_assistant_message(
                    result.output,
                    token_estimate=estimate_tokens(result.output),
                    metadata={
                        "source": "runtime_controller",
                        "runtime_status": result.status.value,
                    },
                )

            self.app_state.session.set_status(SessionStatus.IDLE)
            self.app_state.ui.set_idle("runtime finished")

        elif result.aborted:
            self.app_state.session.set_status(SessionStatus.IDLE)
            self.app_state.ui.set_idle("runtime aborted")

            self.app_state.add_system_message(
                "Runtime aborted.",
                metadata={
                    "source": "runtime_controller",
                },
            )

        else:
            self.app_state.add_error_message(
                result.error or "Runtime failed.",
                metadata={
                    "source": "runtime_controller",
                    "runtime_status": result.status.value,
                },
            )

        self._emit(
            RuntimeControllerEventType.STATE_SYNCED,
            status=self.status,
            message="runtime result synced to app state",
            metadata={
                "runtime_status": result.status.value,
                "success": result.success,
            },
        )

    async def arun(
        self,
        user_input: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeControllerRunResult:
        """
        异步调度一次用户输入。

        TUI 后面应该调用这个方法。
        """
        started_at = utc_now()
        text = self._prepare_user_input(user_input)

        if self.status == RuntimeControllerStatus.RUNNING:
            raise RuntimeControllerAlreadyRunningError(
                "RuntimeController is already running"
            )

        if not text:
            self.app_state.ui.set_idle("empty input ignored")

            return self._make_result(
                success=True,
                status=RuntimeControllerStatus.READY,
                started_at=started_at,
                finished_at=utc_now(),
                input_text=user_input,
                output="",
                metadata=metadata,
            )

        self._emit(
            RuntimeControllerEventType.INPUT_RECEIVED,
            status=self.status,
            message="input received",
            metadata={
                "length": len(text),
            },
        )

        self._set_status(
            RuntimeControllerStatus.RUNNING,
            message="controller started",
            event_type=RuntimeControllerEventType.RUN_STARTED,
        )

        self.app_state.session.set_status(SessionStatus.THINKING)
        self.app_state.ui.set_thinking("runtime running")

        self._record_user_input(text)

        try:
            runtime_result = await self.engine.arun(
                text,
                metadata={
                    **(metadata or {}),
                    "source": "runtime_controller",
                },
            )

            self._sync_runtime_result_to_app_state(runtime_result)

            if runtime_result.success:
                self._set_status(
                    RuntimeControllerStatus.FINISHED,
                    message="controller finished",
                    event_type=RuntimeControllerEventType.RUN_FINISHED,
                )

                return self._make_result(
                    success=True,
                    status=RuntimeControllerStatus.FINISHED,
                    runtime_result=runtime_result,
                    started_at=started_at,
                    finished_at=utc_now(),
                    input_text=text,
                    output=runtime_result.output,
                    metadata=metadata,
                )

            if runtime_result.aborted:
                self._set_status(
                    RuntimeControllerStatus.ABORTED,
                    message="controller aborted",
                    event_type=RuntimeControllerEventType.ABORTED,
                )

                return self._make_result(
                    success=False,
                    status=RuntimeControllerStatus.ABORTED,
                    runtime_result=runtime_result,
                    started_at=started_at,
                    finished_at=utc_now(),
                    input_text=text,
                    output=runtime_result.output,
                    error=runtime_result.error,
                    metadata=metadata,
                )

            self._set_status(
                RuntimeControllerStatus.ERROR,
                message=runtime_result.error or "runtime error",
                event_type=RuntimeControllerEventType.ERROR,
            )

            return self._make_result(
                success=False,
                status=RuntimeControllerStatus.ERROR,
                runtime_result=runtime_result,
                started_at=started_at,
                finished_at=utc_now(),
                input_text=text,
                output=runtime_result.output,
                error=runtime_result.error,
                metadata=metadata,
            )

        except Exception as exc:
            self._set_status(
                RuntimeControllerStatus.ERROR,
                message=str(exc),
                event_type=RuntimeControllerEventType.ERROR,
            )

            self.app_state.add_error_message(
                str(exc),
                metadata={
                    "source": "runtime_controller",
                    "exception_type": type(exc).__name__,
                },
            )

            return self._make_result(
                success=False,
                status=RuntimeControllerStatus.ERROR,
                started_at=started_at,
                finished_at=utc_now(),
                input_text=text,
                output="",
                error=str(exc),
                metadata={
                    **(metadata or {}),
                    "exception_type": type(exc).__name__,
                },
            )

    def run(
        self,
        user_input: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeControllerRunResult:
        """
        同步调度一次用户输入。

        CLI / demo 可用。
        如果已经在异步环境里，请用 await arun()。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(
                    user_input,
                    metadata=metadata,
                )
            )

        raise RuntimeControllerError(
            "RuntimeController.run() cannot be used inside a running event loop. "
            "Use await RuntimeController.arun(...) instead."
        )

    def start_background(
        self,
        user_input: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> asyncio.Task[RuntimeControllerRunResult]:
        """
        在当前事件循环中后台启动一次调度。

        Textual 中如果不想阻塞 UI，可以用这个。
        """
        if self._current_task is not None and not self._current_task.done():
            raise RuntimeControllerAlreadyRunningError(
                "RuntimeController already has a running task"
            )

        self._current_task = asyncio.create_task(
            self.arun(
                user_input,
                metadata=metadata,
            )
        )

        return self._current_task

    def get_current_task(self) -> asyncio.Task[RuntimeControllerRunResult] | None:
        return self._current_task

    def pause(self) -> None:
        self.engine.pause()

        self._set_status(
            RuntimeControllerStatus.PAUSED,
            message="controller paused",
            event_type=RuntimeControllerEventType.PAUSED,
        )

        self.app_state.ui.set_idle("runtime paused")

    def resume(self) -> None:
        self.engine.resume()

        self._set_status(
            RuntimeControllerStatus.READY,
            message="controller resumed",
            event_type=RuntimeControllerEventType.RESUMED,
        )

        self.app_state.ui.set_idle("runtime resumed")

    def abort(self) -> None:
        self.engine.abort()

        self._set_status(
            RuntimeControllerStatus.ABORTING,
            message="controller abort requested",
            event_type=RuntimeControllerEventType.ABORT_REQUESTED,
        )

        self.app_state.ui.set_idle("abort requested")

    def reset(
        self,
        *,
        keep_messages: bool = True,
    ) -> None:
        self.engine.reset(
            keep_messages=keep_messages,
        )

        self.status = RuntimeControllerStatus.READY
        self.app_state.ui.set_idle("controller reset")

        if not keep_messages:
            self.app_state.start_new_session(
                title="PyWork Session",
                metadata={
                    "source": "runtime_controller.reset",
                },
            )

        self._synced_tool_call_ids.clear()
        self._synced_tool_result_ids.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "engine": self.engine.to_dict(),
            "app_state": self.app_state.get_status_summary(),
            "current_task_running": (
                self._current_task is not None
                and not self._current_task.done()
            ),
            "synced_tool_call_count": len(self._synced_tool_call_ids),
            "synced_tool_result_count": len(self._synced_tool_result_ids),
            "events": [
                event.to_dict()
                for event in self._events
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


async def demo() -> None:
    app_state = create_app_state(
        config={
            "permissions": {
                "mode": "default",
            },
            "agent": {
                "max_iterations": 5,
                "max_context_messages": 20,
            },
        }
    )

    controller = RuntimeController(
        app_state=app_state,
    )

    print("Run normal input:")
    result = await controller.arun("hello PyWork")

    print(result.to_json(indent=2))
    print("\nController state:")
    print(controller.to_json(indent=2))

    print("\nRun tool input:")
    result2 = await controller.arun("/tool echo Hello from RuntimeController.")

    print(result2.to_json(indent=2))
    print("\nAppState summary:")
    print(json.dumps(app_state.get_status_summary(), ensure_ascii=False, indent=2))

    print("\nPause / resume:")
    controller.pause()
    print(controller.to_json(indent=2))

    controller.resume()
    print(controller.to_json(indent=2))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())