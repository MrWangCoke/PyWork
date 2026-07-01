from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from threading import RLock
from typing import Any
from uuid import uuid4

from pywork.permission.session_overrides import PermissionGateState
from pywork.runtime.events import RuntimeEventBus, get_default_event_bus
from pywork.runtime.graph import AgentGraphRunner
from pywork.runtime.state import AgentState, AgentStatus, create_agent_state
from pywork.tools.registry import ToolRegistry, create_default_registry


class RuntimeStatus(str, Enum):
    """
    RuntimeEngine 生命周期状态。
    """

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    FINISHED = "finished"
    ABORTING = "aborting"
    ABORTED = "aborted"
    ERROR = "error"


class RuntimeEventType(str, Enum):
    """
    Runtime 事件类型。
    """

    STARTED = "started"
    PAUSED = "paused"
    RESUMED = "resumed"
    FINISHED = "finished"
    ABORT_REQUESTED = "abort_requested"
    ABORTED = "aborted"
    ERROR = "error"
    STATUS_CHANGED = "status_changed"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_runtime_event_id() -> str:
    return f"runtime_event_{uuid4().hex}"


@dataclass(frozen=True)
class RuntimeEvent:
    """
    Runtime 运行事件。

    后面 controller / TUI 可以用这些事件更新界面。
    """

    event_type: RuntimeEventType
    status: RuntimeStatus
    message: str = ""
    event_id: str = field(default_factory=new_runtime_event_id)
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
class RuntimeRunResult:
    """
    一次 Runtime 执行结果。
    """

    success: bool
    status: RuntimeStatus
    agent_state: AgentState

    output: str = ""
    error: str | None = None

    started_at: datetime = field(default_factory=utc_now)
    finished_at: datetime = field(default_factory=utc_now)
    duration_ms: int = 0

    aborted: bool = False
    events: list[RuntimeEvent] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_ms": self.duration_ms,
            "aborted": self.aborted,
            "agent_state": self.agent_state.summary(),
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
class RuntimeEngineConfig:
    """
    RuntimeEngine 配置。

    max_iterations 会传给 AgentState。
    """

    max_iterations: int = 20
    system_prompt: str = "You are PyWork, a coding agent."
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeEngineError(Exception):
    """
    RuntimeEngine 基础异常。
    """

    pass


class RuntimeAlreadyRunningError(RuntimeEngineError):
    """
    Runtime 正在运行，不能重复启动。
    """

    pass


class RuntimeEngine:
    """
    PyWork Runtime Engine。

    职责：
    - 管理 Agent 生命周期
    - 调用 LangGraph 执行图
    - 维护 AgentState
    - 暴露 run / pause / resume / abort 控制方法

    注意：
    当前 LangGraph 是一次性 ainvoke 执行。
    pause() 主要用于阻止下一次 run 开始；
    如果图已经进入某个非流式节点，pause 不会强行冻结该节点。
    abort() 会尝试 cancel 当前 asyncio task。
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        config: dict[str, Any] | None = None,
        agent_state: AgentState | None = None,
        engine_config: RuntimeEngineConfig | None = None,
        event_bus: RuntimeEventBus | None = None,
        emit_events: bool = True,
        approval_handler: Any | None = None,
        permission_gate_state: PermissionGateState | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.config = config or {}
        self.engine_config = engine_config or RuntimeEngineConfig()
        self.event_bus = event_bus or get_default_event_bus()
        self.emit_events = emit_events
        self.approval_handler = approval_handler
        self.permission_gate_state = permission_gate_state or PermissionGateState()

        self.agent_state = agent_state or create_agent_state(
            system_prompt=None,
            max_iterations=int(
                self.config.get("agent", {}).get(
                    "max_iterations",
                    self.engine_config.max_iterations,
                )
            ),
            metadata=self.engine_config.metadata,
        )

        self.graph_runner = AgentGraphRunner(
            registry=self.registry,
            config=self.config,
            event_bus=self.event_bus,
            emit_events=self.emit_events,
            approval_handler=self.approval_handler,
            permission_gate_state=self.permission_gate_state,
        )

        self.status: RuntimeStatus = RuntimeStatus.IDLE

        self._events: list[RuntimeEvent] = []
        self._lock = RLock()

        self._paused = False
        self._abort_requested = False
        self._current_task: asyncio.Task[AgentState] | None = None

        self._last_result: RuntimeRunResult | None = None

    def get_status(self) -> RuntimeStatus:
        return self.status

    def get_agent_state(self) -> AgentState:
        return self.agent_state

    def get_last_result(self) -> RuntimeRunResult | None:
        return self._last_result

    def get_events(self) -> list[RuntimeEvent]:
        return list(self._events)

    def clear_events(self) -> None:
        self._events.clear()

    @property
    def is_running(self) -> bool:
        return self.status == RuntimeStatus.RUNNING

    @property
    def is_paused(self) -> bool:
        return self.status == RuntimeStatus.PAUSED

    @property
    def is_aborting(self) -> bool:
        return self.status == RuntimeStatus.ABORTING

    @property
    def is_terminal(self) -> bool:
        return self.status in {
            RuntimeStatus.FINISHED,
            RuntimeStatus.ABORTED,
            RuntimeStatus.ERROR,
        }

    def _emit(
        self,
        event_type: RuntimeEventType,
        *,
        status: RuntimeStatus | None = None,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeEvent:
        event = RuntimeEvent(
            event_type=event_type,
            status=status or self.status,
            message=message,
            metadata=metadata or {},
        )

        self._events.append(event)
        return event

    def _set_status(
        self,
        status: RuntimeStatus,
        *,
        message: str = "",
        event_type: RuntimeEventType = RuntimeEventType.STATUS_CHANGED,
    ) -> None:
        self.status = status
        self._emit(
            event_type,
            status=status,
            message=message,
        )

    async def _wait_if_paused(self) -> None:
        """
        如果 Runtime 被暂停，则等待 resume 或 abort。

        这个等待发生在一次 graph 执行开始前。
        """
        while self._paused and not self._abort_requested:
            self._set_status(
                RuntimeStatus.PAUSED,
                message="runtime paused",
                event_type=RuntimeEventType.PAUSED,
            )
            await asyncio.sleep(0.05)

    def pause(self) -> None:
        """
        暂停 Runtime。

        当前版本：
        - 如果还没开始运行，会阻止下一次 arun 真正进入 graph。
        - 如果已经在 graph 节点中执行，不强行打断当前节点。
        """
        with self._lock:
            self._paused = True

            if self.status not in {
                RuntimeStatus.ABORTED,
                RuntimeStatus.ERROR,
                RuntimeStatus.FINISHED,
            }:
                self._set_status(
                    RuntimeStatus.PAUSED,
                    message="pause requested",
                    event_type=RuntimeEventType.PAUSED,
                )

    def resume(self) -> None:
        """
        恢复 Runtime。
        """
        with self._lock:
            self._paused = False

            if self.status == RuntimeStatus.PAUSED:
                self._set_status(
                    RuntimeStatus.IDLE,
                    message="resume requested",
                    event_type=RuntimeEventType.RESUMED,
                )
            else:
                self._emit(
                    RuntimeEventType.RESUMED,
                    message="resume requested",
                )

    def abort(self) -> None:
        """
        中止 Runtime。

        如果当前有正在执行的 asyncio task，会尝试 cancel。
        """
        with self._lock:
            self._abort_requested = True
            self._paused = False

            self._set_status(
                RuntimeStatus.ABORTING,
                message="abort requested",
                event_type=RuntimeEventType.ABORT_REQUESTED,
            )

            if self._current_task is not None and not self._current_task.done():
                self._current_task.cancel()

    def reset(
        self,
        *,
        keep_messages: bool = True,
    ) -> None:
        """
        重置 Runtime 状态。

        keep_messages=True：
            保留 AgentState.messages，只重置运行状态。

        keep_messages=False：
            创建全新的 AgentState。
        """
        with self._lock:
            self._paused = False
            self._abort_requested = False
            self._current_task = None

            if keep_messages:
                self.agent_state.reset_runtime()
            else:
                self.agent_state = create_agent_state(
                    system_prompt=None,
                    max_iterations=int(
                        self.config.get("agent", {}).get(
                            "max_iterations",
                            self.engine_config.max_iterations,
                        )
                    ),
                    metadata=self.engine_config.metadata,
                )

            self._set_status(
                RuntimeStatus.IDLE,
                message="runtime reset",
            )

    def _extract_output(self, state: AgentState) -> str:
        last_message = state.get_last_message()

        if last_message is None:
            return ""

        return last_message.content

    def _make_result(
        self,
        *,
        success: bool,
        status: RuntimeStatus,
        started_at: datetime,
        finished_at: datetime,
        agent_state: AgentState,
        output: str = "",
        error: str | None = None,
        aborted: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeRunResult:
        duration_ms = max(
            0,
            int((finished_at - started_at).total_seconds() * 1000),
        )

        result = RuntimeRunResult(
            success=success,
            status=status,
            agent_state=agent_state,
            output=output,
            error=error,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            aborted=aborted,
            events=self.get_events(),
            metadata=metadata or {},
        )

        self._last_result = result
        return result

    async def arun(
        self,
        user_input: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeRunResult:
        """
        异步运行 Agent。

        TUI / Controller 后面应该优先调用这个方法。
        """
        started_at = utc_now()

        if self.status == RuntimeStatus.RUNNING:
            raise RuntimeAlreadyRunningError("Runtime is already running")

        await self._wait_if_paused()

        if self._abort_requested:
            self.agent_state.set_cancelled("aborted before run")
            self._set_status(
                RuntimeStatus.ABORTED,
                message="aborted before run",
                event_type=RuntimeEventType.ABORTED,
            )

            return self._make_result(
                success=False,
                status=RuntimeStatus.ABORTED,
                started_at=started_at,
                finished_at=utc_now(),
                agent_state=self.agent_state,
                output="",
                error="aborted before run",
                aborted=True,
                metadata=metadata,
            )

        self._abort_requested = False
        self._set_status(
            RuntimeStatus.RUNNING,
            message="runtime started",
            event_type=RuntimeEventType.STARTED,
        )

        try:
            task = asyncio.create_task(
                self.graph_runner.arun(
                    user_input,
                    agent_state=self.agent_state,
                    metadata=metadata or {},
                )
            )

            self._current_task = task

            while not task.done():
                if self._abort_requested:
                    task.cancel()
                    break

                await asyncio.sleep(0.05)

            state = await task

            self.agent_state = state

            if self._abort_requested:
                self.agent_state.set_cancelled("aborted")
                self._set_status(
                    RuntimeStatus.ABORTED,
                    message="runtime aborted",
                    event_type=RuntimeEventType.ABORTED,
                )

                return self._make_result(
                    success=False,
                    status=RuntimeStatus.ABORTED,
                    started_at=started_at,
                    finished_at=utc_now(),
                    agent_state=self.agent_state,
                    output=self._extract_output(self.agent_state),
                    error="aborted",
                    aborted=True,
                    metadata=metadata,
                )

            if self.agent_state.status == AgentStatus.ERROR:
                self._set_status(
                    RuntimeStatus.ERROR,
                    message=self.agent_state.last_error or "agent error",
                    event_type=RuntimeEventType.ERROR,
                )

                return self._make_result(
                    success=False,
                    status=RuntimeStatus.ERROR,
                    started_at=started_at,
                    finished_at=utc_now(),
                    agent_state=self.agent_state,
                    output=self._extract_output(self.agent_state),
                    error=self.agent_state.last_error or "agent error",
                    metadata=metadata,
                )

            if self.agent_state.status == AgentStatus.CANCELLED:
                self._set_status(
                    RuntimeStatus.ABORTED,
                    message=self.agent_state.last_error or "agent cancelled",
                    event_type=RuntimeEventType.ABORTED,
                )

                return self._make_result(
                    success=False,
                    status=RuntimeStatus.ABORTED,
                    started_at=started_at,
                    finished_at=utc_now(),
                    agent_state=self.agent_state,
                    output=self._extract_output(self.agent_state),
                    error=self.agent_state.last_error or "agent cancelled",
                    aborted=True,
                    metadata=metadata,
                )

            self._set_status(
                RuntimeStatus.FINISHED,
                message="runtime finished",
                event_type=RuntimeEventType.FINISHED,
            )

            return self._make_result(
                success=True,
                status=RuntimeStatus.FINISHED,
                started_at=started_at,
                finished_at=utc_now(),
                agent_state=self.agent_state,
                output=self._extract_output(self.agent_state),
                metadata=metadata,
            )

        except asyncio.CancelledError:
            self.agent_state.set_cancelled("runtime task cancelled")
            self._set_status(
                RuntimeStatus.ABORTED,
                message="runtime task cancelled",
                event_type=RuntimeEventType.ABORTED,
            )

            return self._make_result(
                success=False,
                status=RuntimeStatus.ABORTED,
                started_at=started_at,
                finished_at=utc_now(),
                agent_state=self.agent_state,
                output=self._extract_output(self.agent_state),
                error="runtime task cancelled",
                aborted=True,
                metadata=metadata,
            )

        except Exception as exc:
            self.agent_state.set_error(str(exc))
            self._set_status(
                RuntimeStatus.ERROR,
                message=str(exc),
                event_type=RuntimeEventType.ERROR,
            )

            return self._make_result(
                success=False,
                status=RuntimeStatus.ERROR,
                started_at=started_at,
                finished_at=utc_now(),
                agent_state=self.agent_state,
                output=self._extract_output(self.agent_state),
                error=str(exc),
                metadata={
                    **(metadata or {}),
                    "exception_type": type(exc).__name__,
                },
            )

        finally:
            self._current_task = None
            self._abort_requested = False

    def run(
        self,
        user_input: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RuntimeRunResult:
        """
        同步运行 Agent。

        CLI / demo 可以用这个。
        如果已经在异步环境里，比如 Textual App，请用 await arun()。
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

        raise RuntimeEngineError(
            "RuntimeEngine.run() cannot be used inside a running event loop. "
            "Use await RuntimeEngine.arun(...) instead."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "paused": self._paused,
            "abort_requested": self._abort_requested,
            "agent_state": self.agent_state.summary(),
            "events": [
                event.to_dict()
                for event in self._events
            ],
            "last_result": (
                self._last_result.to_dict()
                if self._last_result is not None
                else None
            ),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )


async def demo() -> None:
    engine = RuntimeEngine(
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

    print("Run normal input:")
    result = await engine.arun("hello PyWork")

    print(result.to_json(indent=2))
    print("\nOutput:")
    print(result.output)

    print("\nRun tool input:")
    result2 = await engine.arun("/tool echo Hello from RuntimeEngine.")

    print(result2.to_json(indent=2))
    print("\nOutput:")
    print(result2.output)

    print("\nPause / resume demo:")
    engine.pause()
    print(engine.to_json(indent=2))

    engine.resume()
    print(engine.to_json(indent=2))

    print("\nFinal engine state:")
    print(engine.to_json(indent=2))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
