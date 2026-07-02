from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskRecord,
    TaskResult,
    TaskStateError,
    TaskStatus,
)


class LocalTaskError(Exception):
    """本地 Task 后端基础异常。"""


class LocalTaskNotFoundError(LocalTaskError):
    """找不到本地运行中的 Task。"""


class LocalTaskAlreadyRunningError(LocalTaskError):
    """Task 已经在本地后端运行。"""


class LocalTaskInvalidRunnerError(LocalTaskError):
    """Task runner 不合法。"""


LocalTaskRunner = Callable[[TaskRecord], Any | Awaitable[Any]]
LocalTaskEventHandler = Callable[[TaskEvent], Any | Awaitable[Any]]


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def create_task_event(
    record: TaskRecord,
    event_type: TaskEventType,
    *,
    message: str | None = None,
    result: TaskResult | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskEvent:
    return TaskEvent(
        task_id=record.id,
        event_type=event_type,
        status=record.status,
        message=message,
        result=result,
        metadata=metadata or {},
    )


def apply_runner_output_to_record(
    record: TaskRecord,
    output: Any,
) -> None:
    """
    把 runner 返回值写回 TaskRecord。

    支持两种返回：
    1. 普通值：
       record.mark_succeeded(value)

    2. TaskResult：
       success=True  -> succeeded
       success=False -> failed
    """

    if isinstance(output, TaskResult):
        if output.success:
            record.mark_succeeded(
                output.value,
                result=output,
                metadata=output.metadata,
            )
            return

        record.mark_failed(
            output.error or "task failed",
            error_type=output.error_type,
            result=output,
            metadata=output.metadata,
        )
        return

    record.mark_succeeded(output)


@dataclass(slots=True)
class LocalTaskExecution:
    """
    一个本地 asyncio.Task 的运行句柄。
    """

    record: TaskRecord
    asyncio_task: asyncio.Task[TaskRecord]
    started_at: float = field(default_factory=time.time)
    cancel_reason: str | None = None

    @property
    def task_id(self) -> str:
        return self.record.id

    @property
    def done(self) -> bool:
        return self.asyncio_task.done()

    @property
    def cancelled(self) -> bool:
        return self.record.status == TaskStatus.CANCELLED

    @property
    def duration_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    def cancel(
        self,
        reason: str | None = None,
    ) -> None:
        self.cancel_reason = reason or "task cancelled"
        self.record.metadata["cancel_reason"] = self.cancel_reason

        if not self.asyncio_task.done():
            self.asyncio_task.cancel()

    async def wait(
        self,
        timeout: float | None = None,
    ) -> TaskRecord:
        if timeout is None:
            return await self.asyncio_task

        return await asyncio.wait_for(
            self.asyncio_task,
            timeout=timeout,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "done": self.done,
            "cancelled": self.cancelled,
            "duration_ms": self.duration_ms,
            "record": self.record.to_dict(),
        }


class LocalTaskBackend:
    """
    asyncio.Task 本地执行后端。

    负责：
    - start_task
    - wait_task
    - cancel_task
    - cancel_all
    - active task 查询
    - TaskEvent 分发

    不负责：
    - SQLite 持久化
    - 重试策略
    - SubAgent 路由
    - 权限审批
    """

    def __init__(
        self,
        *,
        event_handlers: list[LocalTaskEventHandler] | None = None,
    ) -> None:
        self._active: dict[str, LocalTaskExecution] = {}
        self._event_handlers: list[LocalTaskEventHandler] = list(
            event_handlers or []
        )

    def add_event_handler(
        self,
        handler: LocalTaskEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: LocalTaskEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: TaskEvent,
    ) -> None:
        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    def get_active_task_ids(self) -> list[str]:
        return list(self._active.keys())

    def get_active_executions(self) -> list[LocalTaskExecution]:
        return list(self._active.values())

    def get_active_records(self) -> list[TaskRecord]:
        return [
            execution.record
            for execution in self._active.values()
        ]

    def get_execution(
        self,
        task_id: str,
    ) -> LocalTaskExecution:
        execution = self._active.get(task_id)

        if execution is None:
            raise LocalTaskNotFoundError(f"Task is not running: {task_id}")

        return execution

    def is_running(
        self,
        task_id: str,
    ) -> bool:
        execution = self._active.get(task_id)

        return execution is not None and not execution.done

    async def start_task(
        self,
        record: TaskRecord,
        runner: LocalTaskRunner,
        *,
        agent_id: str | None = None,
        task_name: str | None = None,
    ) -> LocalTaskExecution:
        if not callable(runner):
            raise LocalTaskInvalidRunnerError("runner must be callable")

        if record.id in self._active:
            raise LocalTaskAlreadyRunningError(
                f"Task is already running: {record.id}"
            )

        if record.is_terminal:
            raise TaskStateError(
                f"Cannot start terminal task {record.id}: {record.status.value}"
            )

        record.mark_queued()

        await self.emit_event(
            create_task_event(
                record,
                TaskEventType.QUEUED,
                message="task queued",
            )
        )

        asyncio_task = asyncio.create_task(
            self._run_record(
                record,
                runner,
                agent_id=agent_id,
            ),
            name=task_name or record.name,
        )

        execution = LocalTaskExecution(
            record=record,
            asyncio_task=asyncio_task,
        )

        self._active[record.id] = execution

        # 让新创建的 asyncio.Task 至少运行到 _run_record() 的 try 块里。
        #
        # 否则调用方可能在 start_task() 返回后立刻 cancel_task()，
        # 此时 coroutine 还没开始执行，CancelledError 会在进入函数前抛出，
        # 导致 _run_record() 里的 except asyncio.CancelledError 捕获不到。
        await asyncio.sleep(0)

        return execution

    async def run_task(
        self,
        record: TaskRecord,
        runner: LocalTaskRunner,
        *,
        agent_id: str | None = None,
        timeout: float | None = None,
        task_name: str | None = None,
    ) -> TaskRecord:
        execution = await self.start_task(
            record,
            runner,
            agent_id=agent_id,
            task_name=task_name,
        )

        return await execution.wait(timeout=timeout)

    async def wait_task(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> TaskRecord:
        execution = self.get_execution(task_id)

        return await execution.wait(timeout=timeout)

    def cancel_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> bool:
        execution = self._active.get(task_id)

        if execution is None:
            return False

        execution.cancel(reason)

        return True

    def cancel_all(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        count = 0

        for execution in list(self._active.values()):
            execution.cancel(reason or "all local tasks cancelled")
            count += 1

        return count

    async def _run_record(
        self,
        record: TaskRecord,
        runner: LocalTaskRunner,
        *,
        agent_id: str | None = None,
    ) -> TaskRecord:
        try:
            record.mark_running(agent_id=agent_id)

            await self.emit_event(
                create_task_event(
                    record,
                    TaskEventType.STARTED,
                    message="task started",
                )
            )

            output = await maybe_await(runner(record))

            apply_runner_output_to_record(
                record,
                output,
            )

            if record.status == TaskStatus.SUCCEEDED:
                await self.emit_event(
                    create_task_event(
                        record,
                        TaskEventType.SUCCEEDED,
                        message="task succeeded",
                        result=record.result,
                    )
                )
            elif record.status == TaskStatus.FAILED:
                await self.emit_event(
                    create_task_event(
                        record,
                        TaskEventType.FAILED,
                        message=record.error or "task failed",
                        result=record.result,
                    )
                )

            return record

        except asyncio.CancelledError:
            reason = (
                record.metadata.get("cancel_reason")
                if isinstance(record.metadata, dict)
                else None
            )

            record.mark_cancelled(
                str(reason or "task cancelled"),
            )

            await self.emit_event(
                create_task_event(
                    record,
                    TaskEventType.CANCELLED,
                    message=record.error,
                    result=record.result,
                )
            )

            return record

        except Exception as exc:
            record.mark_failed_from_exception(exc)

            await self.emit_event(
                create_task_event(
                    record,
                    TaskEventType.FAILED,
                    message=record.error,
                    result=record.result,
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )

            return record

        finally:
            self._active.pop(record.id, None)


def create_local_task_backend(
    *,
    event_handlers: list[LocalTaskEventHandler] | None = None,
) -> LocalTaskBackend:
    return LocalTaskBackend(
        event_handlers=event_handlers,
    )


__all__ = [
    "LocalTaskAlreadyRunningError",
    "LocalTaskBackend",
    "LocalTaskError",
    "LocalTaskEventHandler",
    "LocalTaskExecution",
    "LocalTaskInvalidRunnerError",
    "LocalTaskNotFoundError",
    "LocalTaskRunner",
    "apply_runner_output_to_record",
    "create_local_task_backend",
    "create_task_event",
    "maybe_await",
]