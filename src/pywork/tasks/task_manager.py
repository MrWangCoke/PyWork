from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pywork.tasks.local_task import (
    LocalTaskBackend,
    LocalTaskExecution,
    LocalTaskNotFoundError,
    LocalTaskRunner,
    create_local_task_backend,
)
from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskRecord,
    TaskSpec,
    TaskStateError,
    TaskStatus,
    TaskType,
    create_task_record,
    normalize_task_status,
)


class TaskManagerError(Exception):
    """TaskManager 基础异常。"""


class TaskManagerTaskNotFoundError(TaskManagerError):
    """找不到任务。"""


class TaskManagerRunnerNotFoundError(TaskManagerError):
    """找不到任务 runner。"""


class TaskManagerStorageError(TaskManagerError):
    """Task 持久化错误。"""


class TaskStorageProtocol(Protocol):
    """
    TaskStorage 协议。

    下一步 tasks/task_storage.py 会实现这个协议。
    这里用 Protocol 是为了让 TaskManager 先能工作，
    后面 SQLite storage 可以无缝接进来。
    """

    def save_task(self, record: TaskRecord) -> Any:
        ...

    def update_task(self, record: TaskRecord) -> Any:
        ...

    def get_task(self, task_id: str) -> TaskRecord | None:
        ...

    def list_tasks(
        self,
        *,
        status: TaskStatus | str | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        ...

    def save_event(self, event: TaskEvent) -> Any:
        ...


TaskManagerEventHandler = Callable[[TaskEvent], Any | Awaitable[Any]]


@dataclass(slots=True, frozen=True)
class TaskManagerConfig:
    """
    TaskManager 配置。

    default_timeout_seconds:
        run_task / wait_task 没显式传 timeout 时，可以使用这个默认值。
        None 表示不限制。

    watch_poll_interval:
        watch_task 轮询间隔。

    persist_events:
        如果 storage 支持 save_event，是否保存 TaskEvent。
    """

    default_timeout_seconds: float | None = None
    watch_poll_interval: float = 0.05
    persist_events: bool = True


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def create_task_event(
    record: TaskRecord,
    event_type: TaskEventType,
    *,
    message: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskEvent:
    return TaskEvent(
        task_id=record.id,
        event_type=event_type,
        status=record.status,
        message=message,
        result=record.result,
        metadata=metadata or {},
    )


class TaskManager:
    """
    Task 总控层。

    职责：
    - create_task
    - start_task
    - run_task
    - wait_task
    - cancel_task
    - retry_task
    - get_task
    - list_tasks
    - watch_task

    它组合：
    - TaskRecord / TaskSpec
    - LocalTaskBackend
    - 可选 TaskStorage

    不负责：
    - 具体业务逻辑
    - SubAgent 路由
    - LLM 调用
    - SQLite 细节
    """

    def __init__(
        self,
        *,
        backend: LocalTaskBackend | None = None,
        storage: TaskStorageProtocol | None = None,
        config: TaskManagerConfig | None = None,
        event_handlers: Sequence[TaskManagerEventHandler] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or TaskManagerConfig()
        self.storage = storage
        self.metadata = metadata or {}

        self._records: dict[str, TaskRecord] = {}
        self._runners: dict[str, LocalTaskRunner] = {}
        self._event_handlers: list[TaskManagerEventHandler] = list(
            event_handlers or []
        )

        self.backend = backend or create_local_task_backend()
        self.backend.add_event_handler(self._handle_backend_event)

    # ------------------------------------------------------------------
    # event
    # ------------------------------------------------------------------

    def add_event_handler(
        self,
        handler: TaskManagerEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: TaskManagerEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: TaskEvent,
    ) -> None:
        if self.config.persist_events:
            await self._persist_event(event)

        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    async def _handle_backend_event(
        self,
        event: TaskEvent,
    ) -> None:
        record = self._records.get(event.task_id)

        if record is not None:
            await self._persist_record(record)

        await self.emit_event(event)

    # ------------------------------------------------------------------
    # storage
    # ------------------------------------------------------------------

    async def _persist_record(
        self,
        record: TaskRecord,
        *,
        create: bool = False,
    ) -> None:
        self._records[record.id] = record

        if self.storage is None:
            return

        try:
            if create and hasattr(self.storage, "save_task"):
                await maybe_await(self.storage.save_task(record))
                return

            if hasattr(self.storage, "update_task"):
                await maybe_await(self.storage.update_task(record))
                return

            if hasattr(self.storage, "save_task"):
                await maybe_await(self.storage.save_task(record))
                return

        except Exception as exc:
            raise TaskManagerStorageError(str(exc)) from exc

    async def _persist_event(
        self,
        event: TaskEvent,
    ) -> None:
        if self.storage is None:
            return

        save_event = getattr(self.storage, "save_event", None)

        if not callable(save_event):
            return

        try:
            await maybe_await(save_event(event))
        except Exception as exc:
            raise TaskManagerStorageError(str(exc)) from exc

    async def _load_record_from_storage(
        self,
        task_id: str,
    ) -> TaskRecord | None:
        if self.storage is None:
            return None

        get_task = getattr(self.storage, "get_task", None)

        if not callable(get_task):
            return None

        try:
            record = await maybe_await(get_task(task_id))
        except Exception as exc:
            raise TaskManagerStorageError(str(exc)) from exc

        if record is not None:
            self._records[record.id] = record

        return record

    # ------------------------------------------------------------------
    # create / register
    # ------------------------------------------------------------------

    async def create_task(
        self,
        name: str | None = None,
        *,
        spec: TaskSpec | None = None,
        task_type: TaskType | str = TaskType.GENERIC,
        payload: dict[str, Any] | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        created_by: str | None = None,
        task_id: str | None = None,
        runner: LocalTaskRunner | None = None,
    ) -> TaskRecord:
        if spec is not None:
            record = spec.to_record(task_id=task_id)
        else:
            if not name:
                raise TaskManagerError("create_task requires name or spec")

            record = create_task_record(
                name,
                task_type=task_type,
                payload=payload or {},
                parent_id=parent_id,
                agent_id=agent_id,
                metadata=metadata or {},
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                created_by=created_by,
                task_id=task_id,
            )

        if runner is not None:
            self._runners[record.id] = runner

        await self._persist_record(
            record,
            create=True,
        )

        await self.emit_event(
            create_task_event(
                record,
                TaskEventType.CREATED,
                message="task created",
                metadata={
                    "source": "task_manager",
                },
            )
        )

        return record

    async def register_task(
        self,
        record: TaskRecord,
        *,
        runner: LocalTaskRunner | None = None,
    ) -> TaskRecord:
        if runner is not None:
            self._runners[record.id] = runner

        await self._persist_record(
            record,
            create=True,
        )

        return record

    def set_runner(
        self,
        task_id: str,
        runner: LocalTaskRunner,
    ) -> None:
        if not callable(runner):
            raise TaskManagerError("runner must be callable")

        self._runners[task_id] = runner

    def get_runner(
        self,
        task_id: str,
    ) -> LocalTaskRunner:
        runner = self._runners.get(task_id)

        if runner is None:
            raise TaskManagerRunnerNotFoundError(
                f"No runner registered for task: {task_id}"
            )

        return runner

    # ------------------------------------------------------------------
    # get / list
    # ------------------------------------------------------------------

    async def get_task(
        self,
        task_id: str,
    ) -> TaskRecord:
        record = self._records.get(task_id)

        if record is not None:
            return record

        record = await self._load_record_from_storage(task_id)

        if record is not None:
            return record

        raise TaskManagerTaskNotFoundError(f"Task not found: {task_id}")

    async def list_tasks(
        self,
        *,
        status: TaskStatus | str | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        normalized_status = (
            normalize_task_status(status)
            if status is not None
            else None
        )

        records = list(self._records.values())

        if normalized_status is not None:
            records = [
                record
                for record in records
                if record.status == normalized_status
            ]

        if parent_id is not None:
            records = [
                record
                for record in records
                if record.parent_id == parent_id
            ]

        if agent_id is not None:
            records = [
                record
                for record in records
                if record.agent_id == agent_id
            ]

        records.sort(
            key=lambda record: record.created_at,
            reverse=True,
        )

        if limit is not None:
            records = records[:limit]

        return records

    def get_active_task_ids(self) -> list[str]:
        return self.backend.get_active_task_ids()

    def get_active_tasks(self) -> list[TaskRecord]:
        return self.backend.get_active_records()

    # ------------------------------------------------------------------
    # start / run / wait
    # ------------------------------------------------------------------

    async def start_task(
        self,
        task_id: str,
        *,
        runner: LocalTaskRunner | None = None,
        agent_id: str | None = None,
        task_name: str | None = None,
    ) -> LocalTaskExecution:
        record = await self.get_task(task_id)

        if runner is not None:
            self.set_runner(task_id, runner)

        selected_runner = self.get_runner(task_id)

        execution = await self.backend.start_task(
            record,
            selected_runner,
            agent_id=agent_id,
            task_name=task_name,
        )

        await self._persist_record(record)

        return execution

    async def run_task(
        self,
        name: str | None = None,
        *,
        spec: TaskSpec | None = None,
        record: TaskRecord | None = None,
        runner: LocalTaskRunner | None = None,
        task_type: TaskType | str = TaskType.GENERIC,
        payload: dict[str, Any] | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        created_by: str | None = None,
        task_id: str | None = None,
    ) -> TaskRecord:
        if record is not None:
            await self.register_task(
                record,
                runner=runner,
            )
        else:
            record = await self.create_task(
                name,
                spec=spec,
                task_type=task_type,
                payload=payload,
                parent_id=parent_id,
                agent_id=agent_id,
                metadata=metadata,
                max_retries=max_retries,
                timeout_seconds=timeout_seconds,
                created_by=created_by,
                task_id=task_id,
                runner=runner,
            )

        timeout = (
            record.timeout_seconds
            if record.timeout_seconds is not None
            else self.config.default_timeout_seconds
        )

        execution = await self.start_task(
            record.id,
            agent_id=agent_id,
        )

        try:
            result = await execution.wait(timeout=timeout)
        except asyncio.TimeoutError:
            self.backend.cancel_task(
                record.id,
                reason="task timed out",
            )
            result = await execution.wait()

        await self._persist_record(result)

        return result

    async def wait_task(
        self,
        task_id: str,
        *,
        timeout: float | None = None,
    ) -> TaskRecord:
        if self.backend.is_running(task_id):
            try:
                record = await self.backend.wait_task(
                    task_id,
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise

            await self._persist_record(record)

            return record

        record = await self.get_task(task_id)

        if record.is_terminal:
            return record

        raise LocalTaskNotFoundError(
            f"Task is not currently running: {task_id}"
        )

    # ------------------------------------------------------------------
    # cancel / stop
    # ------------------------------------------------------------------

    async def cancel_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        wait: bool = True,
    ) -> TaskRecord:
        record = await self.get_task(task_id)

        if self.backend.is_running(task_id):
            cancelled = self.backend.cancel_task(
                task_id,
                reason=reason,
            )

            if cancelled and wait:
                record = await self.wait_task(task_id)
            else:
                record = await self.get_task(task_id)

            await self._persist_record(record)

            return record

        if record.is_terminal:
            return record

        record.mark_cancelled(reason or "task cancelled")
        await self._persist_record(record)

        await self.emit_event(
            create_task_event(
                record,
                TaskEventType.CANCELLED,
                message=record.error,
                metadata={
                    "source": "task_manager",
                },
            )
        )

        return record

    async def stop_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        wait: bool = True,
    ) -> TaskRecord:
        return await self.cancel_task(
            task_id,
            reason=reason,
            wait=wait,
        )

    async def cancel_all(
        self,
        *,
        reason: str | None = None,
        wait: bool = True,
    ) -> list[TaskRecord]:
        active_ids = self.backend.get_active_task_ids()

        self.backend.cancel_all(
            reason=reason or "all tasks cancelled",
        )

        results: list[TaskRecord] = []

        if wait:
            for task_id in active_ids:
                results.append(
                    await self.wait_task(task_id)
                )

        return results

    # ------------------------------------------------------------------
    # retry
    # ------------------------------------------------------------------

    async def retry_task(
        self,
        task_id: str,
        *,
        runner: LocalTaskRunner | None = None,
        agent_id: str | None = None,
        wait: bool = False,
        timeout: float | None = None,
    ) -> LocalTaskExecution | TaskRecord:
        record = await self.get_task(task_id)

        if runner is not None:
            self.set_runner(task_id, runner)

        selected_runner = self.get_runner(task_id)

        record.mark_retrying(
            reason=record.error,
        )

        await self._persist_record(record)

        await self.emit_event(
            create_task_event(
                record,
                TaskEventType.RETRYING,
                message="task retrying",
                metadata={
                    "source": "task_manager",
                    "retry_count": record.retry_count,
                },
            )
        )

        record.prepare_next_attempt()
        await self._persist_record(record)

        execution = await self.backend.start_task(
            record,
            selected_runner,
            agent_id=agent_id or record.agent_id,
            task_name=record.name,
        )

        if not wait:
            return execution

        try:
            result = await execution.wait(
                timeout=timeout or record.timeout_seconds,
            )
        except asyncio.TimeoutError:
            self.backend.cancel_task(
                record.id,
                reason="task retry timed out",
            )
            result = await execution.wait()

        await self._persist_record(result)

        return result

    # ------------------------------------------------------------------
    # watch
    # ------------------------------------------------------------------

    async def watch_task(
        self,
        task_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
        include_duplicates: bool = False,
    ) -> AsyncIterator[TaskRecord]:
        interval = (
            self.config.watch_poll_interval
            if poll_interval is None
            else poll_interval
        )

        started_at = time.time()
        last_status: TaskStatus | None = None
        last_updated_at: float | None = None

        while True:
            record = await self.get_task(task_id)

            changed = (
                include_duplicates
                or record.status != last_status
                or record.updated_at != last_updated_at
            )

            if changed:
                yield record
                last_status = record.status
                last_updated_at = record.updated_at

            if record.is_terminal:
                break

            if timeout is not None and time.time() - started_at > timeout:
                raise asyncio.TimeoutError(
                    f"watch_task timed out: {task_id}"
                )

            await asyncio.sleep(interval)


def create_task_manager(
    *,
    backend: LocalTaskBackend | None = None,
    storage: TaskStorageProtocol | None = None,
    config: TaskManagerConfig | None = None,
    event_handlers: Sequence[TaskManagerEventHandler] | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskManager:
    return TaskManager(
        backend=backend,
        storage=storage,
        config=config,
        event_handlers=event_handlers,
        metadata=metadata,
    )


__all__ = [
    "TaskManager",
    "TaskManagerConfig",
    "TaskManagerError",
    "TaskManagerEventHandler",
    "TaskManagerRunnerNotFoundError",
    "TaskManagerStorageError",
    "TaskManagerTaskNotFoundError",
    "TaskStorageProtocol",
    "create_task_manager",
]