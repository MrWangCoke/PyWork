from __future__ import annotations

import asyncio

import pytest

from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskResult,
    TaskStatus,
    TaskType,
)
from pywork.tasks.task_manager import (
    TaskManager,
    TaskManagerRunnerNotFoundError,
    TaskManagerTaskNotFoundError,
    create_task_manager,
)


class FakeTaskStorage:
    def __init__(self) -> None:
        self.tasks = {}
        self.events = []

    def save_task(self, record):
        self.tasks[record.id] = record

    def update_task(self, record):
        self.tasks[record.id] = record

    def get_task(self, task_id):
        return self.tasks.get(task_id)

    def list_tasks(
        self,
        *,
        status=None,
        parent_id=None,
        agent_id=None,
        limit=None,
    ):
        records = list(self.tasks.values())

        if status is not None:
            records = [
                record
                for record in records
                if record.status == status
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

        if limit is not None:
            records = records[:limit]

        return records

    def save_event(self, event):
        self.events.append(event)


@pytest.mark.asyncio
async def test_task_manager_creates_task() -> None:
    manager = create_task_manager()

    record = await manager.create_task(
        "planner task",
        task_type=TaskType.SUBAGENT,
        payload={
            "task": "plan feature",
        },
        parent_id="root",
        agent_id="planner",
        max_retries=1,
    )

    assert record.id.startswith("task_")
    assert record.name == "planner task"
    assert record.status == TaskStatus.PENDING
    assert record.task_type == TaskType.SUBAGENT
    assert record.parent_id == "root"
    assert record.agent_id == "planner"

    loaded = await manager.get_task(record.id)

    assert loaded is record


@pytest.mark.asyncio
async def test_task_manager_run_task_success() -> None:
    manager = create_task_manager()

    async def runner(record):
        await asyncio.sleep(0.01)
        return {
            "ok": True,
            "task_id": record.id,
        }

    record = await manager.run_task(
        "run verifier",
        task_type="subagent",
        agent_id="verifier",
        runner=runner,
    )

    assert record.status == TaskStatus.SUCCEEDED
    assert record.result is not None
    assert record.result.success is True
    assert record.result.value["ok"] is True
    assert record.agent_id == "verifier"


@pytest.mark.asyncio
async def test_task_manager_run_task_failure() -> None:
    manager = create_task_manager()

    async def runner(record):
        raise RuntimeError("boom")

    record = await manager.run_task(
        "failing task",
        runner=runner,
    )

    assert record.status == TaskStatus.FAILED
    assert record.error == "boom"
    assert record.result is not None
    assert record.result.error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_task_manager_start_and_wait_task() -> None:
    manager = create_task_manager()

    async def runner(record):
        await asyncio.sleep(0.01)
        return "done"

    record = await manager.create_task(
        "manual start",
        runner=runner,
    )

    execution = await manager.start_task(record.id)

    assert record.id in manager.get_active_task_ids()

    result = await manager.wait_task(record.id)

    assert result is record
    assert result.status == TaskStatus.SUCCEEDED
    assert result.result is not None
    assert result.result.value == "done"
    assert manager.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_task_manager_cancel_running_task() -> None:
    manager = create_task_manager()

    async def runner(record):
        await asyncio.sleep(10)
        return "should not finish"

    record = await manager.create_task(
        "cancel me",
        runner=runner,
    )

    await manager.start_task(record.id)

    assert record.id in manager.get_active_task_ids()

    cancelled = await manager.cancel_task(
        record.id,
        reason="user cancelled",
    )

    assert cancelled.status == TaskStatus.CANCELLED
    assert cancelled.error == "user cancelled"
    assert cancelled.result is not None
    assert cancelled.result.error_type == "Cancelled"
    assert manager.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_task_manager_cancel_pending_task() -> None:
    manager = create_task_manager()

    record = await manager.create_task(
        "pending task",
    )

    cancelled = await manager.cancel_task(
        record.id,
        reason="not needed",
    )

    assert cancelled.status == TaskStatus.CANCELLED
    assert cancelled.error == "not needed"


@pytest.mark.asyncio
async def test_task_manager_retry_failed_task() -> None:
    manager = create_task_manager()

    attempts = 0

    async def runner(record):
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return TaskResult.failure_result(
                "first failure",
                error_type="TestFailure",
            )

        return TaskResult.success_result(
            {
                "attempt": attempts,
            }
        )

    record = await manager.run_task(
        "retryable task",
        max_retries=2,
        runner=runner,
    )

    assert record.status == TaskStatus.FAILED
    assert record.can_retry is True
    assert attempts == 1

    retried = await manager.retry_task(
        record.id,
        wait=True,
    )

    assert retried.status == TaskStatus.SUCCEEDED
    assert retried.result is not None
    assert retried.result.value == {
        "attempt": 2,
    }
    assert retried.retry_count == 1


@pytest.mark.asyncio
async def test_task_manager_retry_requires_runner() -> None:
    manager = create_task_manager()

    record = await manager.create_task(
        "missing runner",
        max_retries=1,
    )

    record.mark_failed("failed")
    await manager.register_task(record)

    with pytest.raises(TaskManagerRunnerNotFoundError):
        await manager.retry_task(record.id)


@pytest.mark.asyncio
async def test_task_manager_list_tasks_filters() -> None:
    manager = create_task_manager()

    root = await manager.create_task(
        "root",
        agent_id="planner",
    )
    child = await manager.create_task(
        "child",
        parent_id=root.id,
        agent_id="reviewer",
    )
    other = await manager.create_task(
        "other",
        agent_id="planner",
    )

    child.mark_succeeded("ok")

    records_by_parent = await manager.list_tasks(
        parent_id=root.id,
    )

    assert [
        record.id
        for record in records_by_parent
    ] == [
        child.id,
    ]

    records_by_agent = await manager.list_tasks(
        agent_id="planner",
    )

    assert {
        record.id
        for record in records_by_agent
    } == {
        root.id,
        other.id,
    }

    succeeded = await manager.list_tasks(
        status="succeeded",
    )

    assert [
        record.id
        for record in succeeded
    ] == [
        child.id,
    ]


@pytest.mark.asyncio
async def test_task_manager_watch_task() -> None:
    manager = create_task_manager()

    async def runner(record):
        await asyncio.sleep(0.05)
        return "ok"

    record = await manager.create_task(
        "watch me",
        runner=runner,
    )

    await manager.start_task(record.id)

    statuses = []

    async for item in manager.watch_task(
        record.id,
        poll_interval=0.01,
        timeout=1,
    ):
        statuses.append(item.status)

    assert TaskStatus.RUNNING in statuses
    assert statuses[-1] == TaskStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_task_manager_emits_events() -> None:
    events: list[TaskEvent] = []

    async def handler(event: TaskEvent):
        events.append(event)

    manager = create_task_manager(
        event_handlers=[
            handler,
        ]
    )

    async def runner(record):
        return "ok"

    record = await manager.run_task(
        "event task",
        runner=runner,
    )

    assert record.status == TaskStatus.SUCCEEDED

    event_types = [
        event.event_type
        for event in events
    ]

    assert TaskEventType.CREATED in event_types
    assert TaskEventType.QUEUED in event_types
    assert TaskEventType.STARTED in event_types
    assert TaskEventType.SUCCEEDED in event_types


@pytest.mark.asyncio
async def test_task_manager_uses_storage() -> None:
    storage = FakeTaskStorage()
    manager = create_task_manager(
        storage=storage,
    )

    async def runner(record):
        return "ok"

    record = await manager.run_task(
        "stored task",
        runner=runner,
    )

    assert record.id in storage.tasks
    assert storage.tasks[record.id].status == TaskStatus.SUCCEEDED

    event_types = [
        event.event_type
        for event in storage.events
    ]

    assert TaskEventType.CREATED in event_types
    assert TaskEventType.SUCCEEDED in event_types


@pytest.mark.asyncio
async def test_task_manager_missing_task_errors() -> None:
    manager = TaskManager()

    with pytest.raises(TaskManagerTaskNotFoundError):
        await manager.get_task("missing")