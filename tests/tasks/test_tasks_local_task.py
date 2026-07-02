from __future__ import annotations

import asyncio

import pytest

from pywork.tasks.local_task import (
    LocalTaskAlreadyRunningError,
    LocalTaskBackend,
    LocalTaskInvalidRunnerError,
    LocalTaskNotFoundError,
)
from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskResult,
    TaskStatus,
    create_task_record,
)


@pytest.mark.asyncio
async def test_local_task_backend_runs_async_task_successfully() -> None:
    backend = LocalTaskBackend()
    record = create_task_record(
        "async success task",
        agent_id="verifier",
    )

    async def runner(task_record):
        await asyncio.sleep(0.01)
        return {
            "ok": True,
            "task_id": task_record.id,
        }

    result_record = await backend.run_task(
        record,
        runner,
        agent_id="verifier",
    )

    assert result_record is record
    assert record.status == TaskStatus.SUCCEEDED
    assert record.result is not None
    assert record.result.success is True
    assert record.result.value["ok"] is True
    assert record.agent_id == "verifier"
    assert record.started_at is not None
    assert record.finished_at is not None
    assert backend.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_local_task_backend_runs_sync_task_successfully() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("sync success task")

    def runner(task_record):
        return "sync result"

    result_record = await backend.run_task(
        record,
        runner,
    )

    assert result_record.status == TaskStatus.SUCCEEDED
    assert result_record.result is not None
    assert result_record.result.value == "sync result"


@pytest.mark.asyncio
async def test_local_task_backend_accepts_task_result_success() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("task result success")

    async def runner(task_record):
        return TaskResult.success_result(
            {
                "exit_code": 0,
            },
            metadata={
                "source": "runner",
            },
        )

    result_record = await backend.run_task(
        record,
        runner,
    )

    assert result_record.status == TaskStatus.SUCCEEDED
    assert result_record.result is not None
    assert result_record.result.value == {
        "exit_code": 0,
    }
    assert result_record.result.metadata["source"] == "runner"


@pytest.mark.asyncio
async def test_local_task_backend_accepts_task_result_failure() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("task result failure")

    async def runner(task_record):
        return TaskResult.failure_result(
            "pytest failed",
            error_type="TestFailure",
        )

    result_record = await backend.run_task(
        record,
        runner,
    )

    assert result_record.status == TaskStatus.FAILED
    assert result_record.error == "pytest failed"
    assert result_record.result is not None
    assert result_record.result.success is False
    assert result_record.result.error_type == "TestFailure"


@pytest.mark.asyncio
async def test_local_task_backend_catches_exceptions() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("exception task")

    async def runner(task_record):
        raise ValueError("bad value")

    result_record = await backend.run_task(
        record,
        runner,
    )

    assert result_record.status == TaskStatus.FAILED
    assert result_record.error == "bad value"
    assert result_record.result is not None
    assert result_record.result.error_type == "ValueError"
    assert result_record.result.traceback is not None
    assert "ValueError" in result_record.result.traceback
    assert backend.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_local_task_backend_can_cancel_running_task() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("cancel task")

    async def runner(task_record):
        await asyncio.sleep(10)
        return "should not finish"

    execution = await backend.start_task(
        record,
        runner,
    )

    assert backend.is_running(record.id)

    cancelled = backend.cancel_task(
        record.id,
        reason="user cancelled",
    )

    assert cancelled is True

    result_record = await execution.wait()

    assert result_record.status == TaskStatus.CANCELLED
    assert result_record.error == "user cancelled"
    assert result_record.result is not None
    assert result_record.result.error_type == "Cancelled"
    assert backend.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_local_task_backend_cancel_all() -> None:
    backend = LocalTaskBackend()

    async def runner(task_record):
        await asyncio.sleep(10)
        return "should not finish"

    record_1 = create_task_record("task 1")
    record_2 = create_task_record("task 2")

    execution_1 = await backend.start_task(
        record_1,
        runner,
    )
    execution_2 = await backend.start_task(
        record_2,
        runner,
    )

    assert len(backend.get_active_task_ids()) == 2

    count = backend.cancel_all(
        reason="cancel all",
    )

    assert count == 2

    result_1 = await execution_1.wait()
    result_2 = await execution_2.wait()

    assert result_1.status == TaskStatus.CANCELLED
    assert result_2.status == TaskStatus.CANCELLED
    assert result_1.error == "cancel all"
    assert result_2.error == "cancel all"
    assert backend.get_active_task_ids() == []


@pytest.mark.asyncio
async def test_local_task_backend_emits_events() -> None:
    events: list[TaskEvent] = []

    async def handler(event: TaskEvent):
        events.append(event)

    backend = LocalTaskBackend(
        event_handlers=[
            handler,
        ]
    )
    record = create_task_record("event task")

    async def runner(task_record):
        return "ok"

    await backend.run_task(
        record,
        runner,
    )

    event_types = [
        event.event_type
        for event in events
    ]

    assert event_types == [
        TaskEventType.QUEUED,
        TaskEventType.STARTED,
        TaskEventType.SUCCEEDED,
    ]

    assert [
        event.task_id
        for event in events
    ] == [
        record.id,
        record.id,
        record.id,
    ]


@pytest.mark.asyncio
async def test_local_task_backend_emits_failed_event() -> None:
    events: list[TaskEvent] = []

    backend = LocalTaskBackend(
        event_handlers=[
            lambda event: events.append(event),
        ]
    )
    record = create_task_record("failed event task")

    async def runner(task_record):
        raise RuntimeError("boom")

    await backend.run_task(
        record,
        runner,
    )

    assert events[-1].event_type == TaskEventType.FAILED
    assert events[-1].status == TaskStatus.FAILED
    assert events[-1].message == "boom"


@pytest.mark.asyncio
async def test_local_task_backend_duplicate_running_task_rejected() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("duplicate task")

    async def runner(task_record):
        await asyncio.sleep(10)

    execution = await backend.start_task(
        record,
        runner,
    )

    with pytest.raises(LocalTaskAlreadyRunningError):
        await backend.start_task(
            record,
            runner,
        )

    backend.cancel_task(
        record.id,
        reason="cleanup",
    )

    await execution.wait()


@pytest.mark.asyncio
async def test_local_task_backend_missing_task_errors() -> None:
    backend = LocalTaskBackend()

    with pytest.raises(LocalTaskNotFoundError):
        backend.get_execution("missing")

    with pytest.raises(LocalTaskNotFoundError):
        await backend.wait_task("missing")

    assert backend.cancel_task("missing") is False


@pytest.mark.asyncio
async def test_local_task_backend_invalid_runner_rejected() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("invalid runner")

    with pytest.raises(LocalTaskInvalidRunnerError):
        await backend.start_task(
            record,
            None,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_local_task_backend_rejects_terminal_record() -> None:
    backend = LocalTaskBackend()
    record = create_task_record("terminal task")
    record.mark_succeeded("done")

    async def runner(task_record):
        return "should not run"

    with pytest.raises(Exception):
        await backend.start_task(
            record,
            runner,
        )