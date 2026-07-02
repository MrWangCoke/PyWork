from __future__ import annotations

import sqlite3

import pytest

from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskResult,
    TaskStatus,
    TaskType,
    create_task_record,
)
from pywork.tasks.task_manager import create_task_manager
from pywork.tasks.task_storage import (
    SQLiteTaskStorage,
    TaskStorageNotFoundError,
    create_sqlite_task_storage,
)


def test_sqlite_task_storage_initializes_database(tmp_path) -> None:
    db_path = tmp_path / "tasks.sqlite3"

    storage = create_sqlite_task_storage(db_path)

    assert db_path.exists()

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }

    assert "tasks" in tables
    assert "task_events" in tables

    storage.close()


def test_sqlite_task_storage_save_and_get_task_roundtrip(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    record = create_task_record(
        "planner task",
        task_type=TaskType.SUBAGENT,
        payload={
            "task": "plan implementation",
        },
        parent_id="root_task",
        agent_id="planner",
        metadata={
            "source": "test",
        },
        max_retries=2,
        timeout_seconds=30,
        created_by="main_agent",
        task_id="task_fixed",
    )

    storage.save_task(record)

    loaded = storage.get_task("task_fixed")

    assert loaded is not None
    assert loaded.id == "task_fixed"
    assert loaded.name == "planner task"
    assert loaded.task_type == TaskType.SUBAGENT
    assert loaded.status == TaskStatus.PENDING
    assert loaded.parent_id == "root_task"
    assert loaded.agent_id == "planner"
    assert loaded.payload["task"] == "plan implementation"
    assert loaded.metadata["source"] == "test"
    assert loaded.max_retries == 2
    assert loaded.timeout_seconds == 30
    assert loaded.created_by == "main_agent"

    storage.close()


def test_sqlite_task_storage_update_task_result(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    record = create_task_record(
        "verifier task",
        task_type="subagent",
        agent_id="verifier",
        task_id="task_verify",
    )

    storage.save_task(record)

    record.mark_running()
    record.mark_succeeded(
        {
            "exit_code": 0,
            "passed": True,
        },
        metadata={
            "stdout": "1 passed",
        },
    )

    storage.update_task(record)

    loaded = storage.require_task("task_verify")

    assert loaded.status == TaskStatus.SUCCEEDED
    assert loaded.result is not None
    assert loaded.result.success is True
    assert loaded.result.value == {
        "exit_code": 0,
        "passed": True,
    }
    assert loaded.result.metadata["stdout"] == "1 passed"
    assert loaded.finished_at is not None

    storage.close()


def test_sqlite_task_storage_require_task_raises(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    with pytest.raises(TaskStorageNotFoundError):
        storage.require_task("missing")

    storage.close()


def test_sqlite_task_storage_list_tasks_filters(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    root = create_task_record(
        "root task",
        agent_id="planner",
        task_id="task_root",
    )
    child = create_task_record(
        "child task",
        parent_id=root.id,
        agent_id="reviewer",
        task_id="task_child",
    )
    other = create_task_record(
        "other task",
        agent_id="planner",
        task_id="task_other",
    )

    child.mark_succeeded("ok")

    storage.save_task(root)
    storage.save_task(child)
    storage.save_task(other)

    by_parent = storage.list_tasks(parent_id=root.id)

    assert [
        record.id
        for record in by_parent
    ] == [
        child.id,
    ]

    by_agent = storage.list_tasks(agent_id="planner")

    assert {
        record.id
        for record in by_agent
    } == {
        root.id,
        other.id,
    }

    succeeded = storage.list_tasks(status="succeeded")

    assert [
        record.id
        for record in succeeded
    ] == [
        child.id,
    ]

    assert storage.count_tasks() == 3
    assert storage.count_tasks(agent_id="planner") == 2
    assert storage.count_tasks(status="succeeded") == 1

    storage.close()


def test_sqlite_task_storage_limit_and_order(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    first = create_task_record("first", task_id="task_first")
    second = create_task_record("second", task_id="task_second")
    third = create_task_record("third", task_id="task_third")

    first.created_at = 1
    second.created_at = 2
    third.created_at = 3

    storage.save_task(first)
    storage.save_task(second)
    storage.save_task(third)

    records = storage.list_tasks(limit=2)

    assert [
        record.id
        for record in records
    ] == [
        "task_third",
        "task_second",
    ]

    storage.close()


def test_sqlite_task_storage_save_and_list_events(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    record = create_task_record(
        "event task",
        task_id="task_event",
    )
    record.mark_running()

    storage.save_task(record)

    started = TaskEvent(
        task_id=record.id,
        event_type=TaskEventType.STARTED,
        status=TaskStatus.RUNNING,
        message="task started",
        metadata={
            "source": "test",
        },
    )

    record.mark_succeeded("ok")

    succeeded = TaskEvent(
        task_id=record.id,
        event_type=TaskEventType.SUCCEEDED,
        status=TaskStatus.SUCCEEDED,
        message="task succeeded",
        result=TaskResult.success_result("ok"),
    )

    storage.save_event(started)
    storage.save_event(succeeded)

    events = storage.list_events(task_id=record.id)

    assert [
        event.event_type
        for event in events
    ] == [
        TaskEventType.STARTED,
        TaskEventType.SUCCEEDED,
    ]

    assert events[0].message == "task started"
    assert events[0].metadata["source"] == "test"
    assert events[1].result is not None
    assert events[1].result.value == "ok"

    succeeded_events = storage.list_events(
        task_id=record.id,
        event_type=TaskEventType.SUCCEEDED,
    )

    assert len(succeeded_events) == 1
    assert succeeded_events[0].event_type == TaskEventType.SUCCEEDED

    storage.close()


def test_sqlite_task_storage_delete_task(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")

    record = create_task_record(
        "delete me",
        task_id="task_delete",
    )
    storage.save_task(record)
    storage.save_event(
        TaskEvent(
            task_id=record.id,
            event_type=TaskEventType.CREATED,
            status=TaskStatus.PENDING,
        )
    )

    deleted = storage.delete_task(
        record.id,
        delete_events=True,
    )

    assert deleted is True
    assert storage.get_task(record.id) is None
    assert storage.list_events(task_id=record.id) == []

    storage.close()


@pytest.mark.asyncio
async def test_task_manager_works_with_sqlite_storage(tmp_path) -> None:
    storage = SQLiteTaskStorage(tmp_path / "tasks.sqlite3")
    manager = create_task_manager(
        storage=storage,
    )

    async def runner(record):
        return {
            "ok": True,
        }

    record = await manager.run_task(
        "stored manager task",
        task_type="subagent",
        agent_id="verifier",
        runner=runner,
    )

    loaded = storage.require_task(record.id)

    assert loaded.status == TaskStatus.SUCCEEDED
    assert loaded.result is not None
    assert loaded.result.value == {
        "ok": True,
    }

    events = storage.list_events(task_id=record.id)

    event_types = [
        event.event_type
        for event in events
    ]

    assert TaskEventType.CREATED in event_types
    assert TaskEventType.QUEUED in event_types
    assert TaskEventType.STARTED in event_types
    assert TaskEventType.SUCCEEDED in event_types

    storage.close()