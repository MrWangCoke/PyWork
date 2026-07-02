from __future__ import annotations

import json

import pytest

from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskRecord,
    TaskResult,
    TaskStateError,
    TaskStatus,
    TaskType,
    create_task_record,
    is_terminal_status,
    normalize_task_status,
    normalize_task_type,
)


def test_create_task_record_defaults() -> None:
    task = create_task_record(
        "Run planner",
        task_type=TaskType.SUBAGENT,
        parent_id="parent_1",
        agent_id="planner_1",
        payload={
            "task": "plan implementation",
        },
        metadata={
            "source": "test",
        },
    )

    assert task.id.startswith("task_")
    assert task.name == "Run planner"
    assert task.task_type == TaskType.SUBAGENT
    assert task.status == TaskStatus.PENDING
    assert task.parent_id == "parent_1"
    assert task.agent_id == "planner_1"
    assert task.payload["task"] == "plan implementation"
    assert task.metadata["source"] == "test"
    assert task.result is None
    assert task.is_terminal is False


def test_task_status_helpers() -> None:
    assert normalize_task_status("running") == TaskStatus.RUNNING
    assert normalize_task_type("subagent") == TaskType.SUBAGENT

    assert is_terminal_status("succeeded") is True
    assert is_terminal_status(TaskStatus.FAILED) is True
    assert is_terminal_status(TaskStatus.RUNNING) is False


def test_task_result_success_and_failure() -> None:
    success = TaskResult.success_result(
        {
            "ok": True,
        },
        metadata={
            "agent": "verifier",
        },
    )

    assert success.success is True
    assert success.value == {
        "ok": True,
    }
    assert success.error is None
    assert success.metadata["agent"] == "verifier"

    failure = TaskResult.failure_result(
        "boom",
        error_type="RuntimeError",
    )

    assert failure.success is False
    assert failure.error == "boom"
    assert failure.error_type == "RuntimeError"


def test_task_lifecycle_success() -> None:
    task = create_task_record(
        "Verify tests",
        task_type="subagent",
        agent_id="verifier",
    )

    task.mark_queued()
    assert task.status == TaskStatus.QUEUED

    task.mark_running()
    assert task.status == TaskStatus.RUNNING
    assert task.started_at is not None
    assert task.duration_ms is not None

    task.mark_succeeded(
        {
            "exit_code": 0,
        }
    )

    assert task.status == TaskStatus.SUCCEEDED
    assert task.is_terminal is True
    assert task.result is not None
    assert task.result.success is True
    assert task.result.value == {
        "exit_code": 0,
    }
    assert task.finished_at is not None


def test_task_lifecycle_failed_and_retry() -> None:
    task = create_task_record(
        "Run debugger",
        task_type="subagent",
        max_retries=2,
    )

    task.mark_running()
    task.mark_failed(
        "test failed",
        error_type="AssertionError",
    )

    assert task.status == TaskStatus.FAILED
    assert task.can_retry is True
    assert task.error == "test failed"
    assert task.result is not None
    assert task.result.success is False

    task.mark_retrying(
        reason="retry after failure",
    )

    assert task.status == TaskStatus.RETRYING
    assert task.retry_count == 1
    assert task.metadata["retry_reason"] == "retry after failure"

    task.prepare_next_attempt()

    assert task.status == TaskStatus.QUEUED
    assert task.result is None
    assert task.error is None
    assert task.retry_count == 1


def test_task_retry_error_when_not_allowed() -> None:
    task = create_task_record(
        "No retry task",
        max_retries=0,
    )

    task.mark_failed("failed once")

    with pytest.raises(TaskStateError):
        task.mark_retrying()


def test_task_cancelled_and_aborted() -> None:
    cancelled = create_task_record("cancel me")
    cancelled.mark_cancelled("user cancelled")

    assert cancelled.status == TaskStatus.CANCELLED
    assert cancelled.is_terminal is True
    assert cancelled.cancelled_at is not None
    assert cancelled.result is not None
    assert cancelled.result.error == "user cancelled"

    aborted = create_task_record("abort me")
    aborted.mark_aborted("system abort")

    assert aborted.status == TaskStatus.ABORTED
    assert aborted.is_terminal is True
    assert aborted.result is not None
    assert aborted.result.error_type == "Aborted"


def test_task_record_to_dict_roundtrip() -> None:
    task = create_task_record(
        "Planner task",
        task_type="subagent",
        parent_id="root_task",
        agent_id="planner",
        payload={
            "task": "plan",
        },
        metadata={
            "nested": {
                "ok": True,
            }
        },
        max_retries=1,
        timeout_seconds=30,
        created_by="main_agent",
        task_id="task_fixed",
    )

    task.mark_running()
    task.mark_succeeded(
        {
            "plan": [
                "step 1",
                "step 2",
            ]
        }
    )

    data = task.to_dict()

    # 确保可以 JSON 序列化，方便后续 SQLite 持久化
    json.dumps(data, ensure_ascii=False)

    restored = TaskRecord.from_dict(data)

    assert restored.id == "task_fixed"
    assert restored.name == "Planner task"
    assert restored.task_type == TaskType.SUBAGENT
    assert restored.status == TaskStatus.SUCCEEDED
    assert restored.parent_id == "root_task"
    assert restored.agent_id == "planner"
    assert restored.created_by == "main_agent"
    assert restored.result is not None
    assert restored.result.success is True
    assert restored.result.value == {
        "plan": [
            "step 1",
            "step 2",
        ]
    }


def test_task_event_to_dict_roundtrip() -> None:
    event = TaskEvent(
        task_id="task_1",
        event_type=TaskEventType.STARTED,
        status=TaskStatus.RUNNING,
        message="task started",
        metadata={
            "agent_id": "debugger",
        },
    )

    data = event.to_dict()

    json.dumps(data, ensure_ascii=False)

    restored = TaskEvent.from_dict(data)

    assert restored.task_id == "task_1"
    assert restored.event_type == TaskEventType.STARTED
    assert restored.status == TaskStatus.RUNNING
    assert restored.message == "task started"
    assert restored.metadata["agent_id"] == "debugger"


def test_task_failed_from_exception() -> None:
    task = create_task_record("exception task")

    try:
        raise ValueError("bad value")
    except ValueError as exc:
        task.mark_failed_from_exception(exc)

    assert task.status == TaskStatus.FAILED
    assert task.result is not None
    assert task.result.error == "bad value"
    assert task.result.error_type == "ValueError"
    assert task.result.traceback is not None
    assert "ValueError" in task.result.traceback