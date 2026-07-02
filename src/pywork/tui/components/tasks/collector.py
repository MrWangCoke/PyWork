from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from pywork.tui.components.tasks.models import (
    ACTIVE_STATUSES,
    TaskProgressRow,
    TaskProgressSnapshot,
    TaskProgressStats,
    datetime_to_timestamp,
    enum_value,
    normalize_status,
    now_timestamp,
    safe_jsonable,
)


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def get_attr(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default

    if isinstance(value, Mapping):
        return value.get(name, default)

    return getattr(value, name, default)


def get_task_id(task: Any) -> str:
    value = (
        get_attr(task, "id")
        or get_attr(task, "task_id")
        or get_attr(task, "record_id")
        or ""
    )

    return str(value)


def get_task_name(task: Any) -> str:
    value = (
        get_attr(task, "name")
        or get_attr(task, "title")
        or get_attr(task, "task")
        or get_attr(task, "description")
        or get_task_id(task)
        or "Task"
    )

    return str(value)


def get_task_agent(task: Any) -> str:
    value = (
        get_attr(task, "agent_id")
        or get_attr(task, "agent_name")
        or get_attr(task, "assigned_to")
        or get_attr(task, "worker_id")
        or ""
    )

    return str(value or "")


def get_task_error(task: Any) -> str | None:
    value = get_attr(task, "error")

    if value is None:
        result = get_attr(task, "result")

        if result is not None:
            value = get_attr(result, "error")

    if value is None:
        return None

    text = str(value)

    return text or None


def calculate_duration_ms(task: Any) -> int | None:
    direct = get_attr(task, "duration_ms")

    if isinstance(direct, int | float):
        return max(0, int(direct))

    started_at = datetime_to_timestamp(get_attr(task, "started_at"))
    finished_at = datetime_to_timestamp(get_attr(task, "finished_at"))

    if started_at is None:
        created_at = datetime_to_timestamp(get_attr(task, "created_at"))

        if created_at is not None:
            started_at = created_at

    if finished_at is None:
        status = normalize_status(get_attr(task, "status"))

        if status in ACTIVE_STATUSES:
            finished_at = now_timestamp()
        else:
            updated_at = datetime_to_timestamp(get_attr(task, "updated_at"))

            if updated_at is not None:
                finished_at = updated_at

    if started_at is None or finished_at is None:
        return None

    return max(0, int((finished_at - started_at) * 1000))


def task_record_to_row(
    task: Any,
    *,
    active_task_ids: set[str] | None = None,
) -> TaskProgressRow:
    task_id = get_task_id(task)
    status = normalize_status(get_attr(task, "status"))
    active_task_ids = active_task_ids or set()

    is_active = (
        task_id in active_task_ids
        or bool(get_attr(task, "is_active", False))
        or status in ACTIVE_STATUSES
    )

    metadata = get_attr(task, "metadata", {})

    if not isinstance(metadata, Mapping):
        metadata = {}

    return TaskProgressRow(
        task_id=task_id,
        name=get_task_name(task),
        agent=get_task_agent(task),
        status=status,
        duration_ms=calculate_duration_ms(task),
        created_at=datetime_to_timestamp(get_attr(task, "created_at")),
        started_at=datetime_to_timestamp(get_attr(task, "started_at")),
        finished_at=datetime_to_timestamp(get_attr(task, "finished_at")),
        updated_at=datetime_to_timestamp(get_attr(task, "updated_at")),
        is_active=is_active,
        error=get_task_error(task),
        metadata=safe_jsonable(dict(metadata)),
    )


def collect_stats(rows: Sequence[TaskProgressRow]) -> TaskProgressStats:
    stats = TaskProgressStats(
        total=len(rows),
    )

    for row in rows:
        if row.is_active:
            stats.active += 1

        if row.status == "pending":
            stats.pending += 1
        elif row.status in {"queued", "running", "retrying"}:
            stats.running += 1
        elif row.status == "succeeded":
            stats.succeeded += 1
        elif row.status == "failed":
            stats.failed += 1
        elif row.status == "cancelled":
            stats.cancelled += 1
        elif row.status == "aborted":
            stats.aborted += 1
        else:
            stats.unknown += 1

    return stats


def build_task_snapshot(
    tasks: Sequence[Any],
    *,
    active_task_ids: set[str] | None = None,
) -> TaskProgressSnapshot:
    rows = [
        task_record_to_row(
            task,
            active_task_ids=active_task_ids,
        )
        for task in tasks
    ]

    rows.sort(
        key=lambda row: (
            not row.is_active,
            row.updated_at or row.started_at or row.created_at or 0,
        ),
        reverse=False,
    )

    return TaskProgressSnapshot(
        rows=rows,
        stats=collect_stats(rows),
    )


async def collect_task_records_from_manager(
    task_manager: Any,
    *,
    limit: int | None = None,
) -> list[Any]:
    list_tasks = getattr(task_manager, "list_tasks", None)

    if callable(list_tasks):
        try:
            records = await maybe_await(
                list_tasks(
                    limit=limit,
                )
            )
        except TypeError:
            records = await maybe_await(list_tasks())

        if records is None:
            return []

        return list(records)

    records = getattr(task_manager, "_records", None)

    if isinstance(records, Mapping):
        values = list(records.values())

        if limit is not None:
            values = values[:limit]

        return values

    return []


def collect_active_task_ids(task_manager: Any) -> set[str]:
    get_active_task_ids = getattr(task_manager, "get_active_task_ids", None)

    if callable(get_active_task_ids):
        try:
            return {
                str(item)
                for item in get_active_task_ids()
            }
        except Exception:
            return set()

    backend = getattr(task_manager, "backend", None)

    if backend is not None:
        backend_get_active = getattr(backend, "get_active_task_ids", None)

        if callable(backend_get_active):
            try:
                return {
                    str(item)
                    for item in backend_get_active()
                }
            except Exception:
                return set()

    return set()


async def build_task_snapshot_from_manager(
    task_manager: Any,
    *,
    limit: int | None = None,
) -> TaskProgressSnapshot:
    records = await collect_task_records_from_manager(
        task_manager,
        limit=limit,
    )
    active_task_ids = collect_active_task_ids(task_manager)

    return build_task_snapshot(
        records,
        active_task_ids=active_task_ids,
    )