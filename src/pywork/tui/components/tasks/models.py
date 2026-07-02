from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskDisplayStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    UNKNOWN = "unknown"


TERMINAL_STATUSES: set[str] = {
    TaskDisplayStatus.SUCCEEDED.value,
    TaskDisplayStatus.FAILED.value,
    TaskDisplayStatus.CANCELLED.value,
    TaskDisplayStatus.ABORTED.value,
}


ACTIVE_STATUSES: set[str] = {
    TaskDisplayStatus.QUEUED.value,
    TaskDisplayStatus.RUNNING.value,
    TaskDisplayStatus.RETRYING.value,
}


def now_timestamp() -> float:
    return datetime.now().timestamp()


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)

    if raw is None:
        return ""

    return str(raw)


def normalize_status(value: Any) -> str:
    text = enum_value(value).strip().lower()

    aliases = {
        "created": "pending",
        "todo": "pending",
        "waiting": "pending",
        "in_progress": "running",
        "progress": "running",
        "started": "running",
        "complete": "succeeded",
        "completed": "succeeded",
        "success": "succeeded",
        "done": "succeeded",
        "error": "failed",
        "fail": "failed",
        "canceled": "cancelled",
        "stopped": "cancelled",
        "stop": "cancelled",
        "abort": "aborted",
    }

    text = aliases.get(text, text)

    if text in {item.value for item in TaskDisplayStatus}:
        return text

    return TaskDisplayStatus.UNKNOWN.value


def datetime_to_timestamp(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        return value.timestamp()

    if isinstance(value, int | float):
        return float(value)

    return None


def safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, dict):
        return {
            str(key): safe_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [
            safe_jsonable(item)
            for item in value
        ]

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            return safe_jsonable(to_dict())
        except Exception:
            pass

    return str(value)


@dataclass(slots=True)
class TaskProgressRow:
    task_id: str
    name: str
    agent: str = ""
    status: str = TaskDisplayStatus.UNKNOWN.value

    duration_ms: int | None = None
    created_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None
    updated_at: float | None = None

    is_active: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "name": self.name,
            "agent": self.agent,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "is_terminal": self.is_terminal,
            "error": self.error,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class TaskProgressStats:
    total: int = 0
    active: int = 0
    pending: int = 0
    running: int = 0
    succeeded: int = 0
    failed: int = 0
    cancelled: int = 0
    aborted: int = 0
    unknown: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "active": self.active,
            "pending": self.pending,
            "running": self.running,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "aborted": self.aborted,
            "unknown": self.unknown,
        }


@dataclass(slots=True)
class TaskProgressSnapshot:
    rows: list[TaskProgressRow] = field(default_factory=list)
    stats: TaskProgressStats = field(default_factory=TaskProgressStats)
    updated_at: float = field(default_factory=now_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [
                row.to_dict()
                for row in self.rows
            ],
            "stats": self.stats.to_dict(),
            "updated_at": self.updated_at,
        }