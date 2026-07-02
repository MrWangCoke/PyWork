from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentDisplayStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    STOPPED = "stopped"
    FAILED = "failed"
    ABORTED = "aborted"
    SUCCEEDED = "succeeded"
    UNKNOWN = "unknown"


ACTIVE_AGENT_STATUSES: set[str] = {
    AgentDisplayStatus.RUNNING.value,
    AgentDisplayStatus.WAITING.value,
}

TERMINAL_AGENT_STATUSES: set[str] = {
    AgentDisplayStatus.STOPPED.value,
    AgentDisplayStatus.FAILED.value,
    AgentDisplayStatus.ABORTED.value,
    AgentDisplayStatus.SUCCEEDED.value,
}


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)

    if raw is None:
        return ""

    return str(raw)


def normalize_agent_status(value: Any) -> str:
    text = enum_value(value).strip().lower()

    aliases = {
        "created": "idle",
        "pending": "idle",
        "queued": "waiting",
        "busy": "running",
        "active": "running",
        "in_progress": "running",
        "progress": "running",
        "started": "running",
        "complete": "succeeded",
        "completed": "succeeded",
        "success": "succeeded",
        "cancelled": "aborted",
        "canceled": "aborted",
        "cancel": "aborted",
        "error": "failed",
        "fail": "failed",
    }

    text = aliases.get(text, text)

    if text in {item.value for item in AgentDisplayStatus}:
        return text

    return AgentDisplayStatus.UNKNOWN.value


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
class AgentActivityRow:
    agent_id: str
    name: str
    role: str = ""
    status: str = AgentDisplayStatus.UNKNOWN.value

    current_task: str = ""
    current_run_id: str | None = None
    current_task_record_id: str | None = None

    started_at: float | None = None
    finished_at: float | None = None
    duration_ms: int | None = None

    is_active: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_AGENT_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "current_task": self.current_task,
            "current_run_id": self.current_run_id,
            "current_task_record_id": self.current_task_record_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "is_active": self.is_active,
            "is_terminal": self.is_terminal,
            "error": self.error,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class AgentActivityStats:
    total: int = 0
    active: int = 0
    idle: int = 0
    waiting: int = 0
    running: int = 0
    stopped: int = 0
    failed: int = 0
    aborted: int = 0
    succeeded: int = 0
    unknown: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "active": self.active,
            "idle": self.idle,
            "waiting": self.waiting,
            "running": self.running,
            "stopped": self.stopped,
            "failed": self.failed,
            "aborted": self.aborted,
            "succeeded": self.succeeded,
            "unknown": self.unknown,
        }


@dataclass(slots=True)
class AgentActivitySnapshot:
    rows: list[AgentActivityRow] = field(default_factory=list)
    stats: AgentActivityStats = field(default_factory=AgentActivityStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [
                row.to_dict()
                for row in self.rows
            ],
            "stats": self.stats.to_dict(),
        }