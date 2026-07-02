from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TeamMemberStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REMOVED = "removed"
    UNKNOWN = "unknown"


class TeamTaskDisplayStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


ACTIVE_TASK_STATUSES: set[str] = {
    TeamTaskDisplayStatus.ASSIGNED.value,
    TeamTaskDisplayStatus.DISPATCHED.value,
    TeamTaskDisplayStatus.RUNNING.value,
}


TERMINAL_TASK_STATUSES: set[str] = {
    TeamTaskDisplayStatus.SUCCEEDED.value,
    TeamTaskDisplayStatus.FAILED.value,
    TeamTaskDisplayStatus.CANCELLED.value,
}


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)

    if raw is None:
        return ""

    return str(raw)


def normalize_member_status(value: Any) -> str:
    text = enum_value(value).strip().lower()

    aliases = {
        "enabled": "active",
        "idle": "active",
        "running": "active",
        "busy": "active",
        "inactive": "disabled",
        "disable": "disabled",
        "stopped": "removed",
    }

    text = aliases.get(text, text)

    if text in {item.value for item in TeamMemberStatus}:
        return text

    return TeamMemberStatus.UNKNOWN.value


def normalize_task_status(value: Any) -> str:
    text = enum_value(value).strip().lower()

    aliases = {
        "todo": "pending",
        "queued": "pending",
        "started": "running",
        "in_progress": "running",
        "done": "succeeded",
        "success": "succeeded",
        "completed": "succeeded",
        "error": "failed",
        "fail": "failed",
        "canceled": "cancelled",
    }

    text = aliases.get(text, text)

    if text in {item.value for item in TeamTaskDisplayStatus}:
        return text

    return TeamTaskDisplayStatus.UNKNOWN.value


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
class TeamMemberRow:
    teammate_id: str
    name: str
    role: str = ""
    agent_name: str = ""
    status: str = TeamMemberStatus.UNKNOWN.value
    current_run_id: str | None = None
    current_task_record_id: str | None = None
    is_busy: bool = False
    is_stopped: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == TeamMemberStatus.ACTIVE.value and not self.is_stopped

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "name": self.name,
            "role": self.role,
            "agent_name": self.agent_name,
            "status": self.status,
            "current_run_id": self.current_run_id,
            "current_task_record_id": self.current_task_record_id,
            "is_busy": self.is_busy,
            "is_stopped": self.is_stopped,
            "is_active": self.is_active,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class TeamTaskRow:
    task_id: str
    title: str
    role: str = ""
    assigned_to: str = ""
    status: str = TeamTaskDisplayStatus.UNKNOWN.value
    priority: str = "normal"
    error: str | None = None
    created_at: float | None = None
    updated_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status in ACTIVE_TASK_STATUSES

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_TASK_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "role": self.role,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "priority": self.priority,
            "error": self.error,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_active": self.is_active,
            "is_terminal": self.is_terminal,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class TeamMailboxStats:
    total: int = 0
    unread: int = 0
    read: int = 0
    acked: int = 0
    archived: int = 0
    deleted: int = 0
    task_messages: int = 0
    result_messages: int = 0
    error_messages: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "unread": self.unread,
            "read": self.read,
            "acked": self.acked,
            "archived": self.archived,
            "deleted": self.deleted,
            "task_messages": self.task_messages,
            "result_messages": self.result_messages,
            "error_messages": self.error_messages,
        }


@dataclass(slots=True)
class TeamViewStats:
    members_total: int = 0
    members_active: int = 0
    members_busy: int = 0
    members_stopped: int = 0

    tasks_total: int = 0
    tasks_active: int = 0
    tasks_pending: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0

    mailbox: TeamMailboxStats = field(default_factory=TeamMailboxStats)

    def to_dict(self) -> dict[str, Any]:
        return {
            "members_total": self.members_total,
            "members_active": self.members_active,
            "members_busy": self.members_busy,
            "members_stopped": self.members_stopped,
            "tasks_total": self.tasks_total,
            "tasks_active": self.tasks_active,
            "tasks_pending": self.tasks_pending,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "tasks_cancelled": self.tasks_cancelled,
            "mailbox": self.mailbox.to_dict(),
        }


@dataclass(slots=True)
class TeamViewSnapshot:
    team_id: str = ""
    name: str = ""
    description: str = ""

    members: list[TeamMemberRow] = field(default_factory=list)
    tasks: list[TeamTaskRow] = field(default_factory=list)
    stats: TeamViewStats = field(default_factory=TeamViewStats)

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "description": self.description,
            "members": [
                member.to_dict()
                for member in self.members
            ],
            "tasks": [
                task.to_dict()
                for task in self.tasks
            ],
            "stats": self.stats.to_dict(),
            "metadata": safe_jsonable(self.metadata),
        }