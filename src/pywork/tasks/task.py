from __future__ import annotations

import time
import traceback as traceback_module
import uuid
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from typing import Any


class TaskModelError(Exception):
    """Task 数据模型基础异常。"""


class TaskStateError(TaskModelError):
    """Task 状态异常。"""


class TaskStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
            TaskStatus.ABORTED,
        }

    @property
    def is_active(self) -> bool:
        return self in {
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.RETRYING,
        }


class TaskType(str, Enum):
    GENERIC = "generic"
    SUBAGENT = "subagent"
    TOOL = "tool"
    RUNTIME = "runtime"
    USER = "user"


class TaskEventType(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    STARTED = "started"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    UPDATED = "updated"


def now_timestamp() -> float:
    return time.time()


def new_task_id(prefix: str = "task") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_task_status(value: TaskStatus | str) -> TaskStatus:
    if isinstance(value, TaskStatus):
        return value

    try:
        return TaskStatus(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TaskStatus)
        raise TaskStateError(
            f"Invalid task status {value!r}. Valid statuses: {valid}"
        ) from exc


def normalize_task_type(value: TaskType | str) -> TaskType:
    if isinstance(value, TaskType):
        return value

    try:
        return TaskType(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TaskType)
        raise TaskModelError(
            f"Invalid task type {value!r}. Valid task types: {valid}"
        ) from exc


def is_terminal_status(value: TaskStatus | str) -> bool:
    return normalize_task_status(value).is_terminal


def safe_jsonable(value: Any) -> Any:
    """
    把常见对象转成 JSON 友好的结构。

    SQLite 持久化时会用到这个函数，避免 Path、Enum、dataclass
    之类的对象直接塞进 JSON 出问题。
    """

    if value is None:
        return None

    if isinstance(value, str | int | float | bool):
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

    if is_dataclass(value):
        return safe_jsonable(asdict(value))

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return safe_jsonable(value.to_dict())

    return str(value)


@dataclass(slots=True)
class TaskResult:
    """
    Task 执行结果。

    success:
        任务是否成功。

    value:
        成功结果。可以是字符串、dict、列表等。

    error / error_type / traceback:
        失败信息。

    metadata:
        额外信息，例如 exit_code、stdout 摘要、SubAgent 名称等。
    """

    success: bool
    value: Any = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=now_timestamp)

    @classmethod
    def success_result(
        cls,
        value: Any = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        return cls(
            success=True,
            value=value,
            metadata=metadata or {},
        )

    @classmethod
    def failure_result(
        cls,
        error: str,
        *,
        error_type: str | None = None,
        traceback: str | None = None,
        value: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        return cls(
            success=False,
            value=value,
            error=error,
            error_type=error_type,
            traceback=traceback,
            metadata=metadata or {},
        )

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        include_traceback: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        return cls.failure_result(
            str(exc),
            error_type=type(exc).__name__,
            traceback=(
                "".join(
                    traceback_module.format_exception(
                        type(exc),
                        exc,
                        exc.__traceback__,
                    )
                )
                if include_traceback
                else None
            ),
            metadata=metadata or {},
        )

    @classmethod
    def cancelled_result(
        cls,
        reason: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TaskResult:
        return cls.failure_result(
            reason or "task cancelled",
            error_type="Cancelled",
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "value": safe_jsonable(self.value),
            "error": self.error,
            "error_type": self.error_type,
            "traceback": self.traceback,
            "metadata": safe_jsonable(self.metadata),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskResult:
        return cls(
            success=bool(data.get("success", False)),
            value=data.get("value"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            traceback=data.get("traceback"),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at") or now_timestamp()),
        )


@dataclass(slots=True)
class TaskSpec:
    """
    Task 创建请求。

    这是“我要创建一个什么任务”的描述。
    还不是运行中的 Task。
    """

    name: str
    task_type: TaskType = TaskType.GENERIC
    payload: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    max_retries: int = 0
    timeout_seconds: float | None = None
    created_by: str | None = None

    def to_record(
        self,
        *,
        task_id: str | None = None,
    ) -> TaskRecord:
        return TaskRecord(
            id=task_id or new_task_id(),
            name=self.name,
            task_type=self.task_type,
            status=TaskStatus.PENDING,
            payload=dict(self.payload),
            parent_id=self.parent_id,
            agent_id=self.agent_id,
            metadata=dict(self.metadata),
            max_retries=self.max_retries,
            timeout_seconds=self.timeout_seconds,
            created_by=self.created_by,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_type": self.task_type.value,
            "payload": safe_jsonable(self.payload),
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "metadata": safe_jsonable(self.metadata),
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskSpec:
        return cls(
            name=str(data["name"]),
            task_type=normalize_task_type(data.get("task_type", TaskType.GENERIC)),
            payload=dict(data.get("payload") or {}),
            parent_id=data.get("parent_id"),
            agent_id=data.get("agent_id"),
            metadata=dict(data.get("metadata") or {}),
            max_retries=int(data.get("max_retries") or 0),
            timeout_seconds=data.get("timeout_seconds"),
            created_by=data.get("created_by"),
        )


@dataclass(slots=True)
class TaskRecord:
    """
    Task 当前状态记录。

    核心字段：
    - id
    - status
    - result
    - parent_id
    - agent_id

    后面 task_storage.py 会直接持久化这个对象。
    """

    id: str
    name: str
    task_type: TaskType = TaskType.GENERIC
    status: TaskStatus = TaskStatus.PENDING
    payload: dict[str, Any] = field(default_factory=dict)

    result: TaskResult | None = None
    error: str | None = None

    parent_id: str | None = None
    agent_id: str | None = None
    created_by: str | None = None

    retry_count: int = 0
    max_retries: int = 0
    timeout_seconds: float | None = None

    created_at: float = field(default_factory=now_timestamp)
    updated_at: float = field(default_factory=now_timestamp)
    started_at: float | None = None
    finished_at: float | None = None
    cancelled_at: float | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def duration_ms(self) -> int | None:
        if self.started_at is None:
            return None

        end = self.finished_at or now_timestamp()

        return int((end - self.started_at) * 1000)

    @property
    def can_retry(self) -> bool:
        return (
            self.status == TaskStatus.FAILED
            and self.retry_count < self.max_retries
        )

    def touch(self) -> None:
        self.updated_at = now_timestamp()

    def set_status(self, status: TaskStatus | str) -> None:
        self.status = normalize_task_status(status)
        self.touch()

    def mark_queued(self) -> None:
        self.status = TaskStatus.QUEUED
        self.touch()

    def mark_running(
        self,
        *,
        agent_id: str | None = None,
    ) -> None:
        self.status = TaskStatus.RUNNING
        self.started_at = self.started_at or now_timestamp()
        self.finished_at = None
        self.cancelled_at = None

        if agent_id is not None:
            self.agent_id = agent_id

        self.touch()

    def mark_succeeded(
        self,
        value: Any = None,
        *,
        result: TaskResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = TaskStatus.SUCCEEDED
        self.result = result or TaskResult.success_result(
            value,
            metadata=metadata or {},
        )
        self.error = None
        self.finished_at = now_timestamp()
        self.touch()

    def mark_failed(
        self,
        error: str,
        *,
        error_type: str | None = None,
        result: TaskResult | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = TaskStatus.FAILED
        self.error = error
        self.result = result or TaskResult.failure_result(
            error,
            error_type=error_type,
            metadata=metadata or {},
        )
        self.finished_at = now_timestamp()
        self.touch()

    def mark_failed_from_exception(
        self,
        exc: BaseException,
        *,
        include_traceback: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        result = TaskResult.from_exception(
            exc,
            include_traceback=include_traceback,
            metadata=metadata or {},
        )

        self.mark_failed(
            result.error or str(exc),
            error_type=result.error_type,
            result=result,
            metadata=metadata or {},
        )

    def mark_cancelled(
        self,
        reason: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = TaskStatus.CANCELLED
        self.error = reason or "task cancelled"
        self.result = TaskResult.cancelled_result(
            reason,
            metadata=metadata or {},
        )
        self.cancelled_at = now_timestamp()
        self.finished_at = self.cancelled_at
        self.touch()

    def mark_aborted(
        self,
        reason: str | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.status = TaskStatus.ABORTED
        self.error = reason or "task aborted"
        self.result = TaskResult.failure_result(
            self.error,
            error_type="Aborted",
            metadata=metadata or {},
        )
        self.cancelled_at = now_timestamp()
        self.finished_at = self.cancelled_at
        self.touch()

    def mark_retrying(
        self,
        *,
        reason: str | None = None,
    ) -> None:
        if not self.can_retry:
            raise TaskStateError(
                f"Task {self.id} cannot retry: "
                f"status={self.status.value}, "
                f"retry_count={self.retry_count}, "
                f"max_retries={self.max_retries}"
            )

        self.retry_count += 1
        self.status = TaskStatus.RETRYING
        self.metadata["retry_reason"] = reason or self.error
        self.touch()

    def prepare_next_attempt(self) -> None:
        """
        重试前调用。

        保留 retry_count，但清理本次执行状态。
        """

        self.status = TaskStatus.QUEUED
        self.result = None
        self.error = None
        self.started_at = None
        self.finished_at = None
        self.cancelled_at = None
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "payload": safe_jsonable(self.payload),
            "result": self.result.to_dict() if self.result else None,
            "error": self.error,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "created_by": self.created_by,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cancelled_at": self.cancelled_at,
            "metadata": safe_jsonable(self.metadata),
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
            "duration_ms": self.duration_ms,
            "can_retry": self.can_retry,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskRecord:
        result_data = data.get("result")

        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            task_type=normalize_task_type(data.get("task_type", TaskType.GENERIC)),
            status=normalize_task_status(data.get("status", TaskStatus.PENDING)),
            payload=dict(data.get("payload") or {}),
            result=(
                TaskResult.from_dict(result_data)
                if isinstance(result_data, dict)
                else None
            ),
            error=data.get("error"),
            parent_id=data.get("parent_id"),
            agent_id=data.get("agent_id"),
            created_by=data.get("created_by"),
            retry_count=int(data.get("retry_count") or 0),
            max_retries=int(data.get("max_retries") or 0),
            timeout_seconds=data.get("timeout_seconds"),
            created_at=float(data.get("created_at") or now_timestamp()),
            updated_at=float(data.get("updated_at") or now_timestamp()),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            cancelled_at=data.get("cancelled_at"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class TaskEvent:
    task_id: str
    event_type: TaskEventType
    status: TaskStatus | None = None
    message: str | None = None
    result: TaskResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "event_type": self.event_type.value,
            "status": self.status.value if self.status else None,
            "message": self.message,
            "result": self.result.to_dict() if self.result else None,
            "metadata": safe_jsonable(self.metadata),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEvent:
        result_data = data.get("result")
        status = data.get("status")

        return cls(
            task_id=str(data["task_id"]),
            event_type=TaskEventType(str(data["event_type"])),
            status=normalize_task_status(status) if status else None,
            message=data.get("message"),
            result=(
                TaskResult.from_dict(result_data)
                if isinstance(result_data, dict)
                else None
            ),
            metadata=dict(data.get("metadata") or {}),
            timestamp=float(data.get("timestamp") or now_timestamp()),
        )


def create_task_record(
    name: str,
    *,
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
    spec = TaskSpec(
        name=name,
        task_type=normalize_task_type(task_type),
        payload=payload or {},
        parent_id=parent_id,
        agent_id=agent_id,
        metadata=metadata or {},
        max_retries=max_retries,
        timeout_seconds=timeout_seconds,
        created_by=created_by,
    )

    return spec.to_record(
        task_id=task_id,
    )


__all__ = [
    "TaskEvent",
    "TaskEventType",
    "TaskModelError",
    "TaskRecord",
    "TaskResult",
    "TaskSpec",
    "TaskStateError",
    "TaskStatus",
    "TaskType",
    "create_task_record",
    "is_terminal_status",
    "new_task_id",
    "normalize_task_status",
    "normalize_task_type",
    "safe_jsonable",
]