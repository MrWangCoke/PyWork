from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from pywork.tasks.task import (
    TaskEvent,
    TaskEventType,
    TaskRecord,
    TaskResult,
    TaskStatus,
    normalize_task_status,
    safe_jsonable,
)


class TaskStorageError(Exception):
    """TaskStorage 基础异常。"""


class TaskStorageNotFoundError(TaskStorageError):
    """找不到指定 Task。"""


def json_dumps(value: Any) -> str:
    return json.dumps(
        safe_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def json_loads(value: str | bytes | None, default: Any = None) -> Any:
    if value is None:
        return default

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if not value:
        return default

    return json.loads(value)


class SQLiteTaskStorage:
    """
    SQLite Task 持久化。

    负责：
    - 初始化 tasks / task_events 表
    - 保存 TaskRecord
    - 更新 TaskRecord
    - 查询 TaskRecord
    - 保存 TaskEvent
    - 查询 TaskEvent

    不负责：
    - 执行 asyncio.Task
    - 重试策略
    - SubAgent 调度
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        auto_init: bool = True,
    ) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row

        if auto_init:
            self.init_db()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> SQLiteTaskStorage:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def init_db(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    parent_id TEXT,
                    agent_id TEXT,
                    created_by TEXT,
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 0,
                    timeout_seconds REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    cancelled_at REAL,
                    metadata_json TEXT NOT NULL
                )
                """
            )

            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_status
                ON tasks(status)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_parent_id
                ON tasks(parent_id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_agent_id
                ON tasks(agent_id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_tasks_created_at
                ON tasks(created_at)
                """
            )

            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT,
                    message TEXT,
                    result_json TEXT,
                    metadata_json TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )

            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_events_task_id
                ON task_events(task_id)
                """
            )
            self._connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_events_timestamp
                ON task_events(timestamp)
                """
            )

            self._connection.commit()

    def save_task(
        self,
        record: TaskRecord,
    ) -> TaskRecord:
        """
        保存或覆盖 TaskRecord。

        用 INSERT OR REPLACE，方便 TaskManager 在 create/update 时都能安全调用。
        """

        with self._lock:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO tasks (
                    id,
                    name,
                    task_type,
                    status,
                    payload_json,
                    result_json,
                    error,
                    parent_id,
                    agent_id,
                    created_by,
                    retry_count,
                    max_retries,
                    timeout_seconds,
                    created_at,
                    updated_at,
                    started_at,
                    finished_at,
                    cancelled_at,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                self._record_to_row_values(record),
            )
            self._connection.commit()

        return record

    def update_task(
        self,
        record: TaskRecord,
    ) -> TaskRecord:
        return self.save_task(record)

    def get_task(
        self,
        task_id: str,
    ) -> TaskRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT *
                FROM tasks
                WHERE id = ?
                """,
                (task_id,),
            ).fetchone()

        if row is None:
            return None

        return self._row_to_record(row)

    def require_task(
        self,
        task_id: str,
    ) -> TaskRecord:
        record = self.get_task(task_id)

        if record is None:
            raise TaskStorageNotFoundError(f"Task not found: {task_id}")

        return record

    def list_tasks(
        self,
        *,
        status: TaskStatus | str | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TaskRecord]:
        where: list[str] = []
        params: list[Any] = []

        if status is not None:
            normalized_status = normalize_task_status(status)
            where.append("status = ?")
            params.append(normalized_status.value)

        if parent_id is not None:
            where.append("parent_id = ?")
            params.append(parent_id)

        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)

        sql = """
            SELECT *
            FROM tasks
        """

        if where:
            sql += " WHERE " + " AND ".join(where)

        sql += " ORDER BY created_at DESC"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._lock:
            rows = self._connection.execute(
                sql,
                tuple(params),
            ).fetchall()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    def list_child_tasks(
        self,
        parent_id: str,
        *,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        return self.list_tasks(
            parent_id=parent_id,
            limit=limit,
        )

    def list_agent_tasks(
        self,
        agent_id: str,
        *,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        return self.list_tasks(
            agent_id=agent_id,
            limit=limit,
        )

    def delete_task(
        self,
        task_id: str,
        *,
        delete_events: bool = False,
    ) -> bool:
        with self._lock:
            cursor = self._connection.execute(
                """
                DELETE FROM tasks
                WHERE id = ?
                """,
                (task_id,),
            )

            if delete_events:
                self._connection.execute(
                    """
                    DELETE FROM task_events
                    WHERE task_id = ?
                    """,
                    (task_id,),
                )

            self._connection.commit()

        return cursor.rowcount > 0

    def count_tasks(
        self,
        *,
        status: TaskStatus | str | None = None,
        parent_id: str | None = None,
        agent_id: str | None = None,
    ) -> int:
        where: list[str] = []
        params: list[Any] = []

        if status is not None:
            normalized_status = normalize_task_status(status)
            where.append("status = ?")
            params.append(normalized_status.value)

        if parent_id is not None:
            where.append("parent_id = ?")
            params.append(parent_id)

        if agent_id is not None:
            where.append("agent_id = ?")
            params.append(agent_id)

        sql = "SELECT COUNT(*) AS count FROM tasks"

        if where:
            sql += " WHERE " + " AND ".join(where)

        with self._lock:
            row = self._connection.execute(
                sql,
                tuple(params),
            ).fetchone()

        return int(row["count"])

    def save_event(
        self,
        event: TaskEvent,
    ) -> TaskEvent:
        with self._lock:
            self._connection.execute(
                """
                INSERT INTO task_events (
                    task_id,
                    event_type,
                    status,
                    message,
                    result_json,
                    metadata_json,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.task_id,
                    event.event_type.value,
                    event.status.value if event.status else None,
                    event.message,
                    json_dumps(event.result.to_dict()) if event.result else None,
                    json_dumps(event.metadata),
                    event.timestamp,
                ),
            )
            self._connection.commit()

        return event

    def list_events(
        self,
        *,
        task_id: str | None = None,
        event_type: TaskEventType | str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[TaskEvent]:
        where: list[str] = []
        params: list[Any] = []

        if task_id is not None:
            where.append("task_id = ?")
            params.append(task_id)

        if event_type is not None:
            event_type_value = (
                event_type.value
                if isinstance(event_type, TaskEventType)
                else str(event_type)
            )
            where.append("event_type = ?")
            params.append(event_type_value)

        sql = """
            SELECT *
            FROM task_events
        """

        if where:
            sql += " WHERE " + " AND ".join(where)

        sql += " ORDER BY timestamp ASC, event_id ASC"

        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

        with self._lock:
            rows = self._connection.execute(
                sql,
                tuple(params),
            ).fetchall()

        return [
            self._row_to_event(row)
            for row in rows
        ]

    def clear(
        self,
    ) -> None:
        with self._lock:
            self._connection.execute("DELETE FROM task_events")
            self._connection.execute("DELETE FROM tasks")
            self._connection.commit()

    def _record_to_row_values(
        self,
        record: TaskRecord,
    ) -> tuple[Any, ...]:
        return (
            record.id,
            record.name,
            record.task_type.value,
            record.status.value,
            json_dumps(record.payload),
            json_dumps(record.result.to_dict()) if record.result else None,
            record.error,
            record.parent_id,
            record.agent_id,
            record.created_by,
            record.retry_count,
            record.max_retries,
            record.timeout_seconds,
            record.created_at,
            record.updated_at,
            record.started_at,
            record.finished_at,
            record.cancelled_at,
            json_dumps(record.metadata),
        )

    def _row_to_record(
        self,
        row: sqlite3.Row,
    ) -> TaskRecord:
        data = {
            "id": row["id"],
            "name": row["name"],
            "task_type": row["task_type"],
            "status": row["status"],
            "payload": json_loads(row["payload_json"], {}),
            "result": json_loads(row["result_json"], None),
            "error": row["error"],
            "parent_id": row["parent_id"],
            "agent_id": row["agent_id"],
            "created_by": row["created_by"],
            "retry_count": row["retry_count"],
            "max_retries": row["max_retries"],
            "timeout_seconds": row["timeout_seconds"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "cancelled_at": row["cancelled_at"],
            "metadata": json_loads(row["metadata_json"], {}),
        }

        return TaskRecord.from_dict(data)

    def _row_to_event(
        self,
        row: sqlite3.Row,
    ) -> TaskEvent:
        result_data = json_loads(row["result_json"], None)

        data = {
            "task_id": row["task_id"],
            "event_type": row["event_type"],
            "status": row["status"],
            "message": row["message"],
            "result": result_data,
            "metadata": json_loads(row["metadata_json"], {}),
            "timestamp": row["timestamp"],
        }

        return TaskEvent.from_dict(data)


def create_sqlite_task_storage(
    db_path: str | Path,
    *,
    auto_init: bool = True,
) -> SQLiteTaskStorage:
    return SQLiteTaskStorage(
        db_path,
        auto_init=auto_init,
    )


__all__ = [
    "SQLiteTaskStorage",
    "TaskStorageError",
    "TaskStorageNotFoundError",
    "create_sqlite_task_storage",
    "json_dumps",
    "json_loads",
]