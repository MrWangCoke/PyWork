from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    SubAgentLLMCallable,
    SubAgentToolScope,
)
from pywork.subagents.manager import SubAgentManager
from pywork.teams.mailbox import (
    AgentMailbox,
    MailboxMessage,
    MailboxMessageType,
    create_agent_mailbox,
    safe_jsonable,
)
from pywork.teams.roster import (
    RosterMember,
    TeamRoster,
    create_team_roster,
)
from pywork.teams.teammate import TeammateAgent


class TeamError(Exception):
    """Team 基础异常。"""


class TeamValidationError(TeamError):
    """Team 参数校验异常。"""


class TeamTaskNotFoundError(TeamError):
    """找不到共享任务。"""


class TeamTaskAssignmentError(TeamError):
    """共享任务分配失败。"""


class TeamTaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TeamTaskStatus.SUCCEEDED,
            TeamTaskStatus.FAILED,
            TeamTaskStatus.CANCELLED,
        }

    @property
    def is_active(self) -> bool:
        return self in {
            TeamTaskStatus.ASSIGNED,
            TeamTaskStatus.DISPATCHED,
            TeamTaskStatus.RUNNING,
        }


class TeamTaskPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


def now_timestamp() -> float:
    return time.time()


def new_team_id(prefix: str = "team") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_team_task_id(prefix: str = "team_task") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_team_task_status(
    value: TeamTaskStatus | str,
) -> TeamTaskStatus:
    if isinstance(value, TeamTaskStatus):
        return value

    try:
        return TeamTaskStatus(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TeamTaskStatus)
        raise TeamValidationError(
            f"Invalid team task status {value!r}. Valid statuses: {valid}"
        ) from exc


def normalize_team_task_priority(
    value: TeamTaskPriority | str | None,
) -> TeamTaskPriority:
    if isinstance(value, TeamTaskPriority):
        return value

    try:
        return TeamTaskPriority(str(value or TeamTaskPriority.NORMAL.value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in TeamTaskPriority)
        raise TeamValidationError(
            f"Invalid team task priority {value!r}. Valid priorities: {valid}"
        ) from exc


@dataclass(slots=True)
class TeamSharedTask:
    """
    Team 共享任务。

    它不是底层 asyncio Task，也不是 SubAgent Task。
    它是 Team 层面的任务池记录，用来描述：

    - 任务是什么
    - 分给谁
    - 当前状态
    - 最终结果
    """

    task_id: str
    title: str
    description: str = ""

    role: str | None = None
    assigned_to: str | None = None
    parent_task_id: str | None = None
    created_by: str | None = None

    status: TeamTaskStatus = TeamTaskStatus.PENDING
    priority: TeamTaskPriority = TeamTaskPriority.NORMAL

    payload: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=now_timestamp)
    updated_at: float = field(default_factory=now_timestamp)
    assigned_at: float | None = None
    dispatched_at: float | None = None
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @property
    def is_active(self) -> bool:
        return self.status.is_active

    @property
    def task_text(self) -> str:
        return self.description or self.title

    def touch(self) -> None:
        self.updated_at = now_timestamp()

    def mark_assigned(
        self,
        teammate_id: str,
    ) -> None:
        self.assigned_to = teammate_id
        self.status = TeamTaskStatus.ASSIGNED
        self.assigned_at = self.assigned_at or now_timestamp()
        self.touch()

    def mark_dispatched(self) -> None:
        self.status = TeamTaskStatus.DISPATCHED
        self.dispatched_at = self.dispatched_at or now_timestamp()
        self.touch()

    def mark_running(self) -> None:
        self.status = TeamTaskStatus.RUNNING
        self.started_at = self.started_at or now_timestamp()
        self.touch()

    def mark_succeeded(
        self,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.status = TeamTaskStatus.SUCCEEDED
        self.result = result or {}
        self.error = None
        self.finished_at = now_timestamp()
        self.touch()

    def mark_failed(
        self,
        error: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> None:
        self.status = TeamTaskStatus.FAILED
        self.error = error
        self.result = result
        self.finished_at = now_timestamp()
        self.touch()

    def mark_cancelled(
        self,
        reason: str | None = None,
    ) -> None:
        self.status = TeamTaskStatus.CANCELLED
        self.error = reason or "team task cancelled"
        self.finished_at = now_timestamp()
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "role": self.role,
            "assigned_to": self.assigned_to,
            "parent_task_id": self.parent_task_id,
            "created_by": self.created_by,
            "status": self.status.value,
            "priority": self.priority.value,
            "payload": safe_jsonable(self.payload),
            "result": safe_jsonable(self.result),
            "error": self.error,
            "metadata": safe_jsonable(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "assigned_at": self.assigned_at,
            "dispatched_at": self.dispatched_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
    ) -> TeamSharedTask:
        return cls(
            task_id=str(data["task_id"]),
            title=str(data["title"]),
            description=str(data.get("description") or ""),
            role=data.get("role"),
            assigned_to=data.get("assigned_to"),
            parent_task_id=data.get("parent_task_id"),
            created_by=data.get("created_by"),
            status=normalize_team_task_status(data.get("status", TeamTaskStatus.PENDING)),
            priority=normalize_team_task_priority(data.get("priority", TeamTaskPriority.NORMAL)),
            payload=dict(data.get("payload") or {}),
            result=(
                dict(data["result"])
                if isinstance(data.get("result"), Mapping)
                else None
            ),
            error=data.get("error"),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at") or now_timestamp()),
            updated_at=float(data.get("updated_at") or now_timestamp()),
            assigned_at=data.get("assigned_at"),
            dispatched_at=data.get("dispatched_at"),
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
        )


@dataclass(slots=True, frozen=True)
class TeamConfig:
    default_assignment_strategy: str = "round_robin"
    auto_mark_result_messages_read: bool = True
    auto_ack_result_messages: bool = True


class Team:
    """
    Team 模型。

    它组合：
    - roster: 成员管理
    - mailbox: 团队邮箱
    - shared_task_list: 团队共享任务池
    """

    def __init__(
        self,
        *,
        team_id: str | None = None,
        name: str = "",
        description: str = "",
        roster: TeamRoster | None = None,
        mailbox: AgentMailbox | None = None,
        manager: SubAgentManager | None = None,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path = ".",
        config: TeamConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.team_id = team_id or new_team_id()
        self.name = name or self.team_id
        self.description = description
        self.workspace_path = Path(workspace_path)
        self.config = config or TeamConfig()
        self.metadata = metadata or {}

        self.mailbox = mailbox or (
            roster.mailbox
            if roster is not None
            else create_agent_mailbox(
                metadata={
                    "owner": "Team",
                    "team_id": self.team_id,
                }
            )
        )

        self.roster = roster or create_team_roster(
            mailbox=self.mailbox,
            manager=manager,
            llm=llm,
            tool_definitions=tool_definitions,
            workspace_path=workspace_path,
            metadata={
                "owner": "Team",
                "team_id": self.team_id,
                **self.metadata,
            },
        )

        self.shared_task_list: dict[str, TeamSharedTask] = {}

    # ------------------------------------------------------------------
    # member helpers
    # ------------------------------------------------------------------

    def create_teammate(
        self,
        *,
        teammate_id: str | None = None,
        name: str = "",
        role: str = "general",
        agent_name: str | None = None,
        description: str = "",
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> RosterMember:
        return self.roster.create_teammate(
            teammate_id=teammate_id,
            name=name,
            role=role,
            agent_name=agent_name,
            description=description,
            workspace_path=self.workspace_path,
            tool_scope=tool_scope,
            max_steps=max_steps,
            metadata=metadata,
            replace=replace,
        )

    def add_teammate(
        self,
        teammate: TeammateAgent,
        *,
        replace: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> RosterMember:
        return self.roster.add_teammate(
            teammate,
            replace=replace,
            metadata=metadata,
        )

    def get_teammate(
        self,
        teammate_id: str,
    ) -> TeammateAgent | None:
        return self.roster.get_teammate(teammate_id)

    def require_teammate(
        self,
        teammate_id: str,
    ) -> TeammateAgent:
        return self.roster.require_teammate(teammate_id)

    def list_members(
        self,
        **kwargs: Any,
    ) -> list[RosterMember]:
        return self.roster.list_members(**kwargs)

    # ------------------------------------------------------------------
    # shared task list
    # ------------------------------------------------------------------

    def create_shared_task(
        self,
        title: str,
        *,
        description: str = "",
        role: str | None = None,
        assigned_to: str | None = None,
        parent_task_id: str | None = None,
        created_by: str | None = None,
        priority: TeamTaskPriority | str | None = TeamTaskPriority.NORMAL,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> TeamSharedTask:
        title = title.strip()

        if not title:
            raise TeamValidationError("shared task title is required")

        task = TeamSharedTask(
            task_id=task_id or new_team_task_id(),
            title=title,
            description=description,
            role=role,
            assigned_to=assigned_to,
            parent_task_id=parent_task_id,
            created_by=created_by or self.team_id,
            priority=normalize_team_task_priority(priority),
            payload=dict(payload or {}),
            metadata={
                "team_id": self.team_id,
                **dict(metadata or {}),
            },
        )

        if task.task_id in self.shared_task_list:
            raise TeamValidationError(f"shared task already exists: {task.task_id}")

        if assigned_to:
            task.mark_assigned(assigned_to)

        self.shared_task_list[task.task_id] = task

        return task

    def add_shared_task(
        self,
        task: TeamSharedTask,
        *,
        replace: bool = False,
    ) -> TeamSharedTask:
        if task.task_id in self.shared_task_list and not replace:
            raise TeamValidationError(f"shared task already exists: {task.task_id}")

        self.shared_task_list[task.task_id] = task

        return task

    def get_shared_task(
        self,
        task_id: str,
    ) -> TeamSharedTask | None:
        return self.shared_task_list.get(task_id)

    def require_shared_task(
        self,
        task_id: str,
    ) -> TeamSharedTask:
        task = self.get_shared_task(task_id)

        if task is None:
            raise TeamTaskNotFoundError(f"shared task not found: {task_id}")

        return task

    def list_shared_tasks(
        self,
        *,
        status: TeamTaskStatus | str | None = None,
        assigned_to: str | None = None,
        role: str | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> list[TeamSharedTask]:
        tasks = list(self.shared_task_list.values())

        if status is not None:
            normalized_status = normalize_team_task_status(status)
            tasks = [
                task
                for task in tasks
                if task.status == normalized_status
            ]

        if assigned_to is not None:
            tasks = [
                task
                for task in tasks
                if task.assigned_to == assigned_to
            ]

        if role is not None:
            tasks = [
                task
                for task in tasks
                if task.role == role
            ]

        if not include_terminal:
            tasks = [
                task
                for task in tasks
                if not task.is_terminal
            ]

        priority_order = {
            TeamTaskPriority.URGENT: 0,
            TeamTaskPriority.HIGH: 1,
            TeamTaskPriority.NORMAL: 2,
            TeamTaskPriority.LOW: 3,
        }

        tasks.sort(
            key=lambda task: (
                priority_order.get(task.priority, 99),
                task.created_at,
            )
        )

        if limit is not None:
            tasks = tasks[:limit]

        return tasks

    def remove_shared_task(
        self,
        task_id: str,
    ) -> TeamSharedTask:
        task = self.shared_task_list.pop(task_id, None)

        if task is None:
            raise TeamTaskNotFoundError(f"shared task not found: {task_id}")

        return task

    def clear_shared_tasks(
        self,
        *,
        include_active: bool = False,
    ) -> int:
        if include_active:
            count = len(self.shared_task_list)
            self.shared_task_list.clear()
            return count

        removable = [
            task_id
            for task_id, task in self.shared_task_list.items()
            if task.is_terminal
        ]

        for task_id in removable:
            self.shared_task_list.pop(task_id, None)

        return len(removable)

    # ------------------------------------------------------------------
    # assign / dispatch
    # ------------------------------------------------------------------

    def assign_shared_task(
        self,
        task_id: str,
        *,
        teammate_id: str | None = None,
        role: str | None = None,
        strategy: str | None = None,
    ) -> TeamSharedTask:
        task = self.require_shared_task(task_id)

        if teammate_id is None:
            selected = self.roster.select_member(
                role=role or task.role,
                strategy=strategy or self.config.default_assignment_strategy,
            )

            if selected is None:
                raise TeamTaskAssignmentError(
                    f"No available teammate for task {task_id!r}"
                )

            teammate_id = selected.teammate_id
        else:
            member = self.roster.require_member(teammate_id)

            if not member.is_available:
                raise TeamTaskAssignmentError(
                    f"Teammate is not available: {teammate_id}"
                )

        task.mark_assigned(teammate_id)

        if role is not None:
            task.role = role

        return task

    async def dispatch_shared_task(
        self,
        task_id: str,
        *,
        teammate_id: str | None = None,
        role: str | None = None,
        strategy: str | None = None,
        subject: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MailboxMessage:
        task = self.require_shared_task(task_id)

        if teammate_id is not None or task.assigned_to is None:
            task = self.assign_shared_task(
                task_id,
                teammate_id=teammate_id,
                role=role,
                strategy=strategy,
            )

        if task.assigned_to is None:
            raise TeamTaskAssignmentError(f"Task is not assigned: {task_id}")

        message = await self.mailbox.send_message(
            sender_id=self.team_id,
            recipient_id=task.assigned_to,
            subject=subject or task.title,
            content=task.task_text,
            message_type=MailboxMessageType.TASK,
            task_id=task.task_id,
            payload={
                "task": task.task_text,
                "shared_task": task.to_dict(),
            },
            metadata={
                "team_id": self.team_id,
                "team_task_id": task.task_id,
                **dict(metadata or {}),
            },
        )

        task.mark_dispatched()

        return message

    async def dispatch_next_task(
        self,
        *,
        role: str | None = None,
        strategy: str | None = None,
    ) -> MailboxMessage | None:
        candidates = self.list_shared_tasks(
            status=TeamTaskStatus.PENDING,
            role=role,
            include_terminal=False,
            limit=1,
        )

        if not candidates:
            return None

        return await self.dispatch_shared_task(
            candidates[0].task_id,
            role=role,
            strategy=strategy,
        )

    async def broadcast_message(
        self,
        *,
        content: str,
        subject: str = "",
        role: str | None = None,
        message_type: MailboxMessageType | str = MailboxMessageType.NOTE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[MailboxMessage]:
        members = self.roster.list_members(
            role=role,
            include_busy=True,
            include_stopped=False,
        )

        recipient_ids = [
            member.teammate_id
            for member in members
            if member.is_active
        ]

        if not recipient_ids:
            return []

        return await self.mailbox.broadcast_message(
            sender_id=self.team_id,
            recipient_ids=recipient_ids,
            subject=subject,
            content=content,
            message_type=message_type,
            payload=payload,
            metadata={
                "team_id": self.team_id,
                **dict(metadata or {}),
            },
        )

    async def send_message_to_member(
        self,
        teammate_id: str,
        *,
        content: str,
        subject: str = "",
        message_type: MailboxMessageType | str = MailboxMessageType.NOTE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MailboxMessage:
        self.roster.require_member(teammate_id)

        return await self.mailbox.send_message(
            sender_id=self.team_id,
            recipient_id=teammate_id,
            subject=subject,
            content=content,
            message_type=message_type,
            payload=payload,
            metadata={
                "team_id": self.team_id,
                **dict(metadata or {}),
            },
        )

    # ------------------------------------------------------------------
    # result collection
    # ------------------------------------------------------------------

    def mark_task_running(
        self,
        task_id: str,
    ) -> TeamSharedTask:
        task = self.require_shared_task(task_id)
        task.mark_running()
        return task

    def mark_task_succeeded(
        self,
        task_id: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> TeamSharedTask:
        task = self.require_shared_task(task_id)
        task.mark_succeeded(result)
        return task

    def mark_task_failed(
        self,
        task_id: str,
        error: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> TeamSharedTask:
        task = self.require_shared_task(task_id)
        task.mark_failed(error, result=result)
        return task

    def mark_task_cancelled(
        self,
        task_id: str,
        *,
        reason: str | None = None,
    ) -> TeamSharedTask:
        task = self.require_shared_task(task_id)
        task.mark_cancelled(reason)
        return task

    async def collect_result_messages(
        self,
        *,
        limit: int | None = None,
        timeout: float | None = None,
    ) -> list[MailboxMessage]:
        poll_result = await self.mailbox.poll_messages(
            self.team_id,
            limit=limit,
            include_read=False,
            timeout=timeout,
        )

        collected: list[MailboxMessage] = []

        for message in poll_result.messages:
            if message.message_type not in {
                MailboxMessageType.RESULT,
                MailboxMessageType.ERROR,
                MailboxMessageType.RESPONSE,
            }:
                continue

            self.apply_result_message(message)
            collected.append(message)

            if self.config.auto_mark_result_messages_read:
                await self.mailbox.mark_read(
                    message.message_id,
                    agent_id=self.team_id,
                )

            if self.config.auto_ack_result_messages:
                await self.mailbox.acknowledge_message(
                    message.message_id,
                    agent_id=self.team_id,
                )

        return collected

    def apply_result_message(
        self,
        message: MailboxMessage,
    ) -> TeamSharedTask | None:
        task_id = message.task_id

        if not task_id:
            task_result = message.payload.get("task_result")

            if isinstance(task_result, Mapping):
                task_id = (
                    task_result.get("task_record_id")
                    or task_result.get("task_id")
                )

        if not task_id or task_id not in self.shared_task_list:
            return None

        task = self.require_shared_task(task_id)

        if task.status == TeamTaskStatus.CANCELLED:
            return task

        payload_result = message.payload.get("task_result")
        result_dict = (
            dict(payload_result)
            if isinstance(payload_result, Mapping)
            else {
                "message": message.to_dict(),
            }
        )

        success = bool(result_dict.get("success", message.message_type == MailboxMessageType.RESULT))

        if success:
            task.mark_succeeded(result_dict)
        else:
            error = (
                str(result_dict.get("error"))
                if result_dict.get("error") is not None
                else message.content
                or "team task failed"
            )
            task.mark_failed(
                error,
                result=result_dict,
            )

        return task

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    async def stop_all_teammates(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        return await self.roster.stop_all(
            reason=reason or f"team {self.team_id} stopped"
        )

    async def cancel_all_current(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        return await self.roster.cancel_all_current(
            reason=reason or f"team {self.team_id} cancelled current tasks"
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "team_id": self.team_id,
            "name": self.name,
            "description": self.description,
            "workspace_path": str(self.workspace_path),
            "roster": self.roster.to_dict(),
            "shared_task_list": [
                task.to_dict()
                for task in self.list_shared_tasks()
            ],
            "mailbox": {
                "message_count": self.mailbox.count_messages(),
            },
            "config": {
                "default_assignment_strategy": self.config.default_assignment_strategy,
                "auto_mark_result_messages_read": self.config.auto_mark_result_messages_read,
                "auto_ack_result_messages": self.config.auto_ack_result_messages,
            },
            "metadata": safe_jsonable(self.metadata),
        }


def create_team(
    *,
    team_id: str | None = None,
    name: str = "",
    description: str = "",
    roster: TeamRoster | None = None,
    mailbox: AgentMailbox | None = None,
    manager: SubAgentManager | None = None,
    llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    workspace_path: str | Path = ".",
    config: TeamConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> Team:
    return Team(
        team_id=team_id,
        name=name,
        description=description,
        roster=roster,
        mailbox=mailbox,
        manager=manager,
        llm=llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        config=config,
        metadata=metadata,
    )


__all__ = [
    "Team",
    "TeamConfig",
    "TeamError",
    "TeamSharedTask",
    "TeamTaskAssignmentError",
    "TeamTaskNotFoundError",
    "TeamTaskPriority",
    "TeamTaskStatus",
    "TeamValidationError",
    "create_team",
    "new_team_id",
    "new_team_task_id",
    "normalize_team_task_priority",
    "normalize_team_task_status",
]