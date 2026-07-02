from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    SubAgentLLMCallable,
    SubAgentToolScope,
)
from pywork.subagents.manager import SubAgentManager
from pywork.teams.mailbox import AgentMailbox, create_agent_mailbox, safe_jsonable
from pywork.teams.teammate import (
    TeammateAgent,
    TeammateSpec,
    TeammateStatus,
    create_teammate,
    normalize_teammate_role,
)


class RosterError(Exception):
    """Roster 基础异常。"""


class RosterValidationError(RosterError):
    """Roster 参数校验异常。"""


class RosterMemberNotFoundError(RosterError):
    """找不到成员。"""


class RosterMemberAlreadyExistsError(RosterError):
    """成员已经存在。"""


class RosterMemberDisabledError(RosterError):
    """成员已禁用。"""


class RosterMemberStatus(str, Enum):
    ACTIVE = "active"
    DISABLED = "disabled"
    REMOVED = "removed"


def now_timestamp() -> float:
    return time.time()


def normalize_member_status(
    value: RosterMemberStatus | str,
) -> RosterMemberStatus:
    if isinstance(value, RosterMemberStatus):
        return value

    try:
        return RosterMemberStatus(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in RosterMemberStatus)
        raise RosterValidationError(
            f"Invalid roster member status {value!r}. Valid statuses: {valid}"
        ) from exc


@dataclass(slots=True)
class RosterMember:
    """
    Roster 里的成员记录。

    teammate:
        真正的 TeammateAgent 实例。

    status:
        roster 层面的启用 / 禁用状态。
        注意这和 TeammateAgent.status 不同：
        - RosterMemberStatus 表示这个成员是否可被团队调度。
        - TeammateStatus 表示 teammate 自己当前是否 idle/running/stopped。
    """

    teammate: TeammateAgent
    status: RosterMemberStatus = RosterMemberStatus.ACTIVE
    joined_at: float = field(default_factory=now_timestamp)
    updated_at: float = field(default_factory=now_timestamp)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def teammate_id(self) -> str:
        return self.teammate.teammate_id

    @property
    def role(self) -> str:
        return self.teammate.role

    @property
    def agent_name(self) -> str:
        return self.teammate.agent_name

    @property
    def name(self) -> str:
        return self.teammate.name

    @property
    def mailbox(self) -> AgentMailbox:
        return self.teammate.mailbox

    @property
    def is_active(self) -> bool:
        return self.status == RosterMemberStatus.ACTIVE

    @property
    def is_available(self) -> bool:
        return (
            self.status == RosterMemberStatus.ACTIVE
            and not self.teammate.is_busy
            and not self.teammate.is_stopped
        )

    def touch(self) -> None:
        self.updated_at = now_timestamp()

    def enable(self) -> None:
        self.status = RosterMemberStatus.ACTIVE
        self.touch()

    def disable(self) -> None:
        self.status = RosterMemberStatus.DISABLED
        self.touch()

    def mark_removed(self) -> None:
        self.status = RosterMemberStatus.REMOVED
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "name": self.name,
            "role": self.role,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "teammate_status": self.teammate.status.value,
            "is_active": self.is_active,
            "is_available": self.is_available,
            "joined_at": self.joined_at,
            "updated_at": self.updated_at,
            "metadata": safe_jsonable(self.metadata),
            "teammate": self.teammate.to_dict(),
        }


RosterFilter = Callable[[RosterMember], bool]


class TeamRoster:
    """
    Team 成员管理器。

    职责：
    - add_teammate
    - create_teammate
    - remove_teammate
    - enable / disable
    - 按 role 查找
    - 选择可用成员
    - round-robin 分配成员
    """

    def __init__(
        self,
        *,
        mailbox: AgentMailbox | None = None,
        manager: SubAgentManager | None = None,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path = ".",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.mailbox = mailbox or create_agent_mailbox(
            metadata={
                "owner": "TeamRoster",
            }
        )
        self.manager = manager
        self.llm = llm
        self.tool_definitions = [
            dict(item)
            for item in (tool_definitions or [])
        ]
        self.workspace_path = Path(workspace_path)
        self.metadata = metadata or {}

        self._members: dict[str, RosterMember] = {}
        self._removed_members: dict[str, RosterMember] = {}
        self._role_cursors: dict[str, int] = {}

    # ------------------------------------------------------------------
    # basics
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._members)

    def __contains__(
        self,
        teammate_id: str,
    ) -> bool:
        return teammate_id in self._members

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_count": len(self._members),
            "active_member_count": len(self.list_members(status=RosterMemberStatus.ACTIVE)),
            "disabled_member_count": len(self.list_members(status=RosterMemberStatus.DISABLED)),
            "roles": self.list_roles(),
            "members": [
                member.to_dict()
                for member in self.list_members()
            ],
            "metadata": safe_jsonable(self.metadata),
        }

    def has_member(
        self,
        teammate_id: str,
    ) -> bool:
        return teammate_id in self._members

    def require_member(
        self,
        teammate_id: str,
    ) -> RosterMember:
        member = self._members.get(teammate_id)

        if member is None:
            raise RosterMemberNotFoundError(
                f"Roster member not found: {teammate_id}"
            )

        return member

    def get_member(
        self,
        teammate_id: str,
    ) -> RosterMember | None:
        return self._members.get(teammate_id)

    def require_teammate(
        self,
        teammate_id: str,
    ) -> TeammateAgent:
        return self.require_member(teammate_id).teammate

    def get_teammate(
        self,
        teammate_id: str,
    ) -> TeammateAgent | None:
        member = self.get_member(teammate_id)

        if member is None:
            return None

        return member.teammate

    # ------------------------------------------------------------------
    # add / create / remove
    # ------------------------------------------------------------------

    def add_teammate(
        self,
        teammate: TeammateAgent,
        *,
        status: RosterMemberStatus | str = RosterMemberStatus.ACTIVE,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> RosterMember:
        teammate_id = teammate.teammate_id

        if not teammate_id:
            raise RosterValidationError("teammate_id is required")

        if teammate_id in self._members and not replace:
            raise RosterMemberAlreadyExistsError(
                f"Roster member already exists: {teammate_id}"
            )

        member = RosterMember(
            teammate=teammate,
            status=normalize_member_status(status),
            metadata=dict(metadata or {}),
        )

        self._members[teammate_id] = member
        self._removed_members.pop(teammate_id, None)

        return member

    def create_teammate(
        self,
        *,
        teammate_id: str | None = None,
        name: str = "",
        role: str = "general",
        agent_name: str | None = None,
        description: str = "",
        workspace_path: str | Path | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        auto_ack_messages: bool = True,
        auto_read_messages: bool = True,
        reply_with_results: bool = True,
        metadata: dict[str, Any] | None = None,
        status: RosterMemberStatus | str = RosterMemberStatus.ACTIVE,
        replace: bool = False,
    ) -> RosterMember:
        teammate = create_teammate(
            teammate_id=teammate_id,
            name=name,
            role=role,
            agent_name=agent_name,
            description=description,
            workspace_path=workspace_path or self.workspace_path,
            mailbox=self.mailbox,
            manager=self.manager,
            llm=self.llm,
            tool_definitions=self.tool_definitions,
            tool_scope=tool_scope,
            max_steps=max_steps,
            auto_ack_messages=auto_ack_messages,
            auto_read_messages=auto_read_messages,
            reply_with_results=reply_with_results,
            metadata=metadata,
        )

        return self.add_teammate(
            teammate,
            status=status,
            metadata=metadata,
            replace=replace,
        )

    def add_from_spec(
        self,
        spec: TeammateSpec,
        *,
        status: RosterMemberStatus | str = RosterMemberStatus.ACTIVE,
        replace: bool = False,
    ) -> RosterMember:
        teammate = create_teammate(
            teammate_id=spec.teammate_id,
            name=spec.name,
            role=spec.role,
            agent_name=spec.agent_name,
            description=spec.description,
            workspace_path=spec.workspace_path,
            mailbox=self.mailbox,
            manager=self.manager,
            llm=self.llm,
            tool_definitions=self.tool_definitions,
            tool_scope=spec.tool_scope,
            max_steps=spec.max_steps,
            auto_ack_messages=spec.auto_ack_messages,
            auto_read_messages=spec.auto_read_messages,
            reply_with_results=spec.reply_with_results,
            metadata=spec.metadata,
        )

        return self.add_teammate(
            teammate,
            status=status,
            metadata=spec.metadata,
            replace=replace,
        )

    def remove_teammate(
        self,
        teammate_id: str,
        *,
        keep_removed: bool = True,
    ) -> RosterMember:
        member = self._members.pop(teammate_id, None)

        if member is None:
            raise RosterMemberNotFoundError(
                f"Roster member not found: {teammate_id}"
            )

        member.mark_removed()

        if keep_removed:
            self._removed_members[teammate_id] = member

        return member

    def clear(
        self,
        *,
        keep_removed: bool = False,
    ) -> None:
        if keep_removed:
            for member in self._members.values():
                member.mark_removed()
                self._removed_members[member.teammate_id] = member

        self._members.clear()
        self._role_cursors.clear()

    # ------------------------------------------------------------------
    # enable / disable
    # ------------------------------------------------------------------

    def enable_member(
        self,
        teammate_id: str,
    ) -> RosterMember:
        member = self.require_member(teammate_id)
        member.enable()
        return member

    def disable_member(
        self,
        teammate_id: str,
    ) -> RosterMember:
        member = self.require_member(teammate_id)
        member.disable()
        return member

    def set_member_status(
        self,
        teammate_id: str,
        status: RosterMemberStatus | str,
    ) -> RosterMember:
        member = self.require_member(teammate_id)
        normalized = normalize_member_status(status)

        if normalized == RosterMemberStatus.ACTIVE:
            member.enable()
        elif normalized == RosterMemberStatus.DISABLED:
            member.disable()
        elif normalized == RosterMemberStatus.REMOVED:
            self.remove_teammate(teammate_id)
            member.mark_removed()

        return member

    # ------------------------------------------------------------------
    # list / filter
    # ------------------------------------------------------------------

    def list_members(
        self,
        *,
        role: str | None = None,
        status: RosterMemberStatus | str | None = None,
        include_busy: bool = True,
        include_stopped: bool = True,
        predicate: RosterFilter | None = None,
    ) -> list[RosterMember]:
        members = list(self._members.values())

        if role is not None:
            normalized_role = normalize_teammate_role(role)
            members = [
                member
                for member in members
                if member.role == normalized_role
            ]

        if status is not None:
            normalized_status = normalize_member_status(status)
            members = [
                member
                for member in members
                if member.status == normalized_status
            ]

        if not include_busy:
            members = [
                member
                for member in members
                if not member.teammate.is_busy
            ]

        if not include_stopped:
            members = [
                member
                for member in members
                if not member.teammate.is_stopped
            ]

        if predicate is not None:
            members = [
                member
                for member in members
                if predicate(member)
            ]

        members.sort(
            key=lambda member: (
                member.role,
                member.joined_at,
                member.teammate_id,
            )
        )

        return members

    def list_teammates(
        self,
        **kwargs: Any,
    ) -> list[TeammateAgent]:
        return [
            member.teammate
            for member in self.list_members(**kwargs)
        ]

    def list_roles(self) -> list[str]:
        return sorted(
            {
                member.role
                for member in self._members.values()
            }
        )

    def count_by_role(self) -> dict[str, int]:
        counts: dict[str, int] = {}

        for member in self._members.values():
            counts[member.role] = counts.get(member.role, 0) + 1

        return counts

    def list_removed_members(self) -> list[RosterMember]:
        members = list(self._removed_members.values())
        members.sort(key=lambda member: member.updated_at)
        return members

    # ------------------------------------------------------------------
    # selection
    # ------------------------------------------------------------------

    def available_members(
        self,
        *,
        role: str | None = None,
    ) -> list[RosterMember]:
        return [
            member
            for member in self.list_members(
                role=role,
                status=RosterMemberStatus.ACTIVE,
                include_busy=False,
                include_stopped=False,
            )
            if member.is_available
        ]

    def require_available_member(
        self,
        *,
        role: str | None = None,
    ) -> RosterMember:
        selected = self.select_member(
            role=role,
        )

        if selected is None:
            role_text = normalize_teammate_role(role) if role else "any"
            raise RosterMemberNotFoundError(
                f"No available roster member for role: {role_text}"
            )

        return selected

    def select_member(
        self,
        *,
        role: str | None = None,
        strategy: str = "round_robin",
    ) -> RosterMember | None:
        candidates = self.available_members(
            role=role,
        )

        if not candidates:
            return None

        strategy = strategy.strip().lower()

        if strategy in {
            "first",
            "first_available",
        }:
            return candidates[0]

        if strategy in {
            "round_robin",
            "rr",
        }:
            key = normalize_teammate_role(role) if role else "__all__"
            cursor = self._role_cursors.get(key, 0)
            selected = candidates[cursor % len(candidates)]
            self._role_cursors[key] = cursor + 1
            return selected

        if strategy in {
            "least_busy",
            "idle",
        }:
            return sorted(
                candidates,
                key=lambda member: (
                    member.teammate.status != TeammateStatus.IDLE,
                    member.joined_at,
                ),
            )[0]

        raise RosterValidationError(
            f"Unknown member selection strategy: {strategy}"
        )

    def select_teammate(
        self,
        *,
        role: str | None = None,
        strategy: str = "round_robin",
    ) -> TeammateAgent | None:
        member = self.select_member(
            role=role,
            strategy=strategy,
        )

        if member is None:
            return None

        return member.teammate

    # ------------------------------------------------------------------
    # batch utilities
    # ------------------------------------------------------------------

    async def stop_all(
        self,
        *,
        reason: str | None = None,
        include_disabled: bool = True,
    ) -> int:
        count = 0

        for member in list(self._members.values()):
            if not include_disabled and member.status == RosterMemberStatus.DISABLED:
                continue

            await member.teammate.stop(
                reason=reason or "roster stop_all"
            )
            count += 1

        return count

    async def cancel_all_current(
        self,
        *,
        reason: str | None = None,
        include_disabled: bool = True,
    ) -> int:
        count = 0

        for member in list(self._members.values()):
            if not include_disabled and member.status == RosterMemberStatus.DISABLED:
                continue

            cancelled = await member.teammate.cancel_current(
                reason=reason or "roster cancel_all_current"
            )

            if cancelled:
                count += 1

        return count


def create_team_roster(
    *,
    mailbox: AgentMailbox | None = None,
    manager: SubAgentManager | None = None,
    llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    workspace_path: str | Path = ".",
    metadata: dict[str, Any] | None = None,
) -> TeamRoster:
    return TeamRoster(
        mailbox=mailbox,
        manager=manager,
        llm=llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        metadata=metadata,
    )


__all__ = [
    "RosterError",
    "RosterFilter",
    "RosterMember",
    "RosterMemberAlreadyExistsError",
    "RosterMemberDisabledError",
    "RosterMemberNotFoundError",
    "RosterMemberStatus",
    "RosterValidationError",
    "TeamRoster",
    "create_team_roster",
    "normalize_member_status",
]
