from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any

from pywork.tui.components.teams.models import (
    TeamMailboxStats,
    TeamMemberRow,
    TeamTaskRow,
    TeamViewSnapshot,
    TeamViewStats,
    normalize_member_status,
    normalize_task_status,
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


def to_mapping_if_possible(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            data = to_dict()

            if isinstance(data, Mapping):
                return data
        except Exception:
            return None

    return None


def first_text(*values: Any) -> str:
    for value in values:
        if value is None:
            continue

        text = str(value).strip()

        if text:
            return text

    return ""


def extract_metadata(value: Any) -> dict[str, Any]:
    metadata = get_attr(value, "metadata", {})

    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


def teammate_to_member_row(teammate: Any) -> TeamMemberRow:
    data = to_mapping_if_possible(teammate)
    source = data if data is not None else teammate

    status = normalize_member_status(
        get_attr(source, "status", None)
        or get_attr(teammate, "status", None)
    )

    is_busy = bool(
        get_attr(teammate, "is_busy", False)
        or status == "running"
    )
    is_stopped = bool(
        get_attr(teammate, "is_stopped", False)
        or status in {"stopped", "removed"}
    )

    return TeamMemberRow(
        teammate_id=first_text(
            get_attr(source, "teammate_id"),
            get_attr(source, "id"),
            get_attr(source, "agent_name"),
        ),
        name=first_text(
            get_attr(source, "name"),
            get_attr(source, "teammate_id"),
            get_attr(source, "agent_name"),
            "Teammate",
        ),
        role=first_text(
            get_attr(source, "role"),
            get_attr(source, "agent_name"),
        ),
        agent_name=first_text(
            get_attr(source, "agent_name"),
        ),
        status=status,
        current_run_id=first_text(
            get_attr(source, "current_run_id"),
        ) or None,
        current_task_record_id=first_text(
            get_attr(source, "current_task_record_id"),
        ) or None,
        is_busy=is_busy,
        is_stopped=is_stopped,
        metadata=safe_jsonable(extract_metadata(source)),
    )


def roster_member_to_row(member: Any) -> TeamMemberRow:
    teammate = get_attr(member, "teammate", None)

    if teammate is None and isinstance(member, Mapping):
        teammate = member.get("teammate")

    if teammate is None:
        return teammate_to_member_row(member)

    row = teammate_to_member_row(teammate)

    member_status = get_attr(member, "status", None)

    if member_status is not None:
        row.status = normalize_member_status(member_status)

    return row


def shared_task_to_row(task: Any) -> TeamTaskRow:
    data = to_mapping_if_possible(task)
    source = data if data is not None else task

    return TeamTaskRow(
        task_id=first_text(
            get_attr(source, "task_id"),
            get_attr(source, "id"),
        ),
        title=first_text(
            get_attr(source, "title"),
            get_attr(source, "name"),
            get_attr(source, "description"),
            "Team task",
        ),
        role=first_text(
            get_attr(source, "role"),
        ),
        assigned_to=first_text(
            get_attr(source, "assigned_to"),
        ),
        status=normalize_task_status(
            get_attr(source, "status", None)
        ),
        priority=first_text(
            get_attr(source, "priority"),
            "normal",
        ),
        error=(
            str(get_attr(source, "error"))
            if get_attr(source, "error") is not None
            else None
        ),
        created_at=get_attr(source, "created_at"),
        updated_at=get_attr(source, "updated_at"),
        metadata=safe_jsonable(extract_metadata(source)),
    )


async def collect_team_members(team: Any) -> list[TeamMemberRow]:
    list_members = getattr(team, "list_members", None)

    if callable(list_members):
        members = await maybe_await(list_members())

        if members is not None:
            return [
                roster_member_to_row(member)
                for member in members
            ]

    roster = get_attr(team, "roster", None)

    if roster is not None:
        roster_list_members = getattr(roster, "list_members", None)

        if callable(roster_list_members):
            members = await maybe_await(roster_list_members())

            if members is not None:
                return [
                    roster_member_to_row(member)
                    for member in members
                ]

    list_teammates = getattr(team, "list_teammates", None)

    if callable(list_teammates):
        teammates = await maybe_await(list_teammates())

        if teammates is not None:
            return [
                teammate_to_member_row(teammate)
                for teammate in teammates
            ]

    return []


async def collect_team_tasks(
    team: Any,
    *,
    task_limit: int | None = None,
    include_terminal: bool = True,
) -> list[TeamTaskRow]:
    list_shared_tasks = getattr(team, "list_shared_tasks", None)

    if callable(list_shared_tasks):
        try:
            tasks = await maybe_await(
                list_shared_tasks(
                    include_terminal=include_terminal,
                    limit=task_limit,
                )
            )
        except TypeError:
            tasks = await maybe_await(list_shared_tasks())

        if tasks is not None:
            return [
                shared_task_to_row(task)
                for task in tasks
            ]

    shared_task_list = get_attr(team, "shared_task_list", None)

    if isinstance(shared_task_list, Mapping):
        tasks = list(shared_task_list.values())

        rows = [
            shared_task_to_row(task)
            for task in tasks
        ]

        if not include_terminal:
            rows = [
                row
                for row in rows
                if not row.is_terminal
            ]

        if task_limit is not None:
            rows = rows[:task_limit]

        return rows

    return []


def collect_mailbox_stats(mailbox: Any | None) -> TeamMailboxStats:
    stats = TeamMailboxStats()

    if mailbox is None:
        return stats

    list_messages = getattr(mailbox, "list_messages", None)

    if callable(list_messages):
        try:
            messages = list_messages(include_deleted=True)
        except TypeError:
            messages = list_messages()
    else:
        messages_map = get_attr(mailbox, "_messages", {})

        if isinstance(messages_map, Mapping):
            messages = list(messages_map.values())
        else:
            messages = []

    stats.total = len(messages)

    for message in messages:
        status = str(get_attr(message, "status", "") or "").lower()
        message_type = str(get_attr(message, "message_type", "") or "").lower()

        if "." in status:
            status = status.rsplit(".", 1)[-1]

        if "." in message_type:
            message_type = message_type.rsplit(".", 1)[-1]

        if status in {"pending", "delivered"}:
            stats.unread += 1
        elif status == "read":
            stats.read += 1
        elif status == "acked":
            stats.acked += 1
        elif status == "archived":
            stats.archived += 1
        elif status == "deleted":
            stats.deleted += 1

        if message_type == "task":
            stats.task_messages += 1
        elif message_type == "result":
            stats.result_messages += 1
        elif message_type == "error":
            stats.error_messages += 1

    return stats


def collect_stats(
    members: Sequence[TeamMemberRow],
    tasks: Sequence[TeamTaskRow],
    mailbox_stats: TeamMailboxStats,
) -> TeamViewStats:
    stats = TeamViewStats(
        members_total=len(members),
        tasks_total=len(tasks),
        mailbox=mailbox_stats,
    )

    for member in members:
        if member.is_active:
            stats.members_active += 1

        if member.is_busy:
            stats.members_busy += 1

        if member.is_stopped:
            stats.members_stopped += 1

    for task in tasks:
        if task.is_active:
            stats.tasks_active += 1

        if task.status == "pending":
            stats.tasks_pending += 1
        elif task.status == "succeeded":
            stats.tasks_succeeded += 1
        elif task.status == "failed":
            stats.tasks_failed += 1
        elif task.status == "cancelled":
            stats.tasks_cancelled += 1

    return stats


async def build_team_snapshot(
    team: Any,
    *,
    task_limit: int | None = 8,
    include_terminal_tasks: bool = True,
) -> TeamViewSnapshot:
    members = await collect_team_members(team)
    tasks = await collect_team_tasks(
        team,
        task_limit=task_limit,
        include_terminal=include_terminal_tasks,
    )
    mailbox_stats = collect_mailbox_stats(
        get_attr(team, "mailbox", None)
    )

    metadata = extract_metadata(team)

    return TeamViewSnapshot(
        team_id=first_text(
            get_attr(team, "team_id"),
            get_attr(team, "id"),
        ),
        name=first_text(
            get_attr(team, "name"),
            get_attr(team, "team_id"),
            "Team",
        ),
        description=first_text(
            get_attr(team, "description"),
        ),
        members=members,
        tasks=tasks,
        stats=collect_stats(
            members,
            tasks,
            mailbox_stats,
        ),
        metadata=safe_jsonable(metadata),
    )


def collect_teams_from_registry(registry: Any) -> list[Any]:
    if registry is None:
        return []

    if isinstance(registry, Mapping):
        return list(registry.values())

    teams = get_attr(registry, "teams", None)

    if isinstance(teams, Mapping):
        return list(teams.values())

    list_teams = getattr(registry, "list_teams", None)

    if callable(list_teams):
        result = list_teams()

        if result is not None:
            return list(result)

    return []