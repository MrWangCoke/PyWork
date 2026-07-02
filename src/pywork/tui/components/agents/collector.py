from __future__ import annotations

import inspect
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pywork.tui.components.agents.models import (
    ACTIVE_AGENT_STATUSES,
    AgentActivityRow,
    AgentActivitySnapshot,
    AgentActivityStats,
    normalize_agent_status,
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


def timestamp_value(value: Any) -> float | None:
    if value is None:
        return None

    if isinstance(value, int | float):
        return float(value)

    if isinstance(value, datetime):
        return value.timestamp()

    return None


def calculate_duration_ms(
    *,
    started_at: Any,
    finished_at: Any = None,
    duration_ms: Any = None,
    is_active: bool = False,
) -> int | None:
    if isinstance(duration_ms, int | float):
        return max(0, int(duration_ms))

    start = timestamp_value(started_at)
    finish = timestamp_value(finished_at)

    if start is None:
        return None

    if finish is None and is_active:
        finish = time.time()

    if finish is None:
        return None

    return max(0, int((finish - start) * 1000))


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


def agent_run_to_row(run: Any) -> AgentActivityRow:
    metadata = extract_metadata(run)

    status = normalize_agent_status(get_attr(run, "status", "running"))
    is_active = status in ACTIVE_AGENT_STATUSES

    teammate_id = metadata.get("teammate_id")
    agent_name = first_text(
        get_attr(run, "agent_name"),
        metadata.get("agent_name"),
    )
    run_id = first_text(
        get_attr(run, "run_id"),
        metadata.get("run_id"),
    )

    agent_id = first_text(
        teammate_id,
        agent_name,
        run_id,
    )

    started_at = get_attr(run, "started_at")
    finished_at = get_attr(run, "finished_at")

    return AgentActivityRow(
        agent_id=agent_id,
        name=agent_name or agent_id or "Agent",
        role=first_text(
            metadata.get("teammate_role"),
            metadata.get("role"),
            agent_name,
        ),
        status=status,
        current_task=first_text(
            get_attr(run, "task"),
            metadata.get("task"),
        ),
        current_run_id=run_id or None,
        current_task_record_id=first_text(
            metadata.get("task_record_id"),
            metadata.get("current_task_record_id"),
        ) or None,
        started_at=timestamp_value(started_at),
        finished_at=timestamp_value(finished_at),
        duration_ms=calculate_duration_ms(
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=get_attr(run, "duration_ms"),
            is_active=is_active,
        ),
        is_active=is_active,
        error=(
            str(get_attr(run, "error"))
            if get_attr(run, "error") is not None
            else None
        ),
        metadata=safe_jsonable(metadata),
    )


def teammate_to_row(teammate: Any) -> AgentActivityRow:
    data = to_mapping_if_possible(teammate)

    source = data if data is not None else teammate
    metadata = extract_metadata(source)

    status = normalize_agent_status(
        get_attr(source, "status", None)
        or get_attr(teammate, "status", None)
    )
    is_active = bool(get_attr(teammate, "is_busy", False)) or status in ACTIVE_AGENT_STATUSES

    last_task_result = get_attr(teammate, "last_task_result", None)

    current_task = first_text(
        get_attr(source, "current_task"),
        get_attr(last_task_result, "task"),
        metadata.get("current_task"),
    )

    started_at = get_attr(last_task_result, "started_at")
    finished_at = get_attr(last_task_result, "finished_at")

    return AgentActivityRow(
        agent_id=first_text(
            get_attr(source, "teammate_id"),
            get_attr(source, "agent_id"),
            get_attr(source, "agent_name"),
        ),
        name=first_text(
            get_attr(source, "name"),
            get_attr(source, "teammate_id"),
            get_attr(source, "agent_name"),
            "Agent",
        ),
        role=first_text(
            get_attr(source, "role"),
            metadata.get("role"),
        ),
        status=status,
        current_task=current_task,
        current_run_id=first_text(
            get_attr(source, "current_run_id"),
            get_attr(last_task_result, "run_id"),
        ) or None,
        current_task_record_id=first_text(
            get_attr(source, "current_task_record_id"),
            get_attr(last_task_result, "task_record_id"),
        ) or None,
        started_at=timestamp_value(started_at),
        finished_at=timestamp_value(finished_at),
        duration_ms=calculate_duration_ms(
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=get_attr(last_task_result, "duration_ms"),
            is_active=is_active,
        ),
        is_active=is_active,
        error=(
            str(get_attr(last_task_result, "error"))
            if get_attr(last_task_result, "error") is not None
            else None
        ),
        metadata=safe_jsonable(metadata),
    )


def roster_member_to_row(member: Any) -> AgentActivityRow:
    teammate = get_attr(member, "teammate", None)

    if teammate is None and isinstance(member, Mapping):
        teammate = member.get("teammate")

    if teammate is None:
        return teammate_to_row(member)

    row = teammate_to_row(teammate)

    member_status = get_attr(member, "status", None)

    if member_status is not None and row.status == "unknown":
        row.status = normalize_agent_status(member_status)
        row.is_active = row.status in ACTIVE_AGENT_STATUSES

    return row


def merge_agent_rows(rows: Sequence[AgentActivityRow]) -> list[AgentActivityRow]:
    merged: dict[str, AgentActivityRow] = {}

    for row in rows:
        key = row.agent_id or row.name or row.current_run_id or ""

        if not key:
            key = f"agent_{len(merged)}"

        existing = merged.get(key)

        if existing is None:
            merged[key] = row
            continue

        if row.is_active and not existing.is_active:
            merged[key] = row
            continue

        if existing.is_active and not row.is_active:
            if not existing.name and row.name:
                existing.name = row.name
            if not existing.role and row.role:
                existing.role = row.role
            continue

        if row.started_at and (
            existing.started_at is None
            or row.started_at > existing.started_at
        ):
            merged[key] = row

    result = list(merged.values())

    result.sort(
        key=lambda row: (
            not row.is_active,
            row.role,
            row.name,
        )
    )

    return result


def collect_stats(rows: Sequence[AgentActivityRow]) -> AgentActivityStats:
    stats = AgentActivityStats(
        total=len(rows),
    )

    for row in rows:
        if row.is_active:
            stats.active += 1

        if row.status == "idle":
            stats.idle += 1
        elif row.status == "waiting":
            stats.waiting += 1
        elif row.status == "running":
            stats.running += 1
        elif row.status == "stopped":
            stats.stopped += 1
        elif row.status == "failed":
            stats.failed += 1
        elif row.status == "aborted":
            stats.aborted += 1
        elif row.status == "succeeded":
            stats.succeeded += 1
        else:
            stats.unknown += 1

    return stats


def build_agent_snapshot(
    agents: Sequence[Any],
    *,
    active_runs: Sequence[Any] | None = None,
    active_only: bool = False,
) -> AgentActivitySnapshot:
    rows: list[AgentActivityRow] = []

    for agent in agents:
        rows.append(teammate_to_row(agent))

    for run in active_runs or []:
        rows.append(agent_run_to_row(run))

    rows = merge_agent_rows(rows)

    if active_only:
        rows = [
            row
            for row in rows
            if row.is_active
        ]

    return AgentActivitySnapshot(
        rows=rows,
        stats=collect_stats(rows),
    )


async def collect_active_runs_from_manager(manager: Any) -> list[Any]:
    get_active_runs = getattr(manager, "get_active_runs", None)

    if callable(get_active_runs):
        runs = await maybe_await(get_active_runs())

        if runs is None:
            return []

        return list(runs)

    active_runs = getattr(manager, "_active_runs", None)

    if isinstance(active_runs, Mapping):
        return list(active_runs.values())

    return []


async def collect_agents_from_team(team: Any) -> list[Any]:
    roster = get_attr(team, "roster", None)

    if roster is not None:
        list_members = getattr(roster, "list_members", None)

        if callable(list_members):
            members = await maybe_await(list_members())

            if members is not None:
                return [
                    get_attr(member, "teammate", member)
                    for member in members
                ]

    list_teammates = getattr(team, "list_teammates", None)

    if callable(list_teammates):
        teammates = await maybe_await(list_teammates())

        if teammates is not None:
            return list(teammates)

    return []


async def build_agent_snapshot_from_manager(
    manager: Any,
    *,
    active_only: bool = True,
) -> AgentActivitySnapshot:
    active_runs = await collect_active_runs_from_manager(manager)

    return build_agent_snapshot(
        [],
        active_runs=active_runs,
        active_only=active_only,
    )


async def build_agent_snapshot_from_team(
    team: Any,
    *,
    active_only: bool = False,
) -> AgentActivitySnapshot:
    agents = await collect_agents_from_team(team)

    manager = get_attr(team, "manager", None)

    if manager is None:
        roster = get_attr(team, "roster", None)
        manager = get_attr(roster, "manager", None)

    active_runs = (
        await collect_active_runs_from_manager(manager)
        if manager is not None
        else []
    )

    return build_agent_snapshot(
        agents,
        active_runs=active_runs,
        active_only=active_only,
    )


async def build_agent_snapshot_from_sources(
    *,
    manager: Any | None = None,
    team: Any | None = None,
    agents: Sequence[Any] | None = None,
    active_only: bool = False,
) -> AgentActivitySnapshot:
    collected_agents: list[Any] = list(agents or [])
    active_runs: list[Any] = []

    if team is not None:
        collected_agents.extend(
            await collect_agents_from_team(team)
        )

        team_manager = get_attr(team, "manager", None)

        if team_manager is None:
            roster = get_attr(team, "roster", None)
            team_manager = get_attr(roster, "manager", None)

        if team_manager is not None:
            active_runs.extend(
                await collect_active_runs_from_manager(team_manager)
            )

    if manager is not None:
        active_runs.extend(
            await collect_active_runs_from_manager(manager)
        )

    return build_agent_snapshot(
        collected_agents,
        active_runs=active_runs,
        active_only=active_only,
    )