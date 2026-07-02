from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pywork.tui.components.teams import (
    TeamViewPanel,
    build_team_snapshot,
    collect_mailbox_stats,
    render_team_view_panel,
    shared_task_to_row,
    teammate_to_member_row,
)


@dataclass
class FakeTeammate:
    teammate_id: str
    name: str
    role: str
    agent_name: str
    status: str = "active"
    current_run_id: str | None = None
    current_task_record_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_busy(self) -> bool:
        return self.status == "running"

    @property
    def is_stopped(self) -> bool:
        return self.status == "stopped"

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "name": self.name,
            "role": self.role,
            "agent_name": self.agent_name,
            "status": self.status,
            "current_run_id": self.current_run_id,
            "current_task_record_id": self.current_task_record_id,
            "metadata": self.metadata,
        }


@dataclass
class FakeTask:
    task_id: str
    title: str
    role: str = ""
    assigned_to: str = ""
    status: str = "pending"
    priority: str = "normal"
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"succeeded", "failed", "cancelled"}

    @property
    def is_active(self) -> bool:
        return self.status in {"assigned", "dispatched", "running"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "role": self.role,
            "assigned_to": self.assigned_to,
            "status": self.status,
            "priority": self.priority,
            "error": self.error,
            "metadata": self.metadata,
            "is_terminal": self.is_terminal,
            "is_active": self.is_active,
        }


@dataclass
class FakeMessage:
    status: str
    message_type: str


class FakeMailbox:
    def __init__(self, messages):
        self.messages = list(messages)

    def count_messages(self):
        return len(self.messages)

    def list_messages(self, include_deleted=False):
        if include_deleted:
            return self.messages

        return [
            message
            for message in self.messages
            if message.status != "deleted"
        ]


class FakeTeam:
    def __init__(self):
        self.team_id = "team_1"
        self.name = "Team One"
        self.description = "测试团队"
        self.metadata = {
            "source": "test",
        }
        self.mailbox = FakeMailbox(
            [
                FakeMessage(status="delivered", message_type="task"),
                FakeMessage(status="read", message_type="result"),
                FakeMessage(status="acked", message_type="note"),
                FakeMessage(status="deleted", message_type="error"),
            ]
        )
        self.members = [
            FakeTeammate(
                teammate_id="planner_1",
                name="Planner",
                role="planner",
                agent_name="planner",
                status="active",
            ),
            FakeTeammate(
                teammate_id="verifier_1",
                name="Verifier",
                role="verifier",
                agent_name="verifier",
                status="running",
                current_run_id="run_1",
            ),
        ]
        self.tasks = [
            FakeTask(
                task_id="task_1",
                title="规划实现",
                role="planner",
                assigned_to="planner_1",
                status="running",
                priority="high",
            ),
            FakeTask(
                task_id="task_2",
                title="运行测试",
                role="verifier",
                assigned_to="verifier_1",
                status="failed",
                priority="urgent",
                error="pytest failed",
            ),
        ]

    def list_members(self):
        return self.members

    def list_shared_tasks(self, include_terminal=True, limit=None):
        tasks = list(self.tasks)

        if not include_terminal:
            tasks = [
                task
                for task in tasks
                if not task.is_terminal
            ]

        if limit is not None:
            tasks = tasks[:limit]

        return tasks


def test_teammate_to_member_row() -> None:
    row = teammate_to_member_row(
        FakeTeammate(
            teammate_id="planner_1",
            name="Planner",
            role="planner",
            agent_name="planner",
            status="running",
            current_run_id="run_1",
        )
    )

    assert row.teammate_id == "planner_1"
    assert row.name == "Planner"
    assert row.role == "planner"
    assert row.agent_name == "planner"
    assert row.is_busy is True


def test_shared_task_to_row() -> None:
    row = shared_task_to_row(
        FakeTask(
            task_id="task_1",
            title="实现 Team 视图",
            role="planner",
            assigned_to="planner_1",
            status="running",
            priority="high",
        )
    )

    assert row.task_id == "task_1"
    assert row.title == "实现 Team 视图"
    assert row.status == "running"
    assert row.priority == "high"
    assert row.is_active is True


def test_collect_mailbox_stats() -> None:
    mailbox = FakeMailbox(
        [
            FakeMessage(status="delivered", message_type="task"),
            FakeMessage(status="read", message_type="result"),
            FakeMessage(status="acked", message_type="note"),
            FakeMessage(status="deleted", message_type="error"),
        ]
    )

    stats = collect_mailbox_stats(mailbox)

    assert stats.total == 4
    assert stats.unread == 1
    assert stats.read == 1
    assert stats.acked == 1
    assert stats.deleted == 1
    assert stats.task_messages == 1
    assert stats.result_messages == 1
    assert stats.error_messages == 1


@pytest.mark.asyncio
async def test_build_team_snapshot() -> None:
    team = FakeTeam()

    snapshot = await build_team_snapshot(team)

    assert snapshot.team_id == "team_1"
    assert snapshot.name == "Team One"
    assert snapshot.description == "测试团队"

    assert snapshot.stats.members_total == 2
    assert snapshot.stats.members_active == 2
    assert snapshot.stats.members_busy == 1

    assert snapshot.stats.tasks_total == 2
    assert snapshot.stats.tasks_active == 1
    assert snapshot.stats.tasks_failed == 1

    assert snapshot.stats.mailbox.total == 4
    assert snapshot.stats.mailbox.unread == 1


@pytest.mark.asyncio
async def test_build_team_snapshot_excludes_terminal_tasks() -> None:
    team = FakeTeam()

    snapshot = await build_team_snapshot(
        team,
        include_terminal_tasks=False,
    )

    assert snapshot.stats.tasks_total == 1
    assert snapshot.tasks[0].task_id == "task_1"


def test_render_team_view_panel() -> None:
    snapshot = TeamViewPanel().snapshot

    renderable = render_team_view_panel(snapshot)

    assert renderable is not None


@pytest.mark.asyncio
async def test_team_view_panel_refresh_from_team() -> None:
    panel = TeamViewPanel()
    team = FakeTeam()

    snapshot = await panel.refresh_from_team(team)

    assert snapshot.team_id == "team_1"

    stats = panel.get_stats()

    assert stats["members_total"] == 2
    assert stats["tasks_total"] == 2
    assert stats["mailbox"]["total"] == 4