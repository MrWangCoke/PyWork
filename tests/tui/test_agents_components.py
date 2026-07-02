from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from pywork.tui.components.agents import (
    AgentActivityPanel,
    agent_run_to_row,
    build_agent_snapshot,
    build_agent_snapshot_from_manager,
    build_agent_snapshot_from_sources,
    format_duration_ms,
    render_agent_activity_panel,
    status_style,
    teammate_to_row,
)


@dataclass
class FakeTaskResult:
    task: str
    run_id: str = "run_1"
    task_record_id: str | None = None
    error: str | None = None
    started_at: float = field(default_factory=lambda: time.time() - 1)
    finished_at: float | None = None

    @property
    def duration_ms(self):
        if self.finished_at is None:
            return None

        return int((self.finished_at - self.started_at) * 1000)


@dataclass
class FakeTeammate:
    teammate_id: str
    name: str
    role: str
    agent_name: str
    status: str = "idle"
    current_run_id: str | None = None
    current_task_record_id: str | None = None
    last_task_result: FakeTaskResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_busy(self) -> bool:
        return self.status == "running"

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


class FakeManager:
    def __init__(self, runs):
        self.runs = list(runs)

    def get_active_runs(self):
        return self.runs


class FakeTeam:
    def __init__(self, agents, manager=None):
        self.agents = list(agents)
        self.manager = manager

    def list_teammates(self):
        return self.agents


def test_format_duration_ms() -> None:
    assert format_duration_ms(None) == "-"
    assert format_duration_ms(200) == "200ms"
    assert format_duration_ms(1_500) == "1.5s"
    assert format_duration_ms(65_000) == "1m 5s"


def test_status_style() -> None:
    assert status_style("running") == "bold yellow"
    assert status_style("idle") == "dim"
    assert status_style("failed") == "bold red"


def test_agent_run_to_row() -> None:
    row = agent_run_to_row(
        {
            "run_id": "run_1",
            "agent_name": "planner",
            "task": "规划实现",
            "status": "running",
            "started_at": time.time() - 2,
            "metadata": {
                "teammate_id": "planner_1",
                "teammate_role": "planner",
            },
        }
    )

    assert row.agent_id == "planner_1"
    assert row.name == "planner"
    assert row.role == "planner"
    assert row.status == "running"
    assert row.current_task == "规划实现"
    assert row.current_run_id == "run_1"
    assert row.is_active is True
    assert row.duration_ms is not None


def test_teammate_to_row() -> None:
    teammate = FakeTeammate(
        teammate_id="verifier_1",
        name="Verifier",
        role="verifier",
        agent_name="verifier",
        status="running",
        current_run_id="run_2",
        current_task_record_id="task_1",
        last_task_result=FakeTaskResult(
            task="运行测试",
            run_id="run_2",
            task_record_id="task_1",
        ),
    )

    row = teammate_to_row(teammate)

    assert row.agent_id == "verifier_1"
    assert row.name == "Verifier"
    assert row.role == "verifier"
    assert row.status == "running"
    assert row.current_task == "运行测试"
    assert row.current_run_id == "run_2"
    assert row.current_task_record_id == "task_1"
    assert row.is_active is True


def test_build_agent_snapshot_active_only() -> None:
    agents = [
        FakeTeammate(
            teammate_id="planner_1",
            name="Planner",
            role="planner",
            agent_name="planner",
            status="idle",
        ),
        FakeTeammate(
            teammate_id="reviewer_1",
            name="Reviewer",
            role="reviewer",
            agent_name="reviewer",
            status="running",
        ),
    ]

    snapshot = build_agent_snapshot(
        agents,
        active_only=True,
    )

    assert snapshot.stats.total == 1
    assert snapshot.stats.active == 1
    assert snapshot.rows[0].agent_id == "reviewer_1"


@pytest.mark.asyncio
async def test_build_agent_snapshot_from_manager() -> None:
    manager = FakeManager(
        [
            {
                "run_id": "run_1",
                "agent_name": "planner",
                "task": "规划任务",
                "status": "running",
                "started_at": time.time() - 1,
            },
            {
                "run_id": "run_2",
                "agent_name": "reviewer",
                "task": "审查代码",
                "status": "running",
                "started_at": time.time() - 2,
            },
        ]
    )

    snapshot = await build_agent_snapshot_from_manager(manager)

    assert snapshot.stats.total == 2
    assert snapshot.stats.active == 2

    names = {
        row.name
        for row in snapshot.rows
    }

    assert names == {
        "planner",
        "reviewer",
    }


@pytest.mark.asyncio
async def test_build_agent_snapshot_from_sources_team_and_manager() -> None:
    manager = FakeManager(
        [
            {
                "run_id": "run_1",
                "agent_name": "planner",
                "task": "规划任务",
                "status": "running",
                "started_at": time.time() - 1,
                "metadata": {
                    "teammate_id": "planner_1",
                },
            },
        ]
    )
    team = FakeTeam(
        [
            FakeTeammate(
                teammate_id="planner_1",
                name="Planner",
                role="planner",
                agent_name="planner",
                status="idle",
            ),
            FakeTeammate(
                teammate_id="reviewer_1",
                name="Reviewer",
                role="reviewer",
                agent_name="reviewer",
                status="idle",
            ),
        ],
        manager=manager,
    )

    snapshot = await build_agent_snapshot_from_sources(
        team=team,
        active_only=False,
    )

    assert snapshot.stats.total == 2
    assert snapshot.stats.active == 1

    planner = [
        row
        for row in snapshot.rows
        if row.agent_id == "planner_1"
    ][0]

    assert planner.is_active is True
    assert planner.current_task == "规划任务"


def test_render_agent_activity_panel() -> None:
    snapshot = build_agent_snapshot(
        [
            FakeTeammate(
                teammate_id="planner_1",
                name="Planner",
                role="planner",
                agent_name="planner",
                status="running",
            )
        ]
    )

    renderable = render_agent_activity_panel(snapshot)

    assert renderable is not None


def test_agent_activity_panel_set_agents() -> None:
    panel = AgentActivityPanel(
        active_only=False,
    )

    panel.set_agents(
        [
            FakeTeammate(
                teammate_id="planner_1",
                name="Planner",
                role="planner",
                agent_name="planner",
                status="running",
            )
        ]
    )

    stats = panel.get_stats()

    assert stats["total"] == 1
    assert stats["active"] == 1