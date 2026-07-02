from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pytest

from pywork.tui.components.tasks import (
    TaskProgressPanel,
    build_task_snapshot,
    build_task_snapshot_from_manager,
    format_duration_ms,
    render_task_progress_panel,
    status_style,
    task_record_to_row,
)


@dataclass
class FakeTask:
    id: str
    name: str
    agent_id: str = ""
    status: str = "pending"
    started_at: datetime | None = None
    finished_at: datetime | None = None
    updated_at: datetime | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class FakeTaskManager:
    def __init__(self, tasks):
        self.tasks = list(tasks)
        self.active_ids = {
            task.id
            for task in self.tasks
            if task.status == "running"
        }

    async def list_tasks(self, limit=None):
        if limit is None:
            return self.tasks

        return self.tasks[:limit]

    def get_active_task_ids(self):
        return list(self.active_ids)


def test_format_duration_ms() -> None:
    assert format_duration_ms(None) == "-"
    assert format_duration_ms(120) == "120ms"
    assert format_duration_ms(1_500) == "1.5s"
    assert format_duration_ms(65_000) == "1m 5s"


def test_status_style() -> None:
    assert status_style("running") == "bold yellow"
    assert status_style("succeeded") == "green"
    assert status_style("failed") == "bold red"


def test_task_record_to_row() -> None:
    started = datetime.now() - timedelta(seconds=2)
    finished = datetime.now()

    task = FakeTask(
        id="task_1",
        name="Run tests",
        agent_id="verifier",
        status="succeeded",
        started_at=started,
        finished_at=finished,
    )

    row = task_record_to_row(task)

    assert row.task_id == "task_1"
    assert row.name == "Run tests"
    assert row.agent == "verifier"
    assert row.status == "succeeded"
    assert row.duration_ms is not None
    assert row.is_terminal is True


def test_build_task_snapshot_counts_statuses() -> None:
    tasks = [
        FakeTask(
            id="task_1",
            name="Plan",
            status="pending",
        ),
        FakeTask(
            id="task_2",
            name="Run tests",
            agent_id="verifier",
            status="running",
        ),
        FakeTask(
            id="task_3",
            name="Review",
            agent_id="reviewer",
            status="failed",
            error="boom",
        ),
    ]

    snapshot = build_task_snapshot(
        tasks,
        active_task_ids={"task_2"},
    )

    assert snapshot.stats.total == 3
    assert snapshot.stats.active == 1
    assert snapshot.stats.pending == 1
    assert snapshot.stats.running == 1
    assert snapshot.stats.failed == 1

    running = [
        row
        for row in snapshot.rows
        if row.task_id == "task_2"
    ][0]

    assert running.is_active is True


@pytest.mark.asyncio
async def test_build_task_snapshot_from_manager() -> None:
    manager = FakeTaskManager(
        [
            FakeTask(
                id="task_1",
                name="Task 1",
                status="running",
                agent_id="agent_a",
            ),
            FakeTask(
                id="task_2",
                name="Task 2",
                status="succeeded",
                agent_id="agent_b",
            ),
        ]
    )

    snapshot = await build_task_snapshot_from_manager(manager)

    assert snapshot.stats.total == 2
    assert snapshot.stats.active == 1

    ids = {
        row.task_id
        for row in snapshot.rows
    }

    assert ids == {
        "task_1",
        "task_2",
    }


def test_render_task_progress_panel() -> None:
    snapshot = build_task_snapshot(
        [
            FakeTask(
                id="task_1",
                name="Run pytest",
                status="running",
                agent_id="verifier",
            )
        ],
        active_task_ids={"task_1"},
    )

    renderable = render_task_progress_panel(snapshot)

    assert renderable is not None


def test_task_progress_panel_set_tasks() -> None:
    panel = TaskProgressPanel()

    panel.set_tasks(
        [
            FakeTask(
                id="task_1",
                name="Run pytest",
                status="running",
                agent_id="verifier",
            )
        ],
        active_task_ids={"task_1"},
    )

    stats = panel.get_stats()

    assert stats["total"] == 1
    assert stats["active"] == 1