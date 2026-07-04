from __future__ import annotations

import pytest

from pywork.runtime.events import RuntimeEvent, RuntimeEventSource
from pywork.tui.components.tasks import (
    build_task_snapshot,
    collect_subagent_run_records,
)
from pywork.tui.app import PyWorkApp


class FakeTaskPanel:
    limit = 8

    def __init__(self) -> None:
        self.snapshot = build_task_snapshot([])

    def set_snapshot(self, snapshot) -> None:
        self.snapshot = snapshot


class EmptyTaskManager:
    async def list_tasks(self, *, limit=None):
        return []


class OneTaskManager:
    async def list_tasks(self, *, limit=None):
        return [
            {
                "id": "task_new",
                "name": "SubAgent reviewer: Review src/pywork/utils/new.py",
                "agent_id": "reviewer",
                "status": "succeeded",
                "created_at": 200.0,
                "updated_at": 201.0,
                "metadata": {},
            }
        ]


class FakeSubAgentManager:
    def get_active_runs(self):
        return []

    def get_history(self, *, limit=None):
        return [
            {
                "run_id": "reviewer_run_1",
                "agent_name": "reviewer",
                "task": "Review src/pywork/utils/diff.py",
                "status": "completed",
                "started_at": 100.0,
                "finished_at": 101.5,
                "metadata": {},
            }
        ]


def test_tui_detects_task_runtime_event() -> None:
    app = PyWorkApp()

    event = RuntimeEvent.status_event(
        status="task_started",
        source=RuntimeEventSource.SYSTEM,
        metadata={
            "task_event": True,
            "task_id": "task_1",
        },
    )

    assert app.is_task_runtime_event(event) is True


def test_tui_ignores_non_task_runtime_event() -> None:
    app = PyWorkApp()

    event = RuntimeEvent.status_event(
        status="thinking",
        source=RuntimeEventSource.ENGINE,
    )

    assert app.is_task_runtime_event(event) is False


def test_tui_task_event_schedules_task_panel_refresh(monkeypatch) -> None:
    app = PyWorkApp()

    scheduled = []

    def fake_schedule() -> None:
        scheduled.append(True)

    monkeypatch.setattr(
        app,
        "schedule_task_panel_refresh",
        fake_schedule,
    )

    event = RuntimeEvent.status_event(
        status="task_finished",
        source=RuntimeEventSource.SYSTEM,
        metadata={
            "task_event": True,
            "task_id": "task_1",
        },
    )

    app.handle_task_runtime_event(event)

    assert scheduled == [True]


@pytest.mark.asyncio
async def test_subagent_run_history_can_render_as_task_rows() -> None:
    records = await collect_subagent_run_records(FakeSubAgentManager())
    snapshot = build_task_snapshot(records)

    assert len(snapshot.rows) == 1
    assert snapshot.rows[0].task_id == "reviewer_run_1"
    assert snapshot.rows[0].agent == "reviewer"
    assert snapshot.rows[0].status == "succeeded"


@pytest.mark.asyncio
async def test_task_panel_falls_back_to_subagent_history(monkeypatch) -> None:
    app = PyWorkApp()
    panel = FakeTaskPanel()
    app.task_progress_panel = panel  # type: ignore[assignment]

    monkeypatch.setattr(
        app,
        "resolve_runtime_task_manager",
        lambda: EmptyTaskManager(),
    )
    monkeypatch.setattr(
        app,
        "resolve_runtime_subagent_manager",
        lambda: FakeSubAgentManager(),
    )

    await app.refresh_task_panel()

    assert len(panel.snapshot.rows) == 1
    assert panel.snapshot.rows[0].task_id == "reviewer_run_1"


@pytest.mark.asyncio
async def test_task_panel_merges_task_manager_and_subagent_history(monkeypatch) -> None:
    app = PyWorkApp()
    panel = FakeTaskPanel()
    app.task_progress_panel = panel  # type: ignore[assignment]

    monkeypatch.setattr(
        app,
        "resolve_runtime_task_manager",
        lambda: OneTaskManager(),
    )
    monkeypatch.setattr(
        app,
        "resolve_runtime_subagent_manager",
        lambda: FakeSubAgentManager(),
    )

    await app.refresh_task_panel()

    task_ids = {
        row.task_id
        for row in panel.snapshot.rows
    }

    assert task_ids == {"task_new", "reviewer_run_1"}


@pytest.mark.asyncio
async def test_find_task_by_id_falls_back_to_subagent_history(monkeypatch) -> None:
    app = PyWorkApp()

    monkeypatch.setattr(
        app,
        "resolve_runtime_task_manager",
        lambda: EmptyTaskManager(),
    )
    monkeypatch.setattr(
        app,
        "resolve_runtime_subagent_manager",
        lambda: FakeSubAgentManager(),
    )

    task = await app.find_task_by_id("reviewer_run_1")

    assert task is not None
    assert task["run_id"] == "reviewer_run_1"


@pytest.mark.asyncio
async def test_subagent_history_detail_uses_run_id_and_completed_title(monkeypatch) -> None:
    app = PyWorkApp()

    monkeypatch.setattr(
        app,
        "resolve_runtime_task_manager",
        lambda: EmptyTaskManager(),
    )
    monkeypatch.setattr(
        app,
        "resolve_runtime_subagent_manager",
        lambda: FakeSubAgentManager(),
    )

    task = await app.find_task_by_id("reviewer_run_1")
    detail = app.render_task_detail_text(task)

    assert "reviewer_run_1" in detail
    assert "Reviewer 已审查 diff.py" in detail
    assert "正在审查" not in detail.splitlines()[0]


def test_tui_task_event_deduplicates_event_id(monkeypatch) -> None:
    app = PyWorkApp()

    scheduled = []

    def fake_schedule() -> None:
        scheduled.append(True)

    monkeypatch.setattr(
        app,
        "schedule_task_panel_refresh",
        fake_schedule,
    )

    event = RuntimeEvent.status_event(
        status="task_finished",
        source=RuntimeEventSource.SYSTEM,
        metadata={
            "task_event": True,
            "task_id": "task_1",
        },
    )

    app.handle_task_runtime_event(event)
    app.handle_task_runtime_event(event)

    assert scheduled == [True]
