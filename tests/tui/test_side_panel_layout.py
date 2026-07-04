from __future__ import annotations

from pywork.tui.app import (
    SIDE_PANEL_AGENTS,
    SIDE_PANEL_TASKS,
    SIDE_PANEL_TEAM,
    SIDE_PANEL_TOOL_LOG,
    PyWorkApp,
    normalize_side_panel_view,
)


def test_normalize_side_panel_view_aliases() -> None:
    assert normalize_side_panel_view("1") == SIDE_PANEL_TOOL_LOG
    assert normalize_side_panel_view("log") == SIDE_PANEL_TOOL_LOG
    assert normalize_side_panel_view("tool-log") == SIDE_PANEL_TOOL_LOG

    assert normalize_side_panel_view("2") == SIDE_PANEL_TASKS
    assert normalize_side_panel_view("task") == SIDE_PANEL_TASKS
    assert normalize_side_panel_view("tasks") == SIDE_PANEL_TASKS

    assert normalize_side_panel_view("3") == SIDE_PANEL_AGENTS
    assert normalize_side_panel_view("agent") == SIDE_PANEL_AGENTS
    assert normalize_side_panel_view("agents") == SIDE_PANEL_AGENTS

    assert normalize_side_panel_view("4") == SIDE_PANEL_TEAM
    assert normalize_side_panel_view("team") == SIDE_PANEL_TEAM
    assert normalize_side_panel_view("mailbox") == SIDE_PANEL_TEAM


def test_default_side_panel_is_tool_log() -> None:
    app = PyWorkApp()

    assert app.active_side_panel_view == SIDE_PANEL_TOOL_LOG


def test_set_side_panel_view_normalizes_alias(monkeypatch) -> None:
    app = PyWorkApp()

    applied = []

    def fake_apply() -> None:
        applied.append(app.active_side_panel_view)

    monkeypatch.setattr(
        app,
        "apply_side_panel_view",
        fake_apply,
    )
    monkeypatch.setattr(app, "schedule_task_panel_refresh", lambda: None)
    monkeypatch.setattr(app, "schedule_agent_panel_refresh", lambda: None)
    monkeypatch.setattr(app, "schedule_team_panel_refresh", lambda: None)

    app.set_side_panel_view("tasks")

    assert app.active_side_panel_view == SIDE_PANEL_TASKS
    assert applied[-1] == SIDE_PANEL_TASKS

    app.set_side_panel_view("team")

    assert app.active_side_panel_view == SIDE_PANEL_TEAM
    assert applied[-1] == SIDE_PANEL_TEAM


def test_render_side_tabs_marks_active_panel() -> None:
    app = PyWorkApp()

    app.active_side_panel_view = SIDE_PANEL_TASKS

    rendered = app.render_side_tabs()

    assert "[2 Tasks]" in rendered
    assert "1 Tool Log" in rendered