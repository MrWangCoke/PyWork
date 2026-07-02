from __future__ import annotations

from pywork.tui.app import PyWorkApp
from pywork.teams import create_team


class FakeAppState:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeEngine:
    def __init__(self, *, team=None, runtime_metadata=None):
        self.team = team
        self.runtime_metadata = runtime_metadata or {}


class FakeController:
    def __init__(self, *, app_state=None, engine=None):
        self.app_state = app_state
        self.engine = engine


def test_resolve_runtime_team_from_app_state_metadata() -> None:
    team = create_team(
        team_id="team_test",
        name="Test Team",
    )

    app = PyWorkApp()
    app.runtime_controller = FakeController(
        app_state=FakeAppState(
            {
                "team": team,
            }
        ),
        engine=FakeEngine(),
    )

    assert app.resolve_runtime_team() is team


def test_resolve_runtime_team_from_engine() -> None:
    team = create_team(
        team_id="team_test",
        name="Test Team",
    )

    app = PyWorkApp()
    app.runtime_controller = FakeController(
        app_state=FakeAppState({}),
        engine=FakeEngine(
            team=team,
        ),
    )

    assert app.resolve_runtime_team() is team


def test_resolve_runtime_team_from_team_registry() -> None:
    team = create_team(
        team_id="team_test",
        name="Test Team",
    )

    app = PyWorkApp()
    app.runtime_controller = FakeController(
        app_state=FakeAppState(
            {
                "team_registry": {
                    team.team_id: team,
                }
            }
        ),
        engine=FakeEngine(),
    )

    assert app.resolve_runtime_team() is team


def test_resolve_runtime_team_returns_none_without_runtime() -> None:
    app = PyWorkApp()

    assert app.resolve_runtime_team() is None