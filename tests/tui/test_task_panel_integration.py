from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pywork.tui.app import PyWorkApp


@dataclass
class FakeTaskManager:
    name: str = "task_manager"


class FakeSubAgentManager:
    def __init__(self, task_manager: FakeTaskManager) -> None:
        self.task_manager = task_manager


class FakeAgentTool:
    def __init__(self, task_manager: FakeTaskManager) -> None:
        self.manager = FakeSubAgentManager(task_manager)


class FakeFallbackRuntime:
    def __init__(self, task_manager: FakeTaskManager) -> None:
        self.manager = FakeSubAgentManager(task_manager)


class FakeAgentToolWithFallback:
    def __init__(self, task_manager: FakeTaskManager) -> None:
        self.manager = None
        self._fallback_runtime = FakeFallbackRuntime(task_manager)


class FakeRegistry:
    def __init__(self, tool: Any) -> None:
        self.tool = tool

    def get(self, name: str) -> Any | None:
        if name == "agent":
            return self.tool

        return None


class FakeEngine:
    def __init__(self, registry: FakeRegistry | None = None) -> None:
        self.registry = registry


class FakeAppState:
    def __init__(self, metadata: dict[str, Any]) -> None:
        self.metadata = metadata


class FakeController:
    def __init__(
        self,
        *,
        engine: FakeEngine | None = None,
        app_state: FakeAppState | None = None,
    ) -> None:
        self.engine = engine
        self.app_state = app_state


def test_resolve_runtime_task_manager_from_app_state_metadata() -> None:
    task_manager = FakeTaskManager()
    app = PyWorkApp()

    app.runtime_controller = FakeController(
        app_state=FakeAppState(
            {
                "task_manager": task_manager,
            }
        ),
        engine=FakeEngine(),
    )

    assert app.resolve_runtime_task_manager() is task_manager


def test_resolve_runtime_task_manager_from_subagent_manager_metadata() -> None:
    task_manager = FakeTaskManager()
    app = PyWorkApp()

    app.runtime_controller = FakeController(
        app_state=FakeAppState(
            {
                "subagent_manager": FakeSubAgentManager(task_manager),
            }
        ),
        engine=FakeEngine(),
    )

    assert app.resolve_runtime_task_manager() is task_manager


def test_resolve_runtime_task_manager_from_agent_tool_manager() -> None:
    task_manager = FakeTaskManager()
    app = PyWorkApp()

    app.runtime_controller = FakeController(
        engine=FakeEngine(
            FakeRegistry(
                FakeAgentTool(task_manager)
            )
        )
    )

    assert app.resolve_runtime_task_manager() is task_manager


def test_resolve_runtime_task_manager_from_agent_tool_fallback_runtime() -> None:
    task_manager = FakeTaskManager()
    app = PyWorkApp()

    app.runtime_controller = FakeController(
        engine=FakeEngine(
            FakeRegistry(
                FakeAgentToolWithFallback(task_manager)
            )
        )
    )

    assert app.resolve_runtime_task_manager() is task_manager


def test_resolve_runtime_task_manager_returns_none_without_runtime() -> None:
    app = PyWorkApp()

    assert app.resolve_runtime_task_manager() is None