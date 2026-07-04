from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pywork.tui.app import PyWorkApp


class FakeChatPanel:
    def __init__(self) -> None:
        self.system_messages: list[str] = []
        self.error_messages: list[str] = []

    def append_system_message(self, text: str, *args, **kwargs) -> None:
        self.system_messages.append(text)

    def append_error_message(self, text: str, *args, **kwargs) -> None:
        self.error_messages.append(text)


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.errors: list[str] = []

    def set_idle(self, message: str = "") -> None:
        self.messages.append(message)

    def set_error(self, message: str = "") -> None:
        self.errors.append(message)


class FakeInputBox:
    def __init__(self) -> None:
        self.focus_count = 0

    def focus_input(self) -> None:
        self.focus_count += 1


@dataclass
class FakeTask:
    id: str
    name: str
    status: str = "running"
    agent_id: str = "planner"
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "agent_id": self.agent_id,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class FakeTaskManager:
    def __init__(self) -> None:
        self.tasks = [
            FakeTask(
                id="task_1",
                name="规划实现方案",
                result={
                    "content": "planner result",
                },
            ),
            FakeTask(
                id="task_2",
                name="审查代码",
                status="succeeded",
                agent_id="reviewer",
            ),
        ]
        self.cancelled: list[str] = []

    async def list_tasks(self, *args, **kwargs):
        return self.tasks

    def get_active_task_ids(self):
        return {
            task.id
            for task in self.tasks
            if task.is_active
        }

    async def get_task(self, task_id: str):
        for task in self.tasks:
            if task.id == task_id:
                return task

        raise KeyError(task_id)

    async def cancel_task(self, task_id: str, *args, **kwargs):
        self.cancelled.append(task_id)
        return FakeTask(
            id=task_id,
            name="cancelled task",
            status="cancelled",
        )


class FakeSubAgentManager:
    def __init__(self, task_manager: FakeTaskManager) -> None:
        self.task_manager = task_manager
        self.abort_all_count = 0

    def get_active_runs(self):
        return [
            {
                "run_id": "run_1",
                "agent_name": "planner",
                "status": "running",
                "task": "规划实现方案",
            }
        ]

    def get_history(self, limit=None):
        return [
            {
                "run_id": "run_done",
                "agent_name": "reviewer",
                "status": "completed",
            }
        ]

    async def cancel_agent_task(self, task_id: str, *args, **kwargs):
        return await self.task_manager.cancel_task(task_id)

    def abort_all(self, *args, **kwargs):
        self.abort_all_count += 1
        return 1


class FakeTeamMailbox:
    def count_messages(self):
        return 2

    def list_messages(self, include_deleted=False):
        return []


class FakeTeam:
    team_id = "team_1"
    name = "Team One"
    description = "test team"
    metadata = {}

    def __init__(self):
        self.mailbox = FakeTeamMailbox()

    def list_members(self):
        return []

    def list_shared_tasks(self, *args, **kwargs):
        return []


class FakeAppState:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeController:
    def __init__(self, metadata):
        self.app_state = FakeAppState(metadata)
        self.engine = None
        self.abort_count = 0

    def abort(self):
        self.abort_count += 1


def make_app() -> tuple[PyWorkApp, FakeTaskManager]:
    task_manager = FakeTaskManager()
    manager = FakeSubAgentManager(task_manager)
    team = FakeTeam()

    app = PyWorkApp()
    app.chat_panel = FakeChatPanel()
    app.status_bar = FakeStatusBar()
    app.input_box = FakeInputBox()
    app.runtime_controller = FakeController(
        {
            "task_manager": task_manager,
            "subagent_manager": manager,
            "team": team,
        }
    )

    return app, task_manager


def test_known_slash_commands_with_arguments() -> None:
    app = PyWorkApp()

    assert app.is_known_slash_command_text("/tasks") is True
    assert app.is_known_slash_command_text("/agents") is True
    assert app.is_known_slash_command_text("/team") is True
    assert app.is_known_slash_command_text("/task task_1") is True
    assert app.is_known_slash_command_text("/task-output task_1") is True
    assert app.is_known_slash_command_text("/stop task_1") is True
    assert app.is_known_slash_command_text("/abort") is True


def test_slash_commands_allowed_while_runtime_busy() -> None:
    app = PyWorkApp()

    assert app.is_slash_command_allowed_while_busy("/tasks") is True
    assert app.is_slash_command_allowed_while_busy("/task task_1") is True
    assert app.is_slash_command_allowed_while_busy("/abort") is True
    assert app.is_slash_command_allowed_while_busy("/tool echo hello") is False
    assert app.is_slash_command_allowed_while_busy("/clear") is False


@pytest.mark.asyncio
async def test_run_tasks_command() -> None:
    app, _task_manager = make_app()

    await app.run_tasks_command()

    assert "Background tasks:" in app.chat_panel.system_messages[-1]
    assert "task_1" in app.chat_panel.system_messages[-1]
    assert "task_2" in app.chat_panel.system_messages[-1]


@pytest.mark.asyncio
async def test_run_agents_command() -> None:
    app, _task_manager = make_app()

    await app.run_agents_command()

    assert "SubAgents:" in app.chat_panel.system_messages[-1]
    assert "run_1" in app.chat_panel.system_messages[-1]


@pytest.mark.asyncio
async def test_run_team_command() -> None:
    app, _task_manager = make_app()

    await app.run_team_command()

    assert "Team `Team One`" in app.chat_panel.system_messages[-1]
    assert "Mailbox:" in app.chat_panel.system_messages[-1]


@pytest.mark.asyncio
async def test_run_task_detail_command() -> None:
    app, _task_manager = make_app()

    await app.run_task_detail_command("task_1")

    assert "Task ID: `task_1`" in app.chat_panel.system_messages[-1]
    assert "规划实现方案" in app.chat_panel.system_messages[-1]


@pytest.mark.asyncio
async def test_run_task_output_command() -> None:
    app, _task_manager = make_app()

    await app.run_task_detail_command(
        "task_1",
        output_only=True,
    )

    assert "Task output `task_1`" in app.chat_panel.system_messages[-1]
    assert "planner result" in app.chat_panel.system_messages[-1]


@pytest.mark.asyncio
async def test_run_stop_task_command() -> None:
    app, task_manager = make_app()

    await app.run_stop_task_command("task_1")

    assert "task_1" in task_manager.cancelled
    assert "Stop requested" in app.chat_panel.system_messages[-1]


def test_abort_runtime_command() -> None:
    app, _task_manager = make_app()
    controller = app.runtime_controller
    manager = controller.app_state.metadata["subagent_manager"]

    assert app.handle_slash_command("/abort") is True

    assert controller.abort_count == 1
    assert manager.abort_all_count == 1
    assert "Abort requested" in app.chat_panel.system_messages[-1]
