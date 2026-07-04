from __future__ import annotations

from pywork.tui.app import PyWorkApp


class FakeSubmitted:
    def __init__(self, text: str) -> None:
        self.text = text
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


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
        self.idle_messages: list[str] = []
        self.errors: list[str] = []

    def set_idle(self, message: str = "") -> None:
        self.idle_messages.append(message)

    def set_error(self, message: str = "") -> None:
        self.errors.append(message)


class FakeInputBox:
    def focus_input(self) -> None:
        pass


class FakeRuntimeController:
    def __init__(self) -> None:
        self.aborted = False

    def abort(self) -> None:
        self.aborted = True


def test_runtime_busy_allows_readonly_and_control_slash_commands(monkeypatch) -> None:
    app = PyWorkApp()
    app.runtime_busy = True

    handled: list[str] = []

    def fake_handle_slash_command(text: str) -> bool:
        handled.append(text)
        return True

    monkeypatch.setattr(
        app,
        "handle_slash_command",
        fake_handle_slash_command,
    )

    for command in (
        "/status",
        "/tasks",
        "/agents",
        "/team",
        "/task task_1",
        "/task-output task_1",
        "/stop task_1",
        "/abort",
    ):
        app.on_input_submitted(FakeSubmitted(command))

    assert handled == [
        "/status",
        "/tasks",
        "/agents",
        "/team",
        "/task task_1",
        "/task-output task_1",
        "/stop task_1",
        "/abort",
    ]


def test_runtime_busy_blocks_regular_chat_input() -> None:
    app = PyWorkApp()
    app.runtime_busy = True
    app.chat_panel = FakeChatPanel()

    app.on_input_submitted(
        FakeSubmitted("请帮我继续实现功能")
    )

    assert app.chat_panel.system_messages
    assert "runtime task is still running" in app.chat_panel.system_messages[-1]


def test_runtime_busy_blocks_tool_runtime_command() -> None:
    app = PyWorkApp()
    app.runtime_busy = True
    app.chat_panel = FakeChatPanel()
    app.status_bar = FakeStatusBar()

    app.on_input_submitted(
        FakeSubmitted("/tool echo hello")
    )

    assert app.chat_panel.system_messages
    assert "not available while the runtime is busy" in app.chat_panel.system_messages[-1]


def test_abort_command_calls_runtime_controller_abort() -> None:
    app = PyWorkApp()
    controller = FakeRuntimeController()

    app.runtime_controller = controller
    app.chat_panel = FakeChatPanel()
    app.status_bar = FakeStatusBar()
    app.input_box = FakeInputBox()

    app.action_abort_runtime()

    assert controller.aborted is True
    assert "Abort requested" in app.chat_panel.system_messages[-1]