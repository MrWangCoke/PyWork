from __future__ import annotations

from pywork.runtime.events import RuntimeEvent
from pywork.schemas.tool_schema import create_tool_call
from pywork.tools.task_tools import make_result
from pywork.tui.app import PyWorkApp


class FakeChatPanel:
    def __init__(self) -> None:
        self.system_messages: list[str] = []

    def append_system_message(self, text: str, *args, **kwargs) -> None:
        self.system_messages.append(text)


class FakeStatusBar:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def set_idle(self, text: str = "") -> None:
        self.messages.append(text)


def test_tui_extracts_task_creation_notice_from_task_create_tool_result() -> None:
    app = PyWorkApp()

    result = make_result(
        create_tool_call(
            "task_create",
            {},
        ),
        tool_name="task_create",
        success=True,
        content="Started background task.",
        data={
            "ui_notice": {
                "kind": "background_task_started",
                "target": "subagent",
                "task_id": "task_1",
                "agent": "reviewer",
                "name": "Review diff.py",
                "status": "running",
                "started": True,
                "message": "已创建后台任务，可在右侧 Tasks 面板查看进度。",
            }
        },
    )

    event = RuntimeEvent.tool_result_event(
        tool_result=result,
    )

    notice = app.extract_task_creation_notice_from_event(event)

    assert notice is not None
    assert notice["task_id"] == "task_1"
    assert notice["agent"] == "reviewer"
    assert notice["name"] == "Review diff.py"


def test_tui_shows_task_creation_feedback_in_chat(monkeypatch) -> None:
    app = PyWorkApp()
    app.chat_panel = FakeChatPanel()
    app.status_bar = FakeStatusBar()

    refreshed = []
    switched = []

    monkeypatch.setattr(
        app,
        "schedule_task_panel_refresh",
        lambda: refreshed.append(True),
    )
    monkeypatch.setattr(
        app,
        "set_side_panel_view",
        lambda view: switched.append(view),
    )

    result = make_result(
        create_tool_call(
            "task_create",
            {},
        ),
        tool_name="task_create",
        success=True,
        content="Started background task.",
        data={
            "ui_notice": {
                "kind": "background_task_started",
                "target": "subagent",
                "task_id": "task_1",
                "agent": "reviewer",
                "name": "Review diff.py",
                "status": "running",
                "started": True,
                "message": "已创建后台任务，可在右侧 Tasks 面板查看进度。",
            }
        },
    )

    event = RuntimeEvent.tool_result_event(
        tool_result=result,
    )

    app.handle_task_creation_feedback(event)

    assert refreshed == [True]
    assert app.chat_panel.system_messages
    assert "Started background task:" in app.chat_panel.system_messages[-1]
    assert "task_1" in app.chat_panel.system_messages[-1]
    assert "reviewer" in app.chat_panel.system_messages[-1]
    assert "已创建后台任务" in app.chat_panel.system_messages[-1]


def test_tui_deduplicates_task_creation_feedback(monkeypatch) -> None:
    app = PyWorkApp()
    app.chat_panel = FakeChatPanel()

    monkeypatch.setattr(
        app,
        "schedule_task_panel_refresh",
        lambda: None,
    )
    monkeypatch.setattr(
        app,
        "set_side_panel_view",
        lambda view: None,
    )

    result = make_result(
        create_tool_call(
            "task_create",
            {},
        ),
        tool_name="task_create",
        success=True,
        content="Started background task.",
        data={
            "ui_notice": {
                "task_id": "task_1",
                "agent": "reviewer",
                "name": "Review diff.py",
                "status": "running",
                "started": True,
                "message": "已创建后台任务，可在右侧 Tasks 面板查看进度。",
            }
        },
    )

    event = RuntimeEvent.tool_result_event(
        tool_result=result,
    )

    app.handle_task_creation_feedback(event)
    app.handle_task_creation_feedback(event)

    assert len(app.chat_panel.system_messages) == 1