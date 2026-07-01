from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from textual import events, on
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Log, Static, TextArea


SUBMIT_KEYS = {
    "enter",
}


CLEAR_KEYS = {
    "escape",
}


@dataclass(frozen=True)
class SubmittedInput:
    text: str
    raw_text: str
    submitted_at: datetime


class InputSubmitted(Message):
    """
    InputBox 提交消息。

    上层 App 可以这样监听：

        def on_input_submitted(self, message: InputSubmitted) -> None:
            print(message.value.text)
    """
    # This must bubble so the parent App can handle the submitted text.
    bubble = True

    def __init__(self, value: SubmittedInput) -> None:
        self.value = value
        super().__init__()


class SubmitRequested(Message):
    """
    SubmitTextArea 内部事件。

    TextArea 捕获 Enter 后，
    先发给 InputBox，再由 InputBox 统一处理提交。
    """

    # This must bubble from SubmitTextArea to InputBox.
    bubble = True


class ClearRequested(Message):
    """
    SubmitTextArea 内部事件。

    TextArea 捕获 Esc 后，
    先发给 InputBox，再由 InputBox 统一处理清空。
    """

    # This must bubble from SubmitTextArea to InputBox.
    bubble = True


class SubmitTextArea(TextArea):
    """
    支持快捷键的多行输入框。

    默认行为：
    - Enter：提交
    - Esc：清空
    """

    DEFAULT_CSS = """
    SubmitTextArea {
        height: 7;
        min-height: 3;
        max-height: 10;
        border: round $accent;
        padding: 0 1;
    }

    SubmitTextArea:focus {
        border: round $success;
    }
    """

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()

        if key in SUBMIT_KEYS:
            event.prevent_default()
            event.stop()
            self.post_message(SubmitRequested())
            return

        if key in CLEAR_KEYS:
            event.prevent_default()
            event.stop()
            self.post_message(ClearRequested())
            return


class InputBox(Widget):
    """
    PyWork TUI 输入组件。

    主要职责：
    1. 管理多行输入框
    2. 提交用户输入
    3. 清空输入
    4. 给上层 App 发送 InputSubmitted 消息
    """

    DEFAULT_CSS = """
    InputBox {
        height: 12;
        border-top: solid $primary;
        padding: 1;
    }

    #input-help {
        height: 1;
        color: $text-muted;
        margin-bottom: 1;
    }

    #input-error {
        height: 1;
        color: $error;
        display: none;
    }

    #input-error.visible {
        display: block;
    }
    """

    def __init__(
        self,
        *,
        initial_text: str = "",
        submit_empty: bool = False,
        clear_on_submit: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.initial_text = initial_text
        self.submit_empty = submit_empty
        self.clear_on_submit = clear_on_submit

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(
                "Enter 提交 · Esc 清空",
                id="input-help",
            )
            yield SubmitTextArea(
                self.initial_text,
                id="prompt-input",
            )
            yield Static(
                "",
                id="input-error",
            )

    def on_mount(self) -> None:
        self.focus_input()

    def on_submit_requested(self, message: SubmitRequested) -> None:
        message.stop()
        self.submit()

    def on_clear_requested(self, message: ClearRequested) -> None:
        message.stop()
        self.clear()

    def focus_input(self) -> None:
        if not self.is_mounted:
            return

        text_area = self.get_text_area()
        text_area.focus()

    def get_text_area(self) -> SubmitTextArea:
        return self.query_one("#prompt-input", SubmitTextArea)

    def get_text(self) -> str:
        text_area = self.get_text_area()
        return str(text_area.text)

    def set_text(self, text: str) -> None:
        text_area = self.get_text_area()

        if hasattr(text_area, "load_text"):
            text_area.load_text(text)
        else:
            text_area.text = text

    def clear(self) -> None:
        self.set_text("")
        self.hide_error()
        self.focus_input()

    def submit(self) -> bool:
        raw_text = self.get_text()
        text = raw_text.strip()

        if not text and not self.submit_empty:
            self.show_error("输入为空，未提交。")
            self.focus_input()
            return False

        self.hide_error()

        submitted = SubmittedInput(
            text=text,
            raw_text=raw_text,
            submitted_at=datetime.now(),
        )

        self.post_message(InputSubmitted(submitted))

        if self.clear_on_submit:
            self.clear()

        self.focus_input()
        return True

    def show_error(self, message: str) -> None:
        error = self.query_one("#input-error", Static)
        error.update(message)
        error.add_class("visible")

    def hide_error(self) -> None:
        error = self.query_one("#input-error", Static)
        error.update("")
        error.remove_class("visible")


class InputBoxDemoApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #demo-log {
        height: 1fr;
        border: round $primary;
    }

    InputBox {
        height: 12;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.demo_log_text = (
            "InputBox demo started.\n\n"
            "输入内容后按 Enter 提交。\n"
            "按 q 退出。"
        )

    def compose(self) -> ComposeResult:
        yield Log(id="demo-log")
        yield InputBox(id="input-box")

    def on_mount(self) -> None:
        log = self.query_one("#demo-log", Log)
        for line in self.demo_log_text.splitlines():
            log.write_line(line)

    @on(InputSubmitted)
    def handle_input_submitted(self, message: InputSubmitted) -> None:
        log = self.query_one("#demo-log", Log)

        self.demo_log_text += (
            "\n\n"
            f"[{message.value.submitted_at.strftime('%H:%M:%S')}] User submitted:\n"
            f"{message.value.text}"
        )

        log.write_line("", scroll_end=True)
        log.write_line(
            f"[{message.value.submitted_at.strftime('%H:%M:%S')}] User submitted:",
            scroll_end=True,
        )
        for line in message.value.text.splitlines() or [""]:
            log.write_line(line, scroll_end=True)

def main() -> int:
    app = InputBoxDemoApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
