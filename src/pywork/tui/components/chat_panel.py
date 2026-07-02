from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static


MessageRole = Literal[
    "user",
    "assistant",
    "system",
    "tool",
    "error",
]


ROLE_LABELS: dict[MessageRole, str] = {
    "user": "User",
    "assistant": "PyWork",
    "system": "System",
    "tool": "Tool",
    "error": "Error",
}


ROLE_BORDER_STYLES: dict[MessageRole, str] = {
    "user": "cyan",
    "assistant": "green",
    "system": "yellow",
    "tool": "magenta",
    "error": "red",
}


@dataclass
class ChatMessage:
    role: MessageRole
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: uuid4().hex)
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBubble(Static):
    """
    单条消息气泡。

    后续可以继续增强：
    - 代码块高亮
    - 工具调用块
    - diff 渲染
    - 流式输出更新
    """

    DEFAULT_CSS = """
    MessageBubble {
        width: 100%;
        margin-bottom: 1;
        padding: 0 1;
        border: round white;
    }

    MessageBubble.role-user {
        border: round cyan;
    }

    MessageBubble.role-assistant {
        border: round green;
    }

    MessageBubble.role-system {
        border: round yellow;
    }

    MessageBubble.role-tool {
        border: round magenta;
    }

    MessageBubble.role-error {
        border: round red;
    }
    """

    def __init__(
        self,
        message: ChatMessage,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        role_class = f"role-{message.role}"
        merged_classes = f"{classes or ''} {role_class}".strip()

        super().__init__("", id=id, classes=merged_classes)
        self.message = message

    def on_mount(self) -> None:
        self.refresh_from_message()

    def refresh_from_message(self) -> None:
        self.update(self.render_message())

    def render_message(self) -> Text:
        role = self.message.role
        label = ROLE_LABELS.get(role, role)
        label_style = ROLE_BORDER_STYLES.get(role, "white")

        time_text = self.message.created_at.strftime("%H:%M:%S")
        title = f"{label} · {time_text}"

        content = self.message.content.strip() or " "

        text = Text()
        text.append(title, style=f"bold {label_style}")
        text.append("\n")

        if role == "error":
            text.append(content, style="red")
        else:
            text.append(content)

        return text


class ChatPanel(Widget):
    """
    消息显示面板。

    主要职责：
    1. 保存消息列表
    2. 渲染 user / assistant / system / tool / error 消息
    3. 自动滚动到底部
    4. 支持后续流式更新最后一条 assistant 消息
    """

    DEFAULT_CSS = """
    ChatPanel {
        height: 1fr;
    }

    #chat-scroll {
        height: 1fr;
        padding: 1;
        border: round $primary;
    }

    .empty-chat {
        color: $text-muted;
    }
    """

    def __init__(
        self,
        *,
        show_welcome: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.show_welcome = show_welcome
        self._messages: list[ChatMessage] = []
        self._message_widgets: dict[str, MessageBubble] = {}

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="chat-scroll"):
            if self.show_welcome:
                yield Static(
                    "PyWork TUI ready. Type something below to start.",
                    classes="empty-chat",
                )

    def on_mount(self) -> None:
        if self._messages:
            self._clear_scroll()
            for message in self._messages:
                self._mount_message(message)

        self.scroll_to_bottom()

    @property
    def messages(self) -> tuple[ChatMessage, ...]:
        return tuple(self._messages)

    def append_message(
        self,
        role: MessageRole,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        self._messages.append(message)

        if self.is_mounted:
            self._mount_message(message)
            self.scroll_to_bottom()

        return message

    def append_user_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        return self.append_message(
            "user",
            content,
            metadata=metadata,
        )

    def append_assistant_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        return self.append_message(
            "assistant",
            content,
            metadata=metadata,
        )

    def append_system_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        return self.append_message(
            "system",
            content,
            metadata=metadata,
        )

    def append_tool_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        return self.append_message(
            "tool",
            content,
            metadata=metadata,
        )

    def append_error_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        return self.append_message(
            "error",
            content,
            metadata=metadata,
        )

    def update_message(
        self,
        message_id: str,
        content: str,
    ) -> bool:
        for message in self._messages:
            if message.message_id == message_id:
                message.content = content

                widget = self._message_widgets.get(message_id)
                if widget is not None:
                    widget.refresh_from_message()
                    self.scroll_to_bottom()

                return True

        return False

    def append_to_message(
        self,
        message_id: str,
        delta: str,
    ) -> bool:
        for message in self._messages:
            if message.message_id == message_id:
                message.content += delta

                widget = self._message_widgets.get(message_id)
                if widget is not None:
                    widget.refresh_from_message()
                    self.scroll_to_bottom()

                return True

        return False

    def update_last_assistant_message(
        self,
        content: str,
    ) -> bool:
        for message in reversed(self._messages):
            if message.role == "assistant":
                return self.update_message(
                    message.message_id,
                    content,
                )

        return False

    def append_to_last_assistant_message(
        self,
        delta: str,
    ) -> bool:
        for message in reversed(self._messages):
            if message.role == "assistant":
                return self.append_to_message(
                    message.message_id,
                    delta,
                )

        return False

    def clear_messages(self) -> None:
        self._messages.clear()
        self._message_widgets.clear()

        if self.is_mounted:
            self._clear_scroll()

            if self.show_welcome:
                scroll = self._get_scroll()
                scroll.mount(
                    Static(
                        "PyWork TUI ready. Type something below to start.",
                        classes="empty-chat",
                    )
                )

    def scroll_to_bottom(self) -> None:
        if not self.is_mounted:
            return

        def _scroll() -> None:
            scroll = self._get_scroll()
            scroll.scroll_end(animate=False)

        self.call_after_refresh(_scroll)

    def _mount_message(self, message: ChatMessage) -> None:
        self._remove_empty_placeholder()

        bubble = MessageBubble(message)
        self._message_widgets[message.message_id] = bubble

        scroll = self._get_scroll()
        scroll.mount(bubble)

    def _get_scroll(self) -> VerticalScroll:
        return self.query_one("#chat-scroll", VerticalScroll)

    def _clear_scroll(self) -> None:
        scroll = self._get_scroll()

        for child in list(scroll.children):
            child.remove()

    def _remove_empty_placeholder(self) -> None:
        for widget in list(self.query(".empty-chat")):
            widget.remove()


class ChatPanelDemoApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    ChatPanel {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield ChatPanel(id="chat-panel")

    def on_mount(self) -> None:
        chat_panel = self.query_one("#chat-panel", ChatPanel)

        chat_panel.append_system_message("ChatPanel demo started.")
        chat_panel.append_user_message("你好，PyWork。")
        chat_panel.append_assistant_message(
            "你好！这里是 `ChatPanel` 的演示消息。\n\n"
            "它支持 **Markdown**，也支持代码块：\n\n"
            "```python\n"
            "print('hello pywork')\n"
            "```"
        )
        chat_panel.append_tool_message("Tool result example: file read completed.")
        chat_panel.append_error_message("Error example: runtime is not connected yet.")


def main() -> int:
    app = ChatPanelDemoApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
