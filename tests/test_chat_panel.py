from __future__ import annotations

from rich.text import Text

from pywork.tui.components.chat_panel import ChatMessage, MessageBubble


def test_message_bubble_renders_selectable_text() -> None:
    message = ChatMessage(
        role="assistant",
        content="hello selectable output",
    )
    bubble = MessageBubble(message)

    rendered = bubble.render_message()

    assert isinstance(rendered, Text)
    assert "hello selectable output" in rendered.plain
