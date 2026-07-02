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


def test_assistant_message_bubble_renders_model_footer() -> None:
    message = ChatMessage(
        role="assistant",
        content="hello from model",
        metadata={
            "model": "qwen3.6-flash/qwen",
        },
    )
    bubble = MessageBubble(message)

    rendered = bubble.render_message()

    assert "hello from model" in rendered.plain
    assert "model: qwen3.6-flash/qwen" in rendered.plain
