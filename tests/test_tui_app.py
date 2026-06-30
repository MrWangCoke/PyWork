from __future__ import annotations

import pytest
from textual.events import Key

from pywork.tui.app import PyWorkApp


@pytest.mark.asyncio
async def test_pywork_app_layout_keeps_chat_input_and_status_visible() -> None:
    app = PyWorkApp(
        config={
            "default": {
                "model": "deepseek-v4-flash",
                "provider": "deepseek",
            },
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        chat_panel = app.query_one("#chat-panel")
        input_box = app.query_one("#input-box")
        prompt_input = app.query_one("#prompt-input")
        status_bar = app.query_one("#status-bar")

        assert chat_panel.size.height > input_box.size.height
        assert input_box.region.y + input_box.region.height <= status_bar.region.y
        assert prompt_input.region.y + prompt_input.region.height < status_bar.region.y


@pytest.mark.asyncio
async def test_pywork_app_submission_is_visible_and_clears_input() -> None:
    app = PyWorkApp(
        config={
            "default": {
                "model": "deepseek-v4-flash",
                "provider": "deepseek",
            },
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.click("#prompt-input")
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("ctrl+j")
        await pilot.pause()

        screenshot = app.export_screenshot()

        assert "hello" in screenshot
        assert str(app.query_one("#prompt-input").text) == ""
        assert "ready" in screenshot


@pytest.mark.asyncio
async def test_pywork_app_ctrl_r_resets_tokens_with_visible_feedback() -> None:
    app = PyWorkApp(
        config={
            "default": {
                "model": "deepseek-v4-flash",
                "provider": "deepseek",
            },
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        status_bar = app.query_one("#status-bar")
        status_bar.add_token_usage(input_tokens=1536, output_tokens=768)

        await pilot.press("ctrl+r")
        await pilot.pause()

        screenshot = app.export_screenshot()
        chat_panel = app.query_one("#chat-panel")

        assert status_bar.total_tokens == 0
        assert "0/0/0" in screenshot
        assert any(message.content == "Token usage reset." for message in chat_panel.messages)


@pytest.mark.asyncio
async def test_pywork_app_ctrl_r_key_event_fallback_resets_tokens() -> None:
    app = PyWorkApp(
        config={
            "default": {
                "model": "deepseek-v4-flash",
                "provider": "deepseek",
            },
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        status_bar = app.query_one("#status-bar")
        status_bar.add_token_usage(input_tokens=1536, output_tokens=768)

        app.on_key(Key("ctrl+r", None))
        await pilot.pause()

        assert status_bar.total_tokens == 0
        assert "0/0/0" in app.export_screenshot()


@pytest.mark.asyncio
async def test_pywork_app_reset_tokens_slash_command() -> None:
    app = PyWorkApp(
        config={
            "default": {
                "model": "deepseek-v4-flash",
                "provider": "deepseek",
            },
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        status_bar = app.query_one("#status-bar")
        status_bar.add_token_usage(input_tokens=1536, output_tokens=768)

        await pilot.click("#prompt-input")
        await pilot.press("/", "r", "e", "s", "e", "t", "-", "t", "o", "k", "e", "n", "s")
        await pilot.press("ctrl+j")
        await pilot.pause()

        assert status_bar.total_tokens == 0
        assert str(app.query_one("#prompt-input").text) == ""


@pytest.mark.asyncio
async def test_pywork_app_does_not_intercept_tool_slash_command() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 30)):
        assert app.handle_slash_command("/tool echo hello") is False


@pytest.mark.asyncio
async def test_pywork_app_status_bar_uses_qwen_runtime_model() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 30)):
        status_bar = app.query_one("#status-bar")

        assert status_bar.model == "qwen3.6-flash/qwen"
        assert status_bar.provider == "qwen"
