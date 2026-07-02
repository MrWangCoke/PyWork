from __future__ import annotations

import pytest
from textual.events import Key

from pywork.tui.app import CommandsDialog, PyWorkApp


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
        await pilot.press("enter")
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
        assert status_bar.permission_mode == "default"
        assert "dir" in screenshot
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
        assert "dir" in app.export_screenshot()


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
        await pilot.press("enter")
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


@pytest.mark.asyncio
async def test_pywork_app_ctrl_p_shows_commands_dialog() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()

        screenshot = app.export_screenshot()
        status_bar = app.query_one("#status-bar")

        assert "Commands" in screenshot
        assert "exit" in screenshot
        assert "Ctrl+P commands" in status_bar.render_status_line()


def test_pywork_app_exit_only_exposed_through_commands() -> None:
    bindings = {binding.key for binding in PyWorkApp.BINDINGS}

    binding_map = {binding.key: binding.action for binding in PyWorkApp.BINDINGS}

    assert "q" not in bindings
    assert binding_map["ctrl+c"] == "copy_selected_text"
    assert "ctrl+p" in bindings


@pytest.mark.asyncio
async def test_pywork_app_ctrl_c_copies_without_exiting() -> None:
    app = PyWorkApp()
    exited = False
    copied: list[str] = []

    def fake_exit(*args, **kwargs) -> None:
        nonlocal exited
        exited = True

    def fake_copy(text: str) -> None:
        copied.append(text)

    app.exit = fake_exit
    app.copy_to_clipboard = fake_copy

    async with app.run_test(size=(120, 30)):
        app.get_selected_text_for_copy = lambda: "copied text"

        app.action_copy_selected_text()

        assert copied == ["copied text"]
        assert exited is False


@pytest.mark.asyncio
async def test_pywork_app_commands_dialog_searches_exit() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.press("e", "x", "i", "t")
        await pilot.pause()

        screenshot = app.export_screenshot()
        dialog = app.screen

        assert "exit" in screenshot
        assert isinstance(dialog, CommandsDialog)
        assert [command.name for command in dialog.filtered_commands] == ["/exit"]


def test_commands_dialog_displays_plain_command_names() -> None:
    app = PyWorkApp()
    dialog = CommandsDialog(app.get_slash_commands())
    exit_command = [
        command
        for command in app.get_slash_commands()
        if command.name == "/exit"
    ][0]

    row = dialog.render_command_row(exit_command)

    assert row.startswith("exit")
    assert not row.startswith("/exit")


@pytest.mark.asyncio
async def test_pywork_app_commands_dialog_navigation_wraps() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.press("ctrl+p")
        await pilot.pause()

        dialog = app.screen
        assert isinstance(dialog, CommandsDialog)

        list_view = dialog.query_one("#command-list")
        command_count = len(dialog.filtered_commands)

        assert list_view.index == 0

        await pilot.press("up")
        await pilot.pause()

        assert list_view.index == command_count - 1

        await pilot.press("down")
        await pilot.pause()

        assert list_view.index == 0


@pytest.mark.asyncio
async def test_pywork_app_shows_slash_suggestions_above_input() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 34)) as pilot:
        chat_panel = app.query_one("#chat-panel")
        tool_log = app.query_one("#tool-log")
        chat_height = chat_panel.region.height
        tool_log_height = tool_log.region.height

        await pilot.click("#prompt-input")
        await pilot.press("/", "e", "x")
        await pilot.pause()

        screenshot = app.export_screenshot()
        suggestions = app.query_one("#slash-suggestions")
        input_box = app.query_one("#input-box")

        assert suggestions.has_class("visible")
        assert "/exit" in screenshot
        assert len(app.query("#command-suggestions")) == 0
        assert suggestions.region.y + suggestions.region.height <= input_box.region.y
        assert chat_panel.region.height == chat_height
        assert tool_log.region.height == tool_log_height


@pytest.mark.asyncio
async def test_pywork_app_slash_suggestion_enter_executes_selected_command() -> None:
    app = PyWorkApp()

    async with app.run_test(size=(120, 34)) as pilot:
        await pilot.click("#prompt-input")
        await pilot.press("/")
        await pilot.pause()

        assert app.query_one("#slash-suggestions").has_class("visible")

        await pilot.press("enter")
        await pilot.pause()

        messages = [
            message.content
            for message in app.query_one("#chat-panel").messages
        ]

        assert any("PyWork available commands:" in message for message in messages)
        assert not any("Unknown command" in message for message in messages)


def test_pywork_app_command_dialog_exit_result_exits() -> None:
    app = PyWorkApp()
    exited = False

    def fake_exit(*args, **kwargs) -> None:
        nonlocal exited
        exited = True

    app.exit = fake_exit

    app.handle_command_dialog_result("/exit")

    assert exited is True


@pytest.mark.asyncio
async def test_pywork_app_tab_switches_permission_mode() -> None:
    app = PyWorkApp(
        config={
            "permissions": {
                "mode": "default",
            },
        }
    )

    async with app.run_test(size=(120, 30)) as pilot:
        status_bar = app.query_one("#status-bar")
        chat_panel = app.query_one("#chat-panel")
        before_count = len(chat_panel.messages)

        await pilot.press("tab")
        await pilot.pause()

        assert app.get_permission_mode() == "accept_edits"
        assert status_bar.permission_mode == "accept_edits"
        assert app.get_runtime_config()["permissions"]["mode"] == "accept_edits"
        assert "[Tab]" in status_bar.render_status_line()
        assert len(chat_panel.messages) == before_count
