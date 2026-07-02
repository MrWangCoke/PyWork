from __future__ import annotations

import pytest

from pywork.tui.components.input_box import InputBoxDemoApp


@pytest.mark.asyncio
async def test_input_box_submit_bubbles_to_demo_app() -> None:
    app = InputBoxDemoApp()

    async with app.run_test() as pilot:
        await pilot.click("#prompt-input")
        await pilot.press("h", "e", "l", "l", "o")
        await pilot.press("enter")
        await pilot.pause()

        input_area = app.query_one("#prompt-input")

        assert "[" in app.demo_log_text
        assert "User submitted:\nhello" in app.demo_log_text
        assert "hello" in app.export_screenshot()
        assert str(input_area.text) == ""
        assert app.focused is input_area


@pytest.mark.asyncio
async def test_input_box_demo_log_scrolls_to_latest_submission() -> None:
    app = InputBoxDemoApp()

    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.click("#prompt-input")

        for index in range(12):
            text = f"message-{index}"
            await pilot.press(*text)
            await pilot.press("enter")

        await pilot.pause()

        screenshot = app.export_screenshot()

        assert "message-11" in app.demo_log_text
        assert "message-11" in screenshot
        assert app.query_one("#prompt-input").size.height >= 3
        assert str(app.query_one("#prompt-input").text) == ""


@pytest.mark.asyncio
async def test_input_box_leaves_bottom_gutter_for_terminal_rendering() -> None:
    app = InputBoxDemoApp()

    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()

        prompt_input = app.query_one("#prompt-input")

        assert prompt_input.region.y + prompt_input.region.height < app.size.height
