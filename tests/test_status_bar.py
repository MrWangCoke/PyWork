from __future__ import annotations

import pytest

from pywork.tui.components.status_bar import StatusBarDemoApp


@pytest.mark.asyncio
async def test_status_bar_demo_key_bindings_update_visible_status() -> None:
    app = StatusBarDemoApp()

    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.press("t")
        await pilot.press("2")
        await pilot.pause()

        status_bar = app.query_one("#status-bar")
        screenshot = app.export_screenshot()

        assert status_bar.total_tokens == 192
        assert status_bar.state == "thinking"
        assert "192" in screenshot
        assert "thinking" in screenshot


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "state"),
    [
        ("1", "idle"),
        ("2", "thinking"),
        ("3", "running_tool"),
        ("4", "error"),
    ],
)
async def test_status_bar_state_changes_are_visible_in_narrow_layout(
    key: str,
    state: str,
) -> None:
    app = StatusBarDemoApp()

    async with app.run_test(size=(80, 16)) as pilot:
        await pilot.press(key)
        await pilot.pause()

        status_bar = app.query_one("#status-bar")
        screenshot = app.export_screenshot()

        assert status_bar.state == state
        assert state in screenshot


@pytest.mark.asyncio
async def test_status_bar_has_visible_content_height() -> None:
    app = StatusBarDemoApp()

    async with app.run_test(size=(120, 24)) as pilot:
        await pilot.pause()

        status_bar = app.query_one("#status-bar")
        status_line = app.query_one("#status-line")

        assert status_bar.size.height >= 1
        assert status_line.size.height == 1
        assert status_line.region.y + status_line.region.height <= app.size.height
