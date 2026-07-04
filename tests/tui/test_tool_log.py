from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from pywork.tui.components.tool_log import ToolLog


class ToolLogTestApp(App[None]):
    def compose(self) -> ComposeResult:
        yield ToolLog(id="tool-log")


@pytest.mark.asyncio
async def test_tool_log_renders_tool_result_as_plain_text() -> None:
    app = ToolLogTestApp()

    async with app.run_test():
        tool_log = app.query_one("#tool-log", ToolLog)

        tool_log.append_tool_result(
            {
                "tool_name": "agent",
                "call_id": "call_1",
                "result_id": "result_1",
                "success": True,
                "content": "review output",
                "data": {
                    "source": (
                        "@dataclass(frozen=True)\n"
                        "class DiffFileStat:\n"
                        "    value = [not rich markup\n"
                    )
                },
            }
        )

        assert "DiffFileStat" in tool_log.render_log()


def test_tool_log_to_text_exports_current_log() -> None:
    tool_log = ToolLog()
    tool_log.append_status("runtime connected")

    text = tool_log.to_text()

    assert "runtime connected" in text
