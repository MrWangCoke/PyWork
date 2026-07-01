from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from pywork.tui.components.diff.models import DiffRenderOptions, DiffStats
from pywork.tui.components.diff.parser import collect_stats, parse_unified_diff
from pywork.tui.components.diff.renderer import render_summary, render_unified_diff


class DiffSummaryBar(Static):
    """Diff 顶部统计条。"""

    DEFAULT_CSS = """
    DiffSummaryBar {
        height: auto;
        padding: 0 1;
        text-style: bold;
    }
    """

    def __init__(
        self,
        stats: DiffStats | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("", **kwargs)
        self.stats = stats or DiffStats()

    def on_mount(self) -> None:
        self.refresh_summary()

    def set_stats(self, stats: DiffStats) -> None:
        self.stats = stats
        self.refresh_summary()

    def refresh_summary(self) -> None:
        self.update(render_summary(self.stats))


class DiffBody(Static):
    """Diff 正文显示区域。"""

    DEFAULT_CSS = """
    DiffBody {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        diff_text: str = "",
        *,
        options: DiffRenderOptions | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__("", **kwargs)
        self.diff_text = diff_text
        self.options = options or DiffRenderOptions(show_header=False)

    def on_mount(self) -> None:
        self.refresh_diff()

    def set_diff(
        self,
        diff_text: str,
        *,
        options: DiffRenderOptions | None = None,
    ) -> None:
        self.diff_text = diff_text

        if options is not None:
            self.options = options

        self.refresh_diff()

    def refresh_diff(self) -> None:
        self.update(
            render_unified_diff(
                self.diff_text,
                options=self.options,
            )
        )


class DiffPanel(VerticalScroll):
    """
    Diff 组合面板。

    由两个子组件组成：
    - DiffSummaryBar
    - DiffBody
    """

    DEFAULT_CSS = """
    DiffPanel {
        height: 1fr;
        border: round $surface;
        padding: 0;
    }

    DiffPanel:focus {
        border: round $accent;
    }
    """

    can_focus = True

    def __init__(
        self,
        diff_text: str = "",
        *,
        title: str | None = None,
        show_summary: bool = True,
        max_lines: int | None = 1_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.diff_text = diff_text
        self.title = title
        self.show_summary = show_summary
        self.max_lines = max_lines

        self.summary_bar = DiffSummaryBar()
        self.body = DiffBody(
            diff_text,
            options=DiffRenderOptions(
                show_header=False,
                show_line_numbers=True,
                max_lines=max_lines,
            ),
        )

    def compose(self) -> ComposeResult:
        if self.title:
            yield Static(
                self.title,
                classes="diff-panel-title",
            )

        if self.show_summary:
            yield self.summary_bar

        yield self.body

    def on_mount(self) -> None:
        self.refresh_diff()

    def set_diff(
        self,
        diff_text: str,
        *,
        title: str | None = None,
        max_lines: int | None = None,
        scroll_top: bool = True,
    ) -> None:
        self.diff_text = diff_text

        if title is not None:
            self.title = title

        if max_lines is not None:
            self.max_lines = max_lines

        self.refresh_diff()

        if scroll_top:
            self.scroll_home(animate=False)

    def clear(self) -> None:
        self.set_diff("")

    def get_stats(self) -> DiffStats:
        lines = parse_unified_diff(self.diff_text)
        return collect_stats(lines)

    def refresh_diff(self) -> None:
        lines = parse_unified_diff(self.diff_text)
        stats = collect_stats(lines)

        self.summary_bar.set_stats(stats)

        self.body.set_diff(
            self.diff_text,
            options=DiffRenderOptions(
                show_header=False,
                show_line_numbers=True,
                max_lines=self.max_lines,
            ),
        )