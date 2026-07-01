from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static


DiffLineKind = Literal[
    "file_header",
    "hunk",
    "addition",
    "deletion",
    "context",
    "meta",
    "empty",
]


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)


@dataclass(slots=True, frozen=True)
class DiffViewerLine:
    """
    DiffViewer 内部展示行。

    old_lineno:
        删除前文件的行号。

    new_lineno:
        修改后文件的行号。

    marker:
        diff 标记：
        - "+" 新增
        - "-" 删除
        - " " 上下文
        - ""  文件头 / hunk / meta

    content:
        展示内容，不包含行尾换行符。

    kind:
        当前行类型，用于决定颜色。
    """

    old_lineno: int | None
    new_lineno: int | None
    marker: str
    content: str
    kind: DiffLineKind


@dataclass(slots=True, frozen=True)
class DiffViewerStats:
    """DiffViewer 解析后的简单统计。"""

    additions: int = 0
    deletions: int = 0
    hunks: int = 0
    files: int = 0
    lines: int = 0

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions


def parse_hunk_header(line: str) -> tuple[int, int] | None:
    """
    解析 hunk header。

    示例：
        @@ -1,3 +1,4 @@

    返回：
        (old_start, new_start)
    """
    match = HUNK_HEADER_RE.match(line)

    if match is None:
        return None

    old_start = int(match.group("old_start"))
    new_start = int(match.group("new_start"))

    return old_start, new_start


def parse_unified_diff_for_view(diff_text: str) -> list[DiffViewerLine]:
    """
    把 unified diff 文本解析成可展示的行。

    这里不做 patch 语义验证，只负责 TUI 展示：
    - 识别新增 / 删除 / 上下文 / hunk / 文件头
    - 根据 hunk header 推导 old/new 行号
    """
    if not diff_text.strip():
        return [
            DiffViewerLine(
                old_lineno=None,
                new_lineno=None,
                marker="",
                content="No changes.",
                kind="empty",
            )
        ]

    viewer_lines: list[DiffViewerLine] = []

    old_lineno: int | None = None
    new_lineno: int | None = None

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")

        hunk_position = parse_hunk_header(line)

        if hunk_position is not None:
            old_lineno, new_lineno = hunk_position

            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="hunk",
                )
            )
            continue

        if line.startswith("diff --git "):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="file_header",
                )
            )
            continue

        if line.startswith("--- ") or line.startswith("+++ "):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="file_header",
                )
            )
            continue

        if line.startswith("+") and not line.startswith("+++"):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=None,
                    new_lineno=new_lineno,
                    marker="+",
                    content=line[1:],
                    kind="addition",
                )
            )

            if new_lineno is not None:
                new_lineno += 1

            continue

        if line.startswith("-") and not line.startswith("---"):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=old_lineno,
                    new_lineno=None,
                    marker="-",
                    content=line[1:],
                    kind="deletion",
                )
            )

            if old_lineno is not None:
                old_lineno += 1

            continue

        if line.startswith(" "):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=old_lineno,
                    new_lineno=new_lineno,
                    marker=" ",
                    content=line[1:],
                    kind="context",
                )
            )

            if old_lineno is not None:
                old_lineno += 1

            if new_lineno is not None:
                new_lineno += 1

            continue

        if line.startswith("\\"):
            viewer_lines.append(
                DiffViewerLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="meta",
                )
            )
            continue

        viewer_lines.append(
            DiffViewerLine(
                old_lineno=None,
                new_lineno=None,
                marker="",
                content=line,
                kind="meta",
            )
        )

    return viewer_lines


def collect_diff_viewer_stats(lines: list[DiffViewerLine]) -> DiffViewerStats:
    """统计 DiffViewer 展示行。"""
    additions = sum(1 for line in lines if line.kind == "addition")
    deletions = sum(1 for line in lines if line.kind == "deletion")
    hunks = sum(1 for line in lines if line.kind == "hunk")

    file_headers = [
        line
        for line in lines
        if line.kind == "file_header" and line.content.startswith("+++ ")
    ]

    return DiffViewerStats(
        additions=additions,
        deletions=deletions,
        hunks=hunks,
        files=len(file_headers),
        lines=len(lines),
    )


def line_number_width(lines: list[DiffViewerLine]) -> int:
    """计算 old/new 行号列宽。"""
    max_lineno = 0

    for line in lines:
        if line.old_lineno is not None:
            max_lineno = max(max_lineno, line.old_lineno)

        if line.new_lineno is not None:
            max_lineno = max(max_lineno, line.new_lineno)

    return max(3, len(str(max_lineno)))


def format_lineno(
    value: int | None,
    *,
    width: int,
) -> str:
    """格式化行号，没有行号时用空格占位。"""
    if value is None:
        return " " * width

    return str(value).rjust(width)


def style_for_line(kind: DiffLineKind) -> str:
    """返回 Rich/Textual 样式。"""
    if kind == "addition":
        return "green"

    if kind == "deletion":
        return "red"

    if kind == "hunk":
        return "bold cyan"

    if kind == "file_header":
        return "bold yellow"

    if kind == "meta":
        return "dim"

    if kind == "empty":
        return "dim italic"

    return ""


def render_unified_diff_text(
    diff_text: str,
    *,
    show_header: bool = True,
    max_lines: int | None = None,
) -> Text:
    """
    渲染 unified diff 为 Rich Text。

    输出格式大概是：

        old new │
          1   1 │ context
          2     - deleted
              2 + added
    """
    lines = parse_unified_diff_for_view(diff_text)
    stats = collect_diff_viewer_stats(lines)

    if max_lines is not None and max_lines > 0 and len(lines) > max_lines:
        visible_lines = lines[:max_lines]
        truncated = True
    else:
        visible_lines = lines
        truncated = False

    width = line_number_width(lines)

    text = Text()

    if show_header:
        summary = (
            f"{stats.files} file(s), "
            f"+{stats.additions}, "
            f"-{stats.deletions}, "
            f"{stats.hunks} hunk(s)"
        )

        text.append(summary, style="bold")
        text.append("\n")

        text.append(
            f"{'old'.rjust(width)} {'new'.rjust(width)} │ diff\n",
            style="dim",
        )

    for line in visible_lines:
        line_style = style_for_line(line.kind)

        if line.kind in {"file_header", "hunk", "meta", "empty"}:
            text.append(" " * width, style="dim")
            text.append(" ", style="dim")
            text.append(" " * width, style="dim")
            text.append(" │ ", style="dim")
            text.append(line.content, style=line_style)
            text.append("\n")
            continue

        old_part = format_lineno(
            line.old_lineno,
            width=width,
        )
        new_part = format_lineno(
            line.new_lineno,
            width=width,
        )

        marker = line.marker or " "

        prefix_style = line_style or "dim"

        text.append(old_part, style=prefix_style)
        text.append(" ", style="dim")
        text.append(new_part, style=prefix_style)
        text.append(" ", style="dim")
        text.append(marker, style=prefix_style)
        text.append(" │ ", style="dim")
        text.append(line.content, style=line_style)
        text.append("\n")

    if truncated:
        text.append(
            f"... diff truncated after {max_lines} rendered line(s)\n",
            style="bold red",
        )

    return text


class DiffViewer(VerticalScroll):
    """
    TUI diff 可视化组件。

    特点：
    - old/new 行号对齐
    - 新增行绿色
    - 删除行红色
    - hunk header 高亮
    - 文件头高亮
    """

    DEFAULT_CSS = """
    DiffViewer {
        height: 1fr;
        border: round $surface;
        padding: 0 1;
    }

    DiffViewer #diff-viewer-title {
        height: 1;
        text-style: bold;
        color: $accent;
    }

    DiffViewer #diff-viewer-body {
        height: auto;
    }
    """

    def __init__(
        self,
        diff_text: str = "",
        *,
        title: str = "Diff",
        show_header: bool = True,
        max_lines: int | None = 1_000,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.diff_text = diff_text
        self.title = title
        self.show_header = show_header
        self.max_lines = max_lines

        self._title_widget: Static | None = None
        self._body_widget: Static | None = None

    def compose(self) -> ComposeResult:
        yield Static(
            self.title,
            id="diff-viewer-title",
        )

        yield Static(
            render_unified_diff_text(
                self.diff_text,
                show_header=self.show_header,
                max_lines=self.max_lines,
            ),
            id="diff-viewer-body",
        )

    def on_mount(self) -> None:
        self._title_widget = self.query_one(
            "#diff-viewer-title",
            Static,
        )
        self._body_widget = self.query_one(
            "#diff-viewer-body",
            Static,
        )

        self.refresh_diff()

    def set_diff(
        self,
        diff_text: str,
        *,
        title: str | None = None,
        show_header: bool | None = None,
        max_lines: int | None = None,
        scroll_top: bool = True,
    ) -> None:
        """设置新的 diff 内容。"""
        self.diff_text = diff_text

        if title is not None:
            self.title = title

        if show_header is not None:
            self.show_header = show_header

        if max_lines is not None:
            self.max_lines = max_lines

        self.refresh_diff()

        if scroll_top:
            self.scroll_home(animate=False)

    def clear(self) -> None:
        """清空 diff。"""
        self.diff_text = ""
        self.refresh_diff()

    def refresh_diff(self) -> None:
        """重新渲染当前 diff。"""
        if self._title_widget is not None:
            self._title_widget.update(self.title)

        if self._body_widget is not None:
            self._body_widget.update(
                render_unified_diff_text(
                    self.diff_text,
                    show_header=self.show_header,
                    max_lines=self.max_lines,
                )
            )

    def get_stats(self) -> DiffViewerStats:
        """返回当前 diff 的展示统计。"""
        lines = parse_unified_diff_for_view(self.diff_text)
        return collect_diff_viewer_stats(lines)


class DiffViewerDemoApp(App[None]):
    """DiffViewer 组件 demo。"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #demo-title {
        height: 1;
        text-style: bold;
    }

    DiffViewer {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("1", "show_basic", "Basic"),
        ("2", "show_new_file", "New file"),
        ("3", "show_empty", "Empty"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "DiffViewer demo: press 1 basic, 2 new file, 3 empty, q quit",
            id="demo-title",
        )

        yield DiffViewer(
            make_basic_demo_diff(),
            title="Basic diff",
            id="diff-viewer",
        )

    def action_show_basic(self) -> None:
        viewer = self.query_one(
            "#diff-viewer",
            DiffViewer,
        )
        viewer.set_diff(
            make_basic_demo_diff(),
            title="Basic diff",
        )

    def action_show_new_file(self) -> None:
        viewer = self.query_one(
            "#diff-viewer",
            DiffViewer,
        )
        viewer.set_diff(
            make_new_file_demo_diff(),
            title="New file diff",
        )

    def action_show_empty(self) -> None:
        viewer = self.query_one(
            "#diff-viewer",
            DiffViewer,
        )
        viewer.set_diff(
            "",
            title="Empty diff",
        )


def make_basic_demo_diff() -> str:
    """生成 demo diff。"""
    return (
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,7 +1,8 @@\n"
        " # PyWork\n"
        " \n"
        "-A Python TUI agent workspace inspired by mature coding-agent architecture.\n"
        "+A Python TUI coding-agent workspace inspired by mature agent architecture.\n"
        " \n"
        " ## Status\n"
        " \n"
        "-Project skeleton initialized.\n"
        "+Project runtime, tools, and TUI initialized.\n"
        "+Diff viewer added.\n"
    )


def make_new_file_demo_diff() -> str:
    """生成新文件 demo diff。"""
    return (
        "--- /dev/null\n"
        "+++ b/src/pywork/demo.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+from __future__ import annotations\n"
        "+\n"
        "+def hello() -> str:\n"
        "+    return \"hello PyWork\"\n"
        "+\n"
    )


def demo() -> None:
    DiffViewerDemoApp().run()


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())