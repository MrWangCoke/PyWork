from __future__ import annotations

from rich.text import Text

from pywork.tui.components.diff.models import (
    DiffLine,
    DiffLineKind,
    DiffRenderOptions,
    DiffStats,
)
from pywork.tui.components.diff.parser import (
    collect_stats,
    line_number_width,
    parse_unified_diff,
)


def style_for_kind(kind: DiffLineKind) -> str:
    """根据行类型返回 Rich 样式。"""
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


def format_lineno(
    lineno: int | None,
    *,
    width: int,
) -> str:
    """格式化行号。None 用空格占位。"""
    if lineno is None:
        return " " * width

    return str(lineno).rjust(width)


def render_summary(stats: DiffStats) -> Text:
    """渲染 diff 统计摘要。"""
    text = Text()

    if stats.is_empty:
        text.append("No changes", style="dim italic")
        return text

    text.append(f"{stats.files} file(s)", style="bold")
    text.append("  ")
    text.append(f"+{stats.additions}", style="green")
    text.append("  ")
    text.append(f"-{stats.deletions}", style="red")
    text.append("  ")
    text.append(f"{stats.hunks} hunk(s)", style="cyan")

    return text


def render_diff_lines(
    lines: list[DiffLine],
    *,
    options: DiffRenderOptions | None = None,
) -> Text:
    """把 DiffLine 列表渲染成 Rich Text。"""
    options = options or DiffRenderOptions()

    if options.max_lines is not None and options.max_lines > 0:
        visible_lines = lines[: options.max_lines]
        truncated = len(lines) > options.max_lines
    else:
        visible_lines = lines
        truncated = False

    width = line_number_width(lines)

    text = Text()

    if options.show_header:
        stats = collect_stats(lines)

        text.append_text(render_summary(stats))
        text.append("\n")

        if options.show_line_numbers:
            text.append(
                f"{'old'.rjust(width)} {'new'.rjust(width)}   diff\n",
                style="dim",
            )

    for line in visible_lines:
        line_style = style_for_kind(line.kind)

        if not options.show_line_numbers:
            marker = line.marker or " "
            text.append(marker, style=line_style or "dim")
            text.append(" ")
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

        if line.kind in {"file_header", "hunk", "meta", "empty"}:
            text.append(" " * width, style="dim")
            text.append(" ", style="dim")
            text.append(" " * width, style="dim")
            text.append("   ", style="dim")
            text.append(line.content, style=line_style)
            text.append("\n")
            continue

        marker = line.marker or " "
        prefix_style = line_style or "dim"

        text.append(old_part, style=prefix_style)
        text.append(" ", style="dim")
        text.append(new_part, style=prefix_style)
        text.append(" ", style="dim")
        text.append(marker, style=prefix_style)
        text.append(" ", style="dim")
        text.append(line.content, style=line_style)
        text.append("\n")

    if truncated:
        text.append(
            f"... diff truncated after {options.max_lines} rendered line(s)\n",
            style="bold red",
        )

    return text


def render_unified_diff(
    diff_text: str,
    *,
    options: DiffRenderOptions | None = None,
) -> Text:
    """直接渲染 unified diff 文本。"""
    lines = parse_unified_diff(diff_text)

    return render_diff_lines(
        lines,
        options=options,
    )