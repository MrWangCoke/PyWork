from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pywork.tui.components.friendly_names import (
    friendly_agent_label,
    friendly_task_title,
)
from pywork.tui.components.tasks.models import (
    TaskProgressRow,
    TaskProgressSnapshot,
    TaskProgressStats,
)


def status_style(status: str) -> str:
    styles = {
        "pending": "dim",
        "queued": "cyan",
        "running": "bold yellow",
        "retrying": "magenta",
        "succeeded": "green",
        "failed": "bold red",
        "cancelled": "red",
        "aborted": "bold red",
        "unknown": "dim",
    }

    return styles.get(status, "white")


def format_duration_ms(duration_ms: int | None) -> str:
    if duration_ms is None:
        return "-"

    if duration_ms < 1_000:
        return f"{duration_ms}ms"

    seconds = duration_ms / 1_000

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    rest = int(seconds % 60)

    if minutes < 60:
        return f"{minutes}m {rest}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{hours}h {minutes}m"


def truncate_text(
    text: str,
    *,
    max_chars: int,
) -> str:
    if len(text) <= max_chars:
        return text

    if max_chars <= 1:
        return "…"

    return text[: max_chars - 1] + "…"


def render_stats(stats: TaskProgressStats) -> Text:
    text = Text()

    text.append("total ", style="dim")
    text.append(str(stats.total), style="bold")
    text.append("  active ", style="dim")
    text.append(str(stats.active), style="bold yellow")
    text.append("  ok ", style="dim")
    text.append(str(stats.succeeded), style="green")
    text.append("  failed ", style="dim")
    text.append(str(stats.failed), style="bold red")
    text.append("  cancelled ", style="dim")
    text.append(str(stats.cancelled), style="red")

    return text


def render_task_table(
    rows: list[TaskProgressRow],
    *,
    selected_task_id: str | None = None,
    max_name_width: int = 40,
    max_error_width: int = 36,
) -> Table:
    table = Table(
        expand=True,
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )

    table.add_column("", width=1, no_wrap=True)
    table.add_column("Name", ratio=3, overflow="fold")
    table.add_column("Agent", ratio=1, overflow="ellipsis")
    table.add_column("Status", ratio=1, no_wrap=True)
    table.add_column("Elapsed", justify="right", no_wrap=True)
    table.add_column("Error", ratio=2, overflow="fold")

    for row in rows:
        selected = row.task_id == selected_task_id
        name = truncate_text(
            friendly_task_title(row),
            max_chars=max_name_width,
        )
        agent = friendly_agent_label(
            {
                "agent_name": row.agent,
                "agent_id": row.agent,
            }
        )
        status = row.status
        elapsed = format_duration_ms(row.duration_ms)
        error = truncate_text(
            row.error or "",
            max_chars=max_error_width,
        )

        table.add_row(
            "▶" if selected else "",
            name,
            agent,
            Text(status, style=status_style(status)),
            elapsed,
            Text(error, style="red" if error else "dim"),
            style="reverse" if selected else None,
        )

    return table


def render_empty_tasks() -> Text:
    text = Text()
    text.append("No background tasks.", style="dim")
    return text


def render_task_progress_panel(
    snapshot: TaskProgressSnapshot,
    *,
    title: str = "Background Tasks",
    show_empty: bool = True,
    selected_task_id: str | None = None,
) -> RenderableType:
    body_items: list[RenderableType] = [
        render_stats(snapshot.stats),
    ]

    if snapshot.rows:
        body_items.append(
            render_task_table(
                snapshot.rows,
                selected_task_id=selected_task_id,
            )
        )
        body_items.append(
            Text("↑/↓ select · Enter detail · o output · s stop · r retry · c copy", style="dim")
        )
    elif show_empty:
        body_items.append(
            render_empty_tasks()
        )

    return Panel(
        Group(*body_items),
        title=title,
        border_style="cyan",
    )
