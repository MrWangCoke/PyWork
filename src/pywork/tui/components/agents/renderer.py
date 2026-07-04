from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pywork.tui.components.agents.models import (
    AgentActivityRow,
    AgentActivitySnapshot,
    AgentActivityStats,
)
from pywork.tui.components.friendly_names import (
    friendly_agent_activity,
    friendly_agent_label,
    role_label,
)


def status_style(status: str) -> str:
    styles = {
        "idle": "dim",
        "waiting": "cyan",
        "running": "bold yellow",
        "succeeded": "green",
        "failed": "bold red",
        "aborted": "red",
        "stopped": "red",
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


def render_stats(stats: AgentActivityStats) -> Text:
    text = Text()

    text.append("total ", style="dim")
    text.append(str(stats.total), style="bold")
    text.append("  active ", style="dim")
    text.append(str(stats.active), style="bold yellow")
    text.append("  idle ", style="dim")
    text.append(str(stats.idle), style="dim")
    text.append("  failed ", style="dim")
    text.append(str(stats.failed), style="bold red")
    text.append("  stopped ", style="dim")
    text.append(str(stats.stopped), style="red")

    return text


def render_agent_table(
    rows: list[AgentActivityRow],
    *,
    selected_agent_id: str | None = None,
    max_task_width: int = 42,
) -> Table:
    table = Table(
        expand=True,
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )

    table.add_column("", width=1, no_wrap=True)
    table.add_column("Agent", ratio=2, overflow="ellipsis")
    table.add_column("Role", ratio=1, overflow="ellipsis")
    table.add_column("Status", ratio=1, no_wrap=True)
    table.add_column("Activity", ratio=3, overflow="fold")
    table.add_column("Elapsed", justify="right", no_wrap=True)

    for row in rows:
        selected = row.agent_id == selected_agent_id
        activity = truncate_text(
            friendly_agent_activity(row),
            max_chars=max_task_width,
        )

        table.add_row(
            "▶" if selected else "",
            friendly_agent_label(row),
            role_label(row.role) or "-",
            Text(row.status, style=status_style(row.status)),
            activity,
            format_duration_ms(row.duration_ms),
            style="reverse" if selected else None,
        )

    return table


def render_empty_agents() -> Text:
    text = Text()
    text.append("No active agents.", style="dim")
    return text


def render_agent_activity_panel(
    snapshot: AgentActivitySnapshot,
    *,
    title: str = "Active Agents",
    show_empty: bool = True,
    selected_agent_id: str | None = None,
) -> RenderableType:
    body_items: list[RenderableType] = [
        render_stats(snapshot.stats),
    ]

    if snapshot.rows:
        body_items.append(
            render_agent_table(
                snapshot.rows,
                selected_agent_id=selected_agent_id,
            )
        )
        body_items.append(
            Text("↑/↓ select · Enter inspect · a abort · h history", style="dim")
        )
    elif show_empty:
        body_items.append(
            render_empty_agents()
        )

    return Panel(
        Group(*body_items),
        title=title,
        border_style="magenta",
    )
