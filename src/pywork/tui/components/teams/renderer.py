from __future__ import annotations

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pywork.tui.components.friendly_names import (
    friendly_assignee_label,
    friendly_team_member_activity,
    friendly_team_member_label,
    role_label,
)
from pywork.tui.components.teams.models import (
    TeamMailboxStats,
    TeamMemberRow,
    TeamTaskRow,
    TeamViewSnapshot,
)


def member_status_style(status: str) -> str:
    styles = {
        "active": "green",
        "disabled": "yellow",
        "removed": "red",
        "unknown": "dim",
    }

    return styles.get(status, "white")


def task_status_style(status: str) -> str:
    styles = {
        "pending": "dim",
        "assigned": "cyan",
        "dispatched": "cyan",
        "running": "bold yellow",
        "succeeded": "green",
        "failed": "bold red",
        "cancelled": "red",
        "unknown": "dim",
    }

    return styles.get(status, "white")


def priority_style(priority: str) -> str:
    styles = {
        "urgent": "bold red",
        "high": "yellow",
        "normal": "white",
        "low": "dim",
    }

    return styles.get(priority, "white")


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


def render_summary(snapshot: TeamViewSnapshot) -> Text:
    stats = snapshot.stats
    mailbox = stats.mailbox

    text = Text()

    text.append(snapshot.name or "Team", style="bold")

    if snapshot.description:
        text.append("\n")
        text.append(snapshot.description, style="dim")

    text.append("\n")
    text.append("members ", style="dim")
    text.append(str(stats.members_total), style="bold")
    text.append("  active ", style="dim")
    text.append(str(stats.members_active), style="green")
    text.append("  busy ", style="dim")
    text.append(str(stats.members_busy), style="bold yellow")

    text.append("    tasks ", style="dim")
    text.append(str(stats.tasks_total), style="bold")
    text.append("  active ", style="dim")
    text.append(str(stats.tasks_active), style="bold yellow")
    text.append("  failed ", style="dim")
    text.append(str(stats.tasks_failed), style="bold red")

    text.append("    messages ", style="dim")
    text.append(str(mailbox.total), style="bold")
    text.append("  unread ", style="dim")
    text.append(str(mailbox.unread), style="cyan")

    return text


def render_members_table(
    members: list[TeamMemberRow],
    *,
    selected_member_id: str | None = None,
) -> RenderableType:
    if not members:
        text = Text()
        text.append("No team members.", style="dim")
        return text

    table = Table(
        expand=True,
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )

    table.add_column("", width=1, no_wrap=True)
    table.add_column("Member", ratio=2, overflow="ellipsis")
    table.add_column("Role", ratio=1, overflow="ellipsis")
    table.add_column("Agent", ratio=1, overflow="ellipsis")
    table.add_column("Status", ratio=1, no_wrap=True)
    table.add_column("Activity", ratio=2, overflow="fold")

    for member in members:
        selected = member.teammate_id == selected_member_id

        table.add_row(
            "▶" if selected else "",
            friendly_team_member_label(member),
            role_label(member.role) or "-",
            role_label(member.agent_name) or friendly_assignee_label(member.agent_name),
            Text(member.status, style=member_status_style(member.status)),
            friendly_team_member_activity(member),
            style="reverse" if selected else None,
        )

    return table


def render_tasks_table(
    tasks: list[TeamTaskRow],
    *,
    max_title_width: int = 42,
) -> RenderableType:
    if not tasks:
        text = Text()
        text.append("No shared tasks.", style="dim")
        return text

    table = Table(
        expand=True,
        show_header=True,
        header_style="bold cyan",
        box=None,
        padding=(0, 1),
    )

    table.add_column("Task", ratio=3, overflow="fold")
    table.add_column("Role", ratio=1, overflow="ellipsis")
    table.add_column("Assigned", ratio=1, overflow="ellipsis")
    table.add_column("Status", ratio=1, no_wrap=True)
    table.add_column("Priority", ratio=1, no_wrap=True)
    table.add_column("Error", ratio=2, overflow="fold")

    for task in tasks:
        table.add_row(
            truncate_text(
                task.title or task.task_id,
                max_chars=max_title_width,
            ),
            task.role or "-",
            friendly_assignee_label(task.assigned_to),
            Text(task.status, style=task_status_style(task.status)),
            Text(task.priority, style=priority_style(task.priority)),
            Text(task.error or "", style="red" if task.error else "dim"),
        )

    return table


def render_mailbox_stats(mailbox: TeamMailboxStats) -> Text:
    text = Text()

    text.append("mailbox ", style="bold")
    text.append("total ", style="dim")
    text.append(str(mailbox.total), style="bold")
    text.append("  unread ", style="dim")
    text.append(str(mailbox.unread), style="cyan")
    text.append("  read ", style="dim")
    text.append(str(mailbox.read), style="white")
    text.append("  acked ", style="dim")
    text.append(str(mailbox.acked), style="green")
    text.append("  task ", style="dim")
    text.append(str(mailbox.task_messages), style="yellow")
    text.append("  result ", style="dim")
    text.append(str(mailbox.result_messages), style="green")
    text.append("  error ", style="dim")
    text.append(str(mailbox.error_messages), style="bold red")

    return text


def render_team_view_panel(
    snapshot: TeamViewSnapshot,
    *,
    title: str = "Team",
    show_members: bool = True,
    show_tasks: bool = True,
    show_mailbox: bool = True,
    selected_member_id: str | None = None,
) -> RenderableType:
    body_items: list[RenderableType] = [
        render_summary(snapshot),
    ]

    if show_members:
        body_items.append(Text("Members", style="bold magenta"))
        body_items.append(
            render_members_table(
                snapshot.members,
                selected_member_id=selected_member_id,
            )
        )

    if show_tasks:
        body_items.append(Text("Shared Tasks", style="bold magenta"))
        body_items.append(render_tasks_table(snapshot.tasks))

    if show_mailbox:
        body_items.append(render_mailbox_stats(snapshot.stats.mailbox))

    body_items.append(
        Text("↑/↓ select member · m mailbox · d dispatch · n message member", style="dim")
    )

    return Panel(
        Group(*body_items),
        title=title,
        border_style="blue",
    )
