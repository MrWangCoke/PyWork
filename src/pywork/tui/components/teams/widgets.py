from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.widgets import Static

from pywork.tui.components.teams.collector import build_team_snapshot
from pywork.tui.components.teams.models import TeamViewSnapshot
from pywork.tui.components.teams.renderer import render_team_view_panel


class TeamViewPanel(Static):
    """
    Team 视图面板。

    显示：
    - Team 基本信息
    - 成员列表
    - shared_task_list
    - mailbox 概览
    """

    DEFAULT_CSS = """
    TeamViewPanel {
        width: 100%;
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        team: Any | None = None,
        title: str = "Team",
        task_limit: int | None = 8,
        include_terminal_tasks: bool = True,
        show_members: bool = True,
        show_tasks: bool = True,
        show_mailbox: bool = True,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
    ) -> None:
        super().__init__(
            "",
            name=name,
            id=id,
            classes=classes,
            disabled=disabled,
        )

        self.team = team
        self.title = title
        self.task_limit = task_limit
        self.include_terminal_tasks = include_terminal_tasks
        self.show_members = show_members
        self.show_tasks = show_tasks
        self.show_mailbox = show_mailbox
        self.snapshot = TeamViewSnapshot()

    def on_mount(self) -> None:
        self.refresh_panel()

    def render_snapshot(self) -> RenderableType:
        return render_team_view_panel(
            self.snapshot,
            title=self.title,
            show_members=self.show_members,
            show_tasks=self.show_tasks,
            show_mailbox=self.show_mailbox,
        )

    def refresh_panel(self) -> None:
        self.update(
            self.render_snapshot()
        )

    def set_snapshot(
        self,
        snapshot: TeamViewSnapshot,
    ) -> None:
        self.snapshot = snapshot
        self.refresh_panel()

    async def refresh_from_team(
        self,
        team: Any | None = None,
    ) -> TeamViewSnapshot:
        if team is not None:
            self.team = team

        if self.team is None:
            self.set_snapshot(TeamViewSnapshot())
            return self.snapshot

        snapshot = await build_team_snapshot(
            self.team,
            task_limit=self.task_limit,
            include_terminal_tasks=self.include_terminal_tasks,
        )

        self.set_snapshot(snapshot)

        return snapshot

    def get_stats(self) -> dict[str, Any]:
        return self.snapshot.stats.to_dict()

    def clear(self) -> None:
        self.set_snapshot(TeamViewSnapshot())