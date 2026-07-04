from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

from pywork.tui.components.teams.collector import build_team_snapshot
from pywork.tui.components.teams.models import TeamViewSnapshot
from pywork.tui.components.teams.renderer import render_team_view_panel


class TeamViewPanel(Static):
    can_focus = True

    BINDINGS = [
        Binding("up", "select_previous_member", "Prev member", show=False),
        Binding("down", "select_next_member", "Next member", show=False),
        Binding("m", "open_mailbox", "Mailbox", show=False),
        Binding("d", "dispatch_task", "Dispatch task", show=False),
        Binding("n", "message_member", "Message member", show=False),
    ]

    class TeamMailboxRequested(Message):
        pass

    class TeamDispatchRequested(Message):
        pass

    class TeamMessageMemberRequested(Message):
        def __init__(self, teammate_id: str, row: Any) -> None:
            self.teammate_id = teammate_id
            self.row = row
            super().__init__()

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
        self.selected_member_index = 0

    def on_mount(self) -> None:
        self.refresh_panel()

    def selected_member_row(self) -> Any | None:
        if not self.snapshot.members:
            return None

        self.selected_member_index = max(
            0,
            min(
                self.selected_member_index,
                len(self.snapshot.members) - 1,
            ),
        )

        return self.snapshot.members[self.selected_member_index]

    def selected_member_id(self) -> str | None:
        row = self.selected_member_row()

        if row is None:
            return None

        return row.teammate_id

    def render_snapshot(self) -> RenderableType:
        return render_team_view_panel(
            self.snapshot,
            title=self.title,
            show_members=self.show_members,
            show_tasks=self.show_tasks,
            show_mailbox=self.show_mailbox,
            selected_member_id=self.selected_member_id(),
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

        if not self.snapshot.members:
            self.selected_member_index = 0
        else:
            self.selected_member_index = max(
                0,
                min(
                    self.selected_member_index,
                    len(self.snapshot.members) - 1,
                ),
            )

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

    def move_member_selection(self, delta: int) -> None:
        if not self.snapshot.members:
            self.selected_member_index = 0
            self.refresh_panel()
            return

        self.selected_member_index = (
            self.selected_member_index + delta
        ) % len(self.snapshot.members)

        self.refresh_panel()

    def action_select_previous_member(self) -> None:
        self.move_member_selection(-1)

    def action_select_next_member(self) -> None:
        self.move_member_selection(1)

    def action_open_mailbox(self) -> None:
        self.post_message(self.TeamMailboxRequested())

    def action_dispatch_task(self) -> None:
        self.post_message(self.TeamDispatchRequested())

    def action_message_member(self) -> None:
        row = self.selected_member_row()

        if row is not None:
            self.post_message(
                self.TeamMessageMemberRequested(
                    row.teammate_id,
                    row,
                )
            )
