from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

from pywork.tui.components.agents.collector import (
    build_agent_snapshot,
    build_agent_snapshot_from_sources,
)
from pywork.tui.components.agents.models import AgentActivitySnapshot
from pywork.tui.components.agents.renderer import render_agent_activity_panel


class AgentActivityPanel(Static):
    can_focus = True

    BINDINGS = [
        Binding("up", "select_previous", "Prev agent", show=False),
        Binding("down", "select_next", "Next agent", show=False),
        Binding("enter", "inspect_agent", "Inspect agent", show=False),
        Binding("a", "abort_agent", "Abort run", show=False),
        Binding("h", "show_history", "Agent history", show=False),
    ]

    class AgentInspectRequested(Message):
        def __init__(self, agent_id: str, row: Any) -> None:
            self.agent_id = agent_id
            self.row = row
            super().__init__()

    class AgentAbortRequested(Message):
        def __init__(self, agent_id: str, row: Any) -> None:
            self.agent_id = agent_id
            self.row = row
            super().__init__()

    class AgentHistoryRequested(Message):
        def __init__(self, agent_id: str, row: Any) -> None:
            self.agent_id = agent_id
            self.row = row
            super().__init__()

    """
    活跃 Agent 列表面板。

    显示：
    - Agent 名称
    - Role
    - 状态
    - 当前任务
    - Run ID
    - 耗时
    """

    DEFAULT_CSS = """
    AgentActivityPanel {
        width: 100%;
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        manager: Any | None = None,
        team: Any | None = None,
        agents: list[Any] | None = None,
        title: str = "Active Agents",
        active_only: bool = True,
        show_empty: bool = True,
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

        self.manager = manager
        self.team = team
        self.agents = list(agents or [])
        self.title = title
        self.active_only = active_only
        self.show_empty = show_empty
        self.snapshot = AgentActivitySnapshot()
        self.selected_index = 0

    def on_mount(self) -> None:
        self.refresh_panel()

    def selected_row(self) -> Any | None:
        if not self.snapshot.rows:
            return None

        self.selected_index = max(
            0,
            min(
                self.selected_index,
                len(self.snapshot.rows) - 1,
            ),
        )

        return self.snapshot.rows[self.selected_index]

    def selected_agent_id(self) -> str | None:
        row = self.selected_row()

        if row is None:
            return None

        return row.agent_id

    def render_snapshot(self) -> RenderableType:
        return render_agent_activity_panel(
            self.snapshot,
            title=self.title,
            show_empty=self.show_empty,
            selected_agent_id=self.selected_agent_id(),
        )

    def refresh_panel(self) -> None:
        self.update(
            self.render_snapshot()
        )

    def set_snapshot(
        self,
        snapshot: AgentActivitySnapshot,
    ) -> None:
        self.snapshot = snapshot

        if not self.snapshot.rows:
            self.selected_index = 0
        else:
            self.selected_index = max(
                0,
                min(
                    self.selected_index,
                    len(self.snapshot.rows) - 1,
                ),
            )

        self.refresh_panel()

    def set_agents(
        self,
        agents: list[Any],
        *,
        active_runs: list[Any] | None = None,
        active_only: bool | None = None,
    ) -> None:
        self.agents = list(agents)

        self.set_snapshot(
            build_agent_snapshot(
                self.agents,
                active_runs=active_runs,
                active_only=self.active_only if active_only is None else active_only,
            )
        )

    async def refresh_from_sources(
        self,
        *,
        manager: Any | None = None,
        team: Any | None = None,
        agents: list[Any] | None = None,
    ) -> AgentActivitySnapshot:
        if manager is not None:
            self.manager = manager

        if team is not None:
            self.team = team

        if agents is not None:
            self.agents = list(agents)

        snapshot = await build_agent_snapshot_from_sources(
            manager=self.manager,
            team=self.team,
            agents=self.agents,
            active_only=self.active_only,
        )

        self.set_snapshot(snapshot)

        return snapshot

    def get_stats(self) -> dict[str, int]:
        return self.snapshot.stats.to_dict()

    def clear(self) -> None:
        self.set_snapshot(AgentActivitySnapshot())

    def move_selection(self, delta: int) -> None:
        if not self.snapshot.rows:
            self.selected_index = 0
            self.refresh_panel()
            return

        self.selected_index = (
            self.selected_index + delta
        ) % len(self.snapshot.rows)

        self.refresh_panel()

    def action_select_previous(self) -> None:
        self.move_selection(-1)

    def action_select_next(self) -> None:
        self.move_selection(1)

    def action_inspect_agent(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.AgentInspectRequested(
                    row.agent_id,
                    row,
                )
            )

    def action_abort_agent(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.AgentAbortRequested(
                    row.agent_id,
                    row,
                )
            )

    def action_show_history(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.AgentHistoryRequested(
                    row.agent_id,
                    row,
                )
            )
