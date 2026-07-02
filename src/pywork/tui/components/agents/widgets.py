from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.widgets import Static

from pywork.tui.components.agents.collector import (
    build_agent_snapshot,
    build_agent_snapshot_from_sources,
)
from pywork.tui.components.agents.models import AgentActivitySnapshot
from pywork.tui.components.agents.renderer import render_agent_activity_panel


class AgentActivityPanel(Static):
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

    def on_mount(self) -> None:
        self.refresh_panel()

    def render_snapshot(self) -> RenderableType:
        return render_agent_activity_panel(
            self.snapshot,
            title=self.title,
            show_empty=self.show_empty,
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