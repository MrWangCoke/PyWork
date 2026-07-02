from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.widgets import Static

from pywork.tui.components.tasks.collector import (
    build_task_snapshot,
    build_task_snapshot_from_manager,
)
from pywork.tui.components.tasks.models import TaskProgressSnapshot
from pywork.tui.components.tasks.renderer import render_task_progress_panel


class TaskProgressPanel(Static):
    """
    后台 Task 进度面板。

    显示：
    - 名称
    - Agent
    - 状态
    - 耗时
    """

    DEFAULT_CSS = """
    TaskProgressPanel {
        width: 100%;
        height: auto;
    }
    """

    def __init__(
        self,
        *,
        task_manager: Any | None = None,
        title: str = "Background Tasks",
        show_empty: bool = True,
        limit: int | None = None,
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
        self.task_manager = task_manager
        self.title = title
        self.show_empty = show_empty
        self.limit = limit
        self.snapshot = TaskProgressSnapshot()

    def on_mount(self) -> None:
        self.refresh_panel()

    def render_snapshot(self) -> RenderableType:
        return render_task_progress_panel(
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
        snapshot: TaskProgressSnapshot,
    ) -> None:
        self.snapshot = snapshot
        self.refresh_panel()

    def set_tasks(
        self,
        tasks: list[Any],
        *,
        active_task_ids: set[str] | None = None,
    ) -> None:
        self.set_snapshot(
            build_task_snapshot(
                tasks,
                active_task_ids=active_task_ids,
            )
        )

    async def refresh_from_task_manager(
        self,
        task_manager: Any | None = None,
    ) -> TaskProgressSnapshot:
        if task_manager is not None:
            self.task_manager = task_manager

        if self.task_manager is None:
            self.set_snapshot(TaskProgressSnapshot())
            return self.snapshot

        snapshot = await build_task_snapshot_from_manager(
            self.task_manager,
            limit=self.limit,
        )
        self.set_snapshot(snapshot)

        return snapshot

    def get_stats(self) -> dict[str, int]:
        return self.snapshot.stats.to_dict()

    def clear(self) -> None:
        self.set_snapshot(TaskProgressSnapshot())