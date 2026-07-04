from __future__ import annotations

from typing import Any

from rich.console import RenderableType
from textual.binding import Binding
from textual.message import Message
from textual.widgets import Static

from pywork.tui.components.tasks.collector import (
    build_task_snapshot,
    build_task_snapshot_from_manager,
)
from pywork.tui.components.tasks.models import TaskProgressSnapshot
from pywork.tui.components.tasks.renderer import render_task_progress_panel


class TaskProgressPanel(Static):
    can_focus = True

    BINDINGS = [
        Binding("up", "select_previous", "Prev task", show=False),
        Binding("down", "select_next", "Next task", show=False),
        Binding("enter", "open_task", "Task detail", show=False),
        Binding("o", "open_task_output", "Task output", show=False),
        Binding("s", "stop_task", "Stop task", show=False),
        Binding("r", "retry_task", "Retry task", show=False),
        Binding("c", "copy_task_id", "Copy task id", show=False),
    ]

    class TaskDetailRequested(Message):
        def __init__(self, task_id: str, row: Any) -> None:
            self.task_id = task_id
            self.row = row
            super().__init__()

    class TaskOutputRequested(Message):
        def __init__(self, task_id: str, row: Any) -> None:
            self.task_id = task_id
            self.row = row
            super().__init__()

    class TaskStopRequested(Message):
        def __init__(self, task_id: str, row: Any) -> None:
            self.task_id = task_id
            self.row = row
            super().__init__()

    class TaskRetryRequested(Message):
        def __init__(self, task_id: str, row: Any) -> None:
            self.task_id = task_id
            self.row = row
            super().__init__()

    class TaskCopyRequested(Message):
        def __init__(self, task_id: str, row: Any) -> None:
            self.task_id = task_id
            self.row = row
            super().__init__()

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

    def selected_task_id(self) -> str | None:
        row = self.selected_row()

        if row is None:
            return None

        return row.task_id

    def render_snapshot(self) -> RenderableType:
        return render_task_progress_panel(
            self.snapshot,
            title=self.title,
            show_empty=self.show_empty,
            selected_task_id=self.selected_task_id(),
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

    def action_open_task(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.TaskDetailRequested(
                    row.task_id,
                    row,
                )
            )

    def action_open_task_output(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.TaskOutputRequested(
                    row.task_id,
                    row,
                )
            )

    def action_stop_task(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.TaskStopRequested(
                    row.task_id,
                    row,
                )
            )

    def action_retry_task(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.TaskRetryRequested(
                    row.task_id,
                    row,
                )
            )

    def action_copy_task_id(self) -> None:
        row = self.selected_row()

        if row is not None:
            self.post_message(
                self.TaskCopyRequested(
                    row.task_id,
                    row,
                )
            )
