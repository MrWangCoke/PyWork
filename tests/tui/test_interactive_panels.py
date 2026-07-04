from __future__ import annotations

from pywork.tui.components.agents.models import (
    AgentActivityRow,
    AgentActivitySnapshot,
)
from pywork.tui.components.agents.widgets import AgentActivityPanel
from pywork.tui.components.tasks.models import (
    TaskProgressRow,
    TaskProgressSnapshot,
)
from pywork.tui.components.tasks.widgets import TaskProgressPanel
from pywork.tui.components.teams.models import (
    TeamMemberRow,
    TeamViewSnapshot,
)
from pywork.tui.components.teams.widgets import TeamViewPanel


def test_task_panel_selection_moves_between_rows() -> None:
    panel = TaskProgressPanel()
    panel.set_snapshot(
        TaskProgressSnapshot(
            rows=[
                TaskProgressRow(task_id="task_1", name="one"),
                TaskProgressRow(task_id="task_2", name="two"),
            ]
        )
    )

    assert panel.selected_task_id() == "task_1"

    panel.action_select_next()

    assert panel.selected_task_id() == "task_2"

    panel.action_select_previous()

    assert panel.selected_task_id() == "task_1"


def test_task_panel_clamps_selection_after_snapshot_update() -> None:
    panel = TaskProgressPanel()
    panel.set_snapshot(
        TaskProgressSnapshot(
            rows=[
                TaskProgressRow(task_id="task_1", name="one"),
                TaskProgressRow(task_id="task_2", name="two"),
            ]
        )
    )
    panel.action_select_next()

    panel.set_snapshot(
        TaskProgressSnapshot(
            rows=[
                TaskProgressRow(task_id="task_3", name="three"),
            ]
        )
    )

    assert panel.selected_task_id() == "task_3"


def test_agent_panel_selection_moves_between_rows() -> None:
    panel = AgentActivityPanel()
    panel.set_snapshot(
        AgentActivitySnapshot(
            rows=[
                AgentActivityRow(agent_id="agent_1", name="one"),
                AgentActivityRow(agent_id="agent_2", name="two"),
            ]
        )
    )

    assert panel.selected_agent_id() == "agent_1"

    panel.action_select_next()

    assert panel.selected_agent_id() == "agent_2"


def test_team_panel_selection_moves_between_members() -> None:
    panel = TeamViewPanel()
    panel.set_snapshot(
        TeamViewSnapshot(
            members=[
                TeamMemberRow(teammate_id="member_1", name="one"),
                TeamMemberRow(teammate_id="member_2", name="two"),
            ]
        )
    )

    assert panel.selected_member_id() == "member_1"

    panel.action_select_next_member()

    assert panel.selected_member_id() == "member_2"