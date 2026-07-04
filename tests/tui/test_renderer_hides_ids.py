from __future__ import annotations

from rich.console import Console

from pywork.tui.components.agents.models import (
    AgentActivityRow,
    AgentActivitySnapshot,
)
from pywork.tui.components.agents.renderer import render_agent_activity_panel
from pywork.tui.components.teams.models import (
    TeamMemberRow,
    TeamViewSnapshot,
)
from pywork.tui.components.teams.renderer import render_team_view_panel


def render_to_text(renderable) -> str:
    console = Console(
        width=120,
        record=True,
        color_system=None,
    )
    console.print(renderable)
    return console.export_text()


def test_agent_panel_main_table_hides_run_id() -> None:
    output = render_to_text(
        render_agent_activity_panel(
            AgentActivitySnapshot(
                rows=[
                    AgentActivityRow(
                        agent_id="agent_reviewer_1",
                        name="",
                        role="reviewer",
                        status="running",
                        current_task="Review src/pywork/utils/diff.py",
                        current_run_id="run_should_not_show",
                        current_task_record_id="task_should_not_show",
                    )
                ]
            )
        )
    )

    assert "Reviewer" in output
    assert "正在审查 diff.py" in output
    assert "run_should_not_show" not in output
    assert "task_should_not_show" not in output


def test_team_panel_main_table_hides_member_run_and_task_ids() -> None:
    output = render_to_text(
        render_team_view_panel(
            TeamViewSnapshot(
                team_id="team_should_not_show",
                name="Runtime Team",
                members=[
                    TeamMemberRow(
                        teammate_id="teammate_reviewer_1",
                        name="",
                        role="reviewer",
                        agent_name="reviewer",
                        status="active",
                        current_run_id="run_should_not_show",
                        current_task_record_id="task_should_not_show",
                        is_busy=True,
                    )
                ],
            )
        )
    )

    assert "Runtime Team" in output
    assert "Reviewer" in output
    assert "正在处理任务" in output
    assert "team_should_not_show" not in output
    assert "run_should_not_show" not in output
    assert "task_should_not_show" not in output