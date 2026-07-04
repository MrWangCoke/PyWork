from __future__ import annotations

from pywork.tui.components.agents.models import AgentActivityRow
from pywork.tui.components.friendly_names import (
    friendly_agent_activity,
    friendly_agent_label,
    friendly_assignee_label,
    friendly_task_title,
    friendly_team_member_activity,
    friendly_team_member_label,
    role_label,
)
from pywork.tui.components.tasks.models import TaskProgressRow
from pywork.tui.components.teams.models import TeamMemberRow


def test_role_label_prefers_known_human_names() -> None:
    assert role_label("reviewer") == "Reviewer"
    assert role_label("verifier") == "Verifier"
    assert role_label("planner") == "Planner"


def test_friendly_agent_label_priority_name_role_agent_id() -> None:
    assert friendly_agent_label(
        {
            "name": "Code Reviewer",
            "role": "planner",
            "agent_name": "verifier",
            "agent_id": "agent_123",
        }
    ) == "Code Reviewer"

    assert friendly_agent_label(
        {
            "role": "reviewer",
            "agent_name": "verifier",
            "agent_id": "agent_123",
        }
    ) == "Reviewer"

    assert friendly_agent_label(
        {
            "agent_name": "verifier",
            "agent_id": "agent_123",
        }
    ) == "Verifier"


def test_task_title_uses_agent_action_and_file_basename() -> None:
    row = TaskProgressRow(
        task_id="task_abc",
        name="SubAgent reviewer: Review src/pywork/utils/diff.py",
        agent="reviewer",
        status="running",
    )

    assert friendly_task_title(row) == "Reviewer 正在审查 diff.py"


def test_task_title_uses_completed_action_for_succeeded_review() -> None:
    row = TaskProgressRow(
        task_id="task_abc",
        name="SubAgent reviewer: Review src/pywork/utils/diff.py",
        agent="reviewer",
        status="succeeded",
    )

    assert friendly_task_title(row) == "Reviewer 已审查 diff.py"


def test_agent_activity_uses_current_task_instead_of_run_id() -> None:
    row = AgentActivityRow(
        agent_id="agent_reviewer_1",
        name="",
        role="reviewer",
        status="running",
        current_task="Review src/pywork/utils/diff.py",
        current_run_id="run_123",
        current_task_record_id="task_456",
    )

    assert friendly_agent_label(row) == "Reviewer"
    assert friendly_agent_activity(row) == "正在审查 diff.py"


def test_team_member_label_does_not_fall_back_to_id_too_early() -> None:
    row = TeamMemberRow(
        teammate_id="teammate_abc",
        name="",
        role="verifier",
        agent_name="general",
        status="active",
        current_run_id="run_123",
        current_task_record_id="task_456",
        is_busy=True,
    )

    assert friendly_team_member_label(row) == "Verifier"
    assert friendly_team_member_activity(row) == "正在处理任务"


def test_assignee_label_humanizes_raw_id_as_last_resort() -> None:
    assert friendly_assignee_label("reviewer") == "Reviewer"
    assert friendly_assignee_label("teammate_reviewer_1") == "Reviewer 1"
