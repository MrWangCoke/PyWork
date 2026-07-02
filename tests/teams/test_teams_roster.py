from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pywork.subagents.manager import create_default_subagent_manager
from pywork.teams.mailbox import create_agent_mailbox
from pywork.teams.roster import (
    RosterMemberAlreadyExistsError,
    RosterMemberNotFoundError,
    RosterMemberStatus,
    TeamRoster,
    create_team_roster,
)
from pywork.teams.teammate import TeammateSpec, TeammateStatus, create_teammate


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run bash",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Run PowerShell",
        },
    },
]


async def fake_llm(messages, *, tools=None, metadata=None):
    return {
        "content": f"agent={metadata['agent_name']} task={messages[-1]['content']}",
        "metadata": {},
    }


def make_manager(tmp_path: Path):
    return create_default_subagent_manager(
        llm=fake_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )


def test_roster_create_teammate(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    member = roster.create_teammate(
        teammate_id="planner_1",
        name="Planner One",
        role="planner",
    )

    assert len(roster) == 1
    assert member.teammate_id == "planner_1"
    assert member.name == "Planner One"
    assert member.role == "planner"
    assert member.agent_name == "planner"
    assert member.status == RosterMemberStatus.ACTIVE
    assert member.is_available is True

    teammate = roster.require_teammate("planner_1")

    assert teammate.teammate_id == "planner_1"
    assert teammate.role == "planner"


def test_roster_add_existing_teammate(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()
    manager = make_manager(tmp_path)

    teammate = create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
        mailbox=mailbox,
        manager=manager,
        workspace_path=tmp_path,
    )

    roster = TeamRoster(
        mailbox=mailbox,
        manager=manager,
        workspace_path=tmp_path,
    )

    member = roster.add_teammate(teammate)

    assert member.teammate is teammate
    assert roster.require_teammate("reviewer_1") is teammate


def test_roster_rejects_duplicate_member(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="debugger_1",
        role="debugger",
    )

    with pytest.raises(RosterMemberAlreadyExistsError):
        roster.create_teammate(
            teammate_id="debugger_1",
            role="debugger",
        )


def test_roster_replace_duplicate_member(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    old_member = roster.create_teammate(
        teammate_id="mate_1",
        role="planner",
    )

    new_member = roster.create_teammate(
        teammate_id="mate_1",
        role="reviewer",
        replace=True,
    )

    assert old_member is not new_member
    assert roster.require_member("mate_1").role == "reviewer"


def test_roster_remove_member(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="verifier_1",
        role="verifier",
    )

    removed = roster.remove_teammate("verifier_1")

    assert removed.status == RosterMemberStatus.REMOVED
    assert "verifier_1" not in roster
    assert roster.list_removed_members()[0].teammate_id == "verifier_1"

    with pytest.raises(RosterMemberNotFoundError):
        roster.require_member("verifier_1")


def test_roster_enable_disable_member(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )

    disabled = roster.disable_member("planner_1")

    assert disabled.status == RosterMemberStatus.DISABLED
    assert disabled.is_available is False

    assert roster.select_member(role="planner") is None

    enabled = roster.enable_member("planner_1")

    assert enabled.status == RosterMemberStatus.ACTIVE
    assert roster.select_member(role="planner") is not None


def test_roster_list_members_by_role_and_status(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    roster.create_teammate(
        teammate_id="planner_2",
        role="planner",
    )
    roster.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )

    roster.disable_member("planner_2")

    planners = roster.list_members(role="planner")

    assert {
        member.teammate_id
        for member in planners
    } == {
        "planner_1",
        "planner_2",
    }

    active_planners = roster.list_members(
        role="planner",
        status=RosterMemberStatus.ACTIVE,
    )

    assert [
        member.teammate_id
        for member in active_planners
    ] == [
        "planner_1",
    ]

    assert roster.count_by_role() == {
        "planner": 2,
        "reviewer": 1,
    }

    assert roster.list_roles() == [
        "planner",
        "reviewer",
    ]


def test_roster_select_member_round_robin(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    roster.create_teammate(
        teammate_id="planner_2",
        role="planner",
    )

    first = roster.select_member(role="planner")
    second = roster.select_member(role="planner")
    third = roster.select_member(role="planner")

    assert first is not None
    assert second is not None
    assert third is not None

    assert first.teammate_id == "planner_1"
    assert second.teammate_id == "planner_2"
    assert third.teammate_id == "planner_1"


def test_roster_select_first_available(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )
    roster.create_teammate(
        teammate_id="reviewer_2",
        role="reviewer",
    )

    selected = roster.select_member(
        role="reviewer",
        strategy="first",
    )

    assert selected is not None
    assert selected.teammate_id == "reviewer_1"


def test_roster_add_from_spec(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    spec = TeammateSpec(
        teammate_id="debugger_1",
        name="Debugger One",
        role="debugger",
        description="Debug specialist",
        workspace_path=tmp_path,
        metadata={
            "level": "senior",
        },
    )

    member = roster.add_from_spec(spec)

    assert member.teammate_id == "debugger_1"
    assert member.name == "Debugger One"
    assert member.role == "debugger"
    assert member.metadata["level"] == "senior"


def test_roster_to_dict(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
        metadata={
            "team": "test",
        },
    )

    roster.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    roster.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )
    roster.disable_member("reviewer_1")

    data = roster.to_dict()

    assert data["member_count"] == 2
    assert data["active_member_count"] == 1
    assert data["disabled_member_count"] == 1
    assert data["roles"] == [
        "planner",
        "reviewer",
    ]
    assert data["metadata"]["team"] == "test"


@pytest.mark.asyncio
async def test_roster_stop_all(tmp_path: Path) -> None:
    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    roster.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    roster.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )

    count = await roster.stop_all(
        reason="test stop",
    )

    assert count == 2

    assert roster.require_teammate("planner_1").status == TeammateStatus.STOPPED
    assert roster.require_teammate("reviewer_1").status == TeammateStatus.STOPPED


@pytest.mark.asyncio
async def test_roster_cancel_all_current(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return {
            "content": "too late",
            "metadata": {},
        }

    roster = create_team_roster(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    teammate = create_teammate(
        teammate_id="debugger_1",
        role="debugger",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    # 替换成慢 manager，方便测试取消。
    teammate.manager = make_manager(tmp_path)
    teammate.manager.set_llm(slow_llm)

    roster.add_teammate(teammate)

    running = asyncio.create_task(
        teammate.execute_task(
            "慢任务",
            execution_mode="task",
        )
    )

    for _ in range(50):
        if teammate.current_task_record_id:
            break
        await asyncio.sleep(0.01)

    count = await roster.cancel_all_current(
        reason="test cancel",
    )

    result = await running

    assert count == 1
    assert result.success is False