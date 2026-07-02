from __future__ import annotations

from pathlib import Path

import pytest

from pywork.subagents.manager import create_default_subagent_manager
from pywork.teams.mailbox import MailboxMessageStatus, MailboxMessageType
from pywork.teams.team import (
    TeamTaskAssignmentError,
    TeamTaskStatus,
    create_team,
)


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


def test_team_create_and_add_members(tmp_path: Path) -> None:
    team = create_team(
        team_id="team_alpha",
        name="Alpha Team",
        description="Test team",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    planner = team.create_teammate(
        teammate_id="planner_1",
        role="planner",
        name="Planner One",
    )
    reviewer = team.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
        name="Reviewer One",
    )

    assert team.team_id == "team_alpha"
    assert team.name == "Alpha Team"
    assert planner.role == "planner"
    assert reviewer.role == "reviewer"
    assert len(team.list_members()) == 2
    assert team.require_teammate("planner_1").role == "planner"


def test_team_create_shared_task(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    task = team.create_shared_task(
        "实现 team.py",
        description="实现 Team 模型：roster + shared_task_list",
        role="planner",
        priority="high",
        payload={
            "file": "src/pywork/teams/team.py",
        },
    )

    assert task.task_id.startswith("team_task_")
    assert task.title == "实现 team.py"
    assert task.status == TeamTaskStatus.PENDING
    assert task.role == "planner"
    assert task.payload["file"] == "src/pywork/teams/team.py"

    listed = team.list_shared_tasks(role="planner")

    assert listed == [task]


def test_team_assign_shared_task_by_role(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )

    task = team.create_shared_task(
        "规划实现",
        role="planner",
    )

    assigned = team.assign_shared_task(task.task_id)

    assert assigned.status == TeamTaskStatus.ASSIGNED
    assert assigned.assigned_to == "planner_1"
    assert assigned.assigned_at is not None


def test_team_assign_shared_task_requires_available_member(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    task = team.create_shared_task(
        "需要 verifier",
        role="verifier",
    )

    with pytest.raises(TeamTaskAssignmentError):
        team.assign_shared_task(task.task_id)


@pytest.mark.asyncio
async def test_team_dispatch_shared_task_sends_mailbox_message(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )

    task = team.create_shared_task(
        "规划 teams/team.py",
        description="规划 Team 模型实现",
        role="planner",
    )

    message = await team.dispatch_shared_task(task.task_id)

    assert message.sender_id == team.team_id
    assert message.recipient_id == "planner_1"
    assert message.message_type == MailboxMessageType.TASK
    assert message.task_id == task.task_id

    inbox = team.mailbox.get_inbox("planner_1")

    assert len(inbox) == 1
    assert inbox[0].message_id == message.message_id

    assert task.status == TeamTaskStatus.DISPATCHED
    assert task.assigned_to == "planner_1"


@pytest.mark.asyncio
async def test_team_dispatch_next_task(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="verifier_1",
        role="verifier",
    )

    first = team.create_shared_task(
        "低优先级",
        role="verifier",
        priority="low",
    )
    second = team.create_shared_task(
        "高优先级",
        role="verifier",
        priority="high",
    )

    message = await team.dispatch_next_task(
        role="verifier",
    )

    assert message is not None
    assert message.task_id == second.task_id
    assert second.status == TeamTaskStatus.DISPATCHED
    assert first.status == TeamTaskStatus.PENDING


@pytest.mark.asyncio
async def test_team_teammate_processes_dispatched_task_and_team_collects_result(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )

    teammate = team.require_teammate("planner_1")

    task = team.create_shared_task(
        "规划 Team",
        description="规划 Team 模型实现",
        role="planner",
    )

    await team.dispatch_shared_task(task.task_id)

    handle_result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert handle_result.success is True

    result_messages = await team.collect_result_messages(
        timeout=0.1,
    )

    assert len(result_messages) == 1

    updated = team.require_shared_task(task.task_id)

    assert updated.status == TeamTaskStatus.SUCCEEDED
    assert updated.result is not None
    assert updated.result["success"] is True

    reply = result_messages[0]

    assert reply.status == MailboxMessageStatus.ACKED


@pytest.mark.asyncio
async def test_team_collects_failed_result_message(tmp_path: Path) -> None:
    async def failing_llm(messages, *, tools=None, metadata=None):
        raise RuntimeError("boom")

    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.roster.manager = make_manager(tmp_path)
    team.roster.manager.set_llm(failing_llm)

    team.create_teammate(
        teammate_id="debugger_1",
        role="debugger",
    )

    teammate = team.require_teammate("debugger_1")

    task = team.create_shared_task(
        "调试失败",
        role="debugger",
    )

    await team.dispatch_shared_task(task.task_id)

    handle_result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert handle_result.success is False

    await team.collect_result_messages(
        timeout=0.1,
    )

    updated = team.require_shared_task(task.task_id)

    assert updated.status == TeamTaskStatus.FAILED
    assert updated.error is not None


@pytest.mark.asyncio
async def test_team_broadcast_message(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    team.create_teammate(
        teammate_id="planner_2",
        role="planner",
    )
    team.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )

    messages = await team.broadcast_message(
        role="planner",
        subject="Notice",
        content="只发给 planner",
    )

    assert len(messages) == 2

    assert len(team.mailbox.get_inbox("planner_1")) == 1
    assert len(team.mailbox.get_inbox("planner_2")) == 1
    assert len(team.mailbox.get_inbox("reviewer_1")) == 0


def test_team_mark_task_statuses(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    task = team.create_shared_task(
        "状态测试",
    )

    team.mark_task_running(task.task_id)

    assert task.status == TeamTaskStatus.RUNNING

    team.mark_task_succeeded(
        task.task_id,
        result={
            "ok": True,
        },
    )

    assert task.status == TeamTaskStatus.SUCCEEDED
    assert task.result == {
        "ok": True,
    }


def test_team_clear_terminal_tasks(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    done = team.create_shared_task("done")
    active = team.create_shared_task("active")

    team.mark_task_succeeded(done.task_id)

    removed = team.clear_shared_tasks()

    assert removed == 1
    assert team.get_shared_task(done.task_id) is None
    assert team.get_shared_task(active.task_id) is active


@pytest.mark.asyncio
async def test_team_stop_and_cancel_helpers(tmp_path: Path) -> None:
    team = create_team(
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    team.create_teammate(
        teammate_id="reviewer_1",
        role="reviewer",
    )

    count = await team.stop_all_teammates(
        reason="test stop",
    )

    assert count == 2
    assert team.require_teammate("planner_1").is_stopped
    assert team.require_teammate("reviewer_1").is_stopped


def test_team_to_dict(tmp_path: Path) -> None:
    team = create_team(
        team_id="team_dict",
        name="Dict Team",
        description="Serialize team",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
        metadata={
            "source": "test",
        },
    )

    team.create_teammate(
        teammate_id="planner_1",
        role="planner",
    )
    team.create_shared_task(
        "任务一",
        role="planner",
    )

    data = team.to_dict()

    assert data["team_id"] == "team_dict"
    assert data["name"] == "Dict Team"
    assert data["roster"]["member_count"] == 1
    assert len(data["shared_task_list"]) == 1
    assert data["metadata"]["source"] == "test"