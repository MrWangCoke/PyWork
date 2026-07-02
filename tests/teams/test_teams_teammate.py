from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from pywork.subagents.manager import create_default_subagent_manager
from pywork.teams.mailbox import (
    MailboxMessageStatus,
    MailboxMessageType,
    create_agent_mailbox,
)
from pywork.teams.teammate import (
    TeammateBusyError,
    TeammateExecutionMode,
    TeammateMessageAction,
    TeammateStatus,
    create_teammate,
)
from pywork.tasks.task import TaskStatus


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
    tool_names = [
        tool["function"]["name"]
        for tool in tools or []
    ]

    return {
        "content": (
            f"agent={metadata['agent_name']} "
            f"role={metadata['agent_role']} "
            f"mode={metadata['permission_mode']} "
            f"tools={','.join(tool_names)} "
            f"task={messages[-1]['content']}"
        ),
        "metadata": {},
    }


def make_manager(tmp_path: Path, llm=fake_llm):
    return create_default_subagent_manager(
        llm=llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )


def test_create_teammate_defaults(tmp_path: Path) -> None:
    teammate = create_teammate(
        teammate_id="mate_planner",
        name="Planner Mate",
        role="planner",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    assert teammate.teammate_id == "mate_planner"
    assert teammate.name == "Planner Mate"
    assert teammate.role == "planner"
    assert teammate.agent_name == "planner"
    assert teammate.status == TeammateStatus.IDLE

    data = teammate.to_dict()

    assert data["teammate_id"] == "mate_planner"
    assert data["role"] == "planner"
    assert data["agent_name"] == "planner"


@pytest.mark.asyncio
async def test_teammate_can_send_message(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_a",
        role="general",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    message = await teammate.send_message(
        recipient_id="mate_b",
        subject="Hello",
        content="你好",
        message_type="note",
    )

    assert message.sender_id == "mate_a"
    assert message.recipient_id == "mate_b"
    assert message.subject == "Hello"

    inbox = mailbox.get_inbox("mate_b")

    assert len(inbox) == 1
    assert inbox[0].message_id == message.message_id


@pytest.mark.asyncio
async def test_teammate_execute_direct_task(tmp_path: Path) -> None:
    teammate = create_teammate(
        teammate_id="mate_reviewer",
        role="reviewer",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    result = await teammate.execute_task(
        "审查 teammate.py 的实现",
    )

    assert result.success is True
    assert result.agent_name == "reviewer"
    assert result.execution_mode == TeammateExecutionMode.DIRECT
    assert "agent=reviewer" in result.content
    assert teammate.status == TeammateStatus.IDLE


@pytest.mark.asyncio
async def test_teammate_execute_task_backed(tmp_path: Path) -> None:
    teammate = create_teammate(
        teammate_id="mate_verifier",
        role="verifier",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    result = await teammate.execute_task(
        "验证 teammate.py 的测试",
        execution_mode=TeammateExecutionMode.TASK,
    )

    assert result.success is True
    assert result.execution_mode == TeammateExecutionMode.TASK
    assert result.task_record is not None
    assert result.task_record.status == TaskStatus.SUCCEEDED
    assert result.task_record.agent_id == "verifier"


@pytest.mark.asyncio
async def test_teammate_processes_task_message_and_replies(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_planner",
        role="planner",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    message = await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_planner",
        subject="Plan",
        content="规划 teams/teammate.py 的实现",
        message_type="task",
        task_id="task_1",
    )

    result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert result.handled is True
    assert result.success is True
    assert result.action == TeammateMessageAction.EXECUTE_TASK
    assert result.task_result is not None
    assert result.result_message_id is not None

    original = mailbox.get_message(message.message_id)

    assert original.status == MailboxMessageStatus.ACKED

    replies = mailbox.get_inbox(
        "coordinator",
        include_read=True,
    )

    assert len(replies) == 1
    assert replies[0].message_type == MailboxMessageType.RESULT
    assert replies[0].parent_message_id == message.message_id
    assert replies[0].payload["task_result"]["success"] is True


@pytest.mark.asyncio
async def test_teammate_processes_request_message(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_debugger",
        role="debugger",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    message = await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_debugger",
        subject="Question",
        content="为什么 pytest 失败？",
        message_type="request",
    )

    result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert result.handled is True
    assert result.action == TeammateMessageAction.RESPOND_REQUEST
    assert result.success is True

    replies = mailbox.get_inbox(
        "coordinator",
        include_read=True,
    )

    assert len(replies) == 1
    assert replies[0].message_type == MailboxMessageType.RESPONSE
    assert replies[0].parent_message_id == message.message_id


@pytest.mark.asyncio
async def test_teammate_acks_note_message(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_general",
        role="general",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    message = await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_general",
        content="普通通知",
        message_type="note",
    )

    result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert result.handled is True
    assert result.action == TeammateMessageAction.ACK
    assert result.success is True

    original = mailbox.get_message(message.message_id)

    assert original.status == MailboxMessageStatus.ACKED


@pytest.mark.asyncio
async def test_teammate_control_stop_message(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_stop",
        role="general",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_stop",
        content="stop",
        message_type="control",
    )

    result = await teammate.process_next_message(
        timeout=0.1,
    )

    assert result.handled is True
    assert result.action == TeammateMessageAction.STOP
    assert result.success is True
    assert teammate.status == TeammateStatus.STOPPED


@pytest.mark.asyncio
async def test_teammate_process_next_message_timeout(tmp_path: Path) -> None:
    teammate = create_teammate(
        teammate_id="mate_timeout",
        role="general",
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    result = await teammate.process_next_message(
        timeout=0.02,
    )

    assert result.handled is False
    assert result.success is True
    assert result.action == TeammateMessageAction.NONE
    assert result.metadata["timed_out"] is True


@pytest.mark.asyncio
async def test_teammate_busy_error(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(0.05)
        return {
            "content": "slow done",
            "metadata": {},
        }

    teammate = create_teammate(
        teammate_id="mate_busy",
        role="planner",
        manager=make_manager(tmp_path, llm=slow_llm),
        workspace_path=tmp_path,
    )

    running = asyncio.create_task(
        teammate.execute_task(
            "慢任务",
        )
    )

    for _ in range(20):
        if teammate.is_busy:
            break
        await asyncio.sleep(0.01)

    assert teammate.is_busy

    with pytest.raises(TeammateBusyError):
        await teammate.execute_task(
            "另一个任务",
        )

    result = await running

    assert result.success is True


@pytest.mark.asyncio
async def test_teammate_cancel_task_backed_run(tmp_path: Path) -> None:
    async def slow_llm(messages, *, tools=None, metadata=None):
        await asyncio.sleep(10)
        return {
            "content": "too late",
            "metadata": {},
        }

    teammate = create_teammate(
        teammate_id="mate_cancel",
        role="debugger",
        manager=make_manager(tmp_path, llm=slow_llm),
        workspace_path=tmp_path,
    )

    running = asyncio.create_task(
        teammate.execute_task(
            "慢速调试任务",
            execution_mode=TeammateExecutionMode.TASK,
        )
    )

    for _ in range(50):
        if teammate.current_task_record_id:
            break
        await asyncio.sleep(0.01)

    cancelled = await teammate.cancel_current(
        reason="user cancelled",
    )

    result = await running

    assert cancelled is True
    assert result.success is False
    assert result.task_record is not None
    assert result.task_record.status == TaskStatus.CANCELLED


@pytest.mark.asyncio
async def test_teammate_run_loop_processes_messages(tmp_path: Path) -> None:
    mailbox = create_agent_mailbox()

    teammate = create_teammate(
        teammate_id="mate_loop",
        role="general",
        mailbox=mailbox,
        manager=make_manager(tmp_path),
        workspace_path=tmp_path,
    )

    await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_loop",
        content="通知一",
        message_type="note",
    )

    await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="mate_loop",
        content="stop",
        message_type="control",
    )

    results = await teammate.run_loop(
        poll_timeout=0.05,
        max_iterations=5,
    )

    actions = [
        result.action
        for result in results
        if result.handled
    ]

    assert TeammateMessageAction.ACK in actions
    assert TeammateMessageAction.STOP in actions
    assert teammate.status == TeammateStatus.STOPPED