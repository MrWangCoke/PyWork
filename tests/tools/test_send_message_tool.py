from __future__ import annotations

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.teams.mailbox import (
    MailboxMessageStatus,
    MailboxMessageType,
    create_agent_mailbox,
)
from pywork.teams.team import create_team
from pywork.tools.send_message import SendMessageTool
from pywork.tools.tool import ToolExecutionContext


def make_context(**metadata):
    return ToolExecutionContext(
        workspace_path=".",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_send_message_tool_sends_direct_message() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "send",
            "sender_id": "agent_a",
            "recipient_id": "agent_b",
            "subject": "Hello",
            "content": "你好",
            "message_type": "note",
            "payload": {
                "x": 1,
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(mailbox=mailbox),
    )

    assert result.success is True
    assert result.data["action"] == "send"

    message = result.data["message"]

    assert message["sender_id"] == "agent_a"
    assert message["recipient_id"] == "agent_b"
    assert message["subject"] == "Hello"
    assert message["payload"]["x"] == 1

    inbox = mailbox.get_inbox("agent_b")

    assert len(inbox) == 1
    assert inbox[0].content == "你好"


@pytest.mark.asyncio
async def test_send_message_tool_broadcasts_message() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "broadcast",
            "sender_id": "coordinator",
            "recipient_ids": [
                "agent_a",
                "agent_b",
            ],
            "content": "广播通知",
            "message_type": "note",
        },
    )

    result = await tool.execute(
        call,
        make_context(mailbox=mailbox),
    )

    assert result.success is True
    assert result.data["count"] == 2

    assert len(mailbox.get_inbox("agent_a")) == 1
    assert len(mailbox.get_inbox("agent_b")) == 1


@pytest.mark.asyncio
async def test_send_message_tool_broadcasts_by_team_role(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
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

    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "broadcast",
            "role": "planner",
            "content": "只发给 planner",
            "message_type": "note",
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["count"] == 2

    assert len(team.mailbox.get_inbox("planner_1")) == 1
    assert len(team.mailbox.get_inbox("planner_2")) == 1
    assert len(team.mailbox.get_inbox("reviewer_1")) == 0


@pytest.mark.asyncio
async def test_send_message_tool_replies_to_message() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    original = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        subject="Question",
        content="完成了吗？",
        message_type="request",
    )

    call = create_tool_call(
        "send_message",
        {
            "action": "reply",
            "sender_id": "agent_b",
            "message_id": original.message_id,
            "content": "完成了。",
            "message_type": "response",
        },
    )

    result = await tool.execute(
        call,
        make_context(mailbox=mailbox),
    )

    assert result.success is True

    reply = result.data["message"]

    assert reply["recipient_id"] == "agent_a"
    assert reply["thread_id"] == original.thread_id
    assert reply["parent_message_id"] == original.message_id

    inbox = mailbox.get_inbox("agent_a")

    assert len(inbox) == 1
    assert inbox[0].content == "完成了。"


@pytest.mark.asyncio
async def test_send_message_tool_lists_inbox_and_outbox() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="one",
    )
    await mailbox.send_message(
        sender_id="agent_b",
        recipient_id="agent_a",
        content="two",
    )

    inbox_call = create_tool_call(
        "send_message",
        {
            "action": "inbox",
            "agent_id": "agent_b",
        },
    )

    inbox_result = await tool.execute(
        inbox_call,
        make_context(mailbox=mailbox),
    )

    assert inbox_result.success is True
    assert inbox_result.data["count"] == 1
    assert inbox_result.data["messages"][0]["content"] == "one"

    outbox_call = create_tool_call(
        "send_message",
        {
            "action": "outbox",
            "agent_id": "agent_a",
        },
    )

    outbox_result = await tool.execute(
        outbox_call,
        make_context(mailbox=mailbox),
    )

    assert outbox_result.success is True
    assert outbox_result.data["count"] == 1
    assert outbox_result.data["messages"][0]["content"] == "one"


@pytest.mark.asyncio
async def test_send_message_tool_poll_mark_read_and_ack() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="poll me",
    )

    poll_call = create_tool_call(
        "send_message",
        {
            "action": "poll",
            "agent_id": "agent_b",
            "mark_read": True,
        },
    )

    poll_result = await tool.execute(
        poll_call,
        make_context(mailbox=mailbox),
    )

    assert poll_result.success is True
    assert poll_result.data["poll_result"]["has_messages"] is True

    assert mailbox.get_message(message.message_id).status == MailboxMessageStatus.READ

    ack_call = create_tool_call(
        "send_message",
        {
            "action": "ack",
            "message_id": message.message_id,
            "agent_id": "agent_b",
        },
    )

    ack_result = await tool.execute(
        ack_call,
        make_context(mailbox=mailbox),
    )

    assert ack_result.success is True
    assert mailbox.get_message(message.message_id).status == MailboxMessageStatus.ACKED


@pytest.mark.asyncio
async def test_send_message_tool_archive_and_delete() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="delete me",
    )

    archive_call = create_tool_call(
        "send_message",
        {
            "action": "archive",
            "message_id": message.message_id,
            "agent_id": "agent_b",
        },
    )

    archive_result = await tool.execute(
        archive_call,
        make_context(mailbox=mailbox),
    )

    assert archive_result.success is True
    assert mailbox.get_message(message.message_id).status == MailboxMessageStatus.ARCHIVED

    delete_call = create_tool_call(
        "send_message",
        {
            "action": "delete",
            "message_id": message.message_id,
            "agent_id": "agent_b",
        },
    )

    delete_result = await tool.execute(
        delete_call,
        make_context(mailbox=mailbox),
    )

    assert delete_result.success is True
    assert mailbox.get_message(message.message_id).status == MailboxMessageStatus.DELETED


@pytest.mark.asyncio
async def test_send_message_tool_resolves_sender_from_context() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "send",
            "recipient_id": "agent_b",
            "content": "from context",
        },
    )

    result = await tool.execute(
        call,
        make_context(
            mailbox=mailbox,
            current_agent_id="agent_a",
        ),
    )

    assert result.success is True
    assert result.data["message"]["sender_id"] == "agent_a"


@pytest.mark.asyncio
async def test_send_message_tool_missing_mailbox_returns_error() -> None:
    tool = SendMessageTool()

    call = create_tool_call(
        "send_message",
        {
            "action": "send",
            "sender_id": "agent_a",
            "recipient_id": "agent_b",
            "content": "hello",
        },
    )

    result = await tool.execute(
        call,
        make_context(),
    )

    assert result.success is False
    assert "requires AgentMailbox" in result.error


@pytest.mark.asyncio
async def test_send_message_tool_filters_by_message_type() -> None:
    mailbox = create_agent_mailbox()
    tool = SendMessageTool()

    task_message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="task",
        message_type="task",
    )
    await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="note",
        message_type="note",
    )

    call = create_tool_call(
        "send_message",
        {
            "action": "inbox",
            "agent_id": "agent_b",
            "message_type": "task",
        },
    )

    result = await tool.execute(
        call,
        make_context(mailbox=mailbox),
    )

    assert result.success is True
    assert result.data["count"] == 1
    assert result.data["messages"][0]["message_id"] == task_message.message_id
    assert result.data["messages"][0]["message_type"] == MailboxMessageType.TASK.value