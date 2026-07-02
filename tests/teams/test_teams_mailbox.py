from __future__ import annotations

import asyncio

import pytest

from pywork.teams.mailbox import (
    AgentMailbox,
    MailboxDeliveryMode,
    MailboxEventType,
    MailboxMessageStatus,
    MailboxMessageType,
    MailboxPermissionError,
    create_agent_mailbox,
)


@pytest.mark.asyncio
async def test_mailbox_sends_direct_message() -> None:
    mailbox = create_agent_mailbox()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        subject="Hello",
        content="你好，agent_b",
        message_type=MailboxMessageType.NOTE,
        payload={
            "x": 1,
        },
    )

    assert message.message_id.startswith("msg_")
    assert message.sender_id == "agent_a"
    assert message.recipient_id == "agent_b"
    assert message.subject == "Hello"
    assert message.status == MailboxMessageStatus.DELIVERED
    assert message.message_type == MailboxMessageType.NOTE
    assert message.payload["x"] == 1
    assert message.thread_id is not None

    inbox = mailbox.get_inbox("agent_b")

    assert len(inbox) == 1
    assert inbox[0].message_id == message.message_id

    outbox = mailbox.get_outbox("agent_a")

    assert len(outbox) == 1
    assert outbox[0].message_id == message.message_id


@pytest.mark.asyncio
async def test_mailbox_poll_messages_and_mark_read() -> None:
    mailbox = AgentMailbox()

    await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="worker_1",
        content="请执行子任务",
        message_type="task",
    )

    result = await mailbox.poll_messages(
        "worker_1",
        mark_read=True,
    )

    assert result.has_messages is True
    assert len(result.messages) == 1
    assert result.messages[0].status == MailboxMessageStatus.READ

    unread = await mailbox.poll_messages(
        "worker_1",
    )

    assert unread.messages == []

    read_messages = await mailbox.poll_messages(
        "worker_1",
        include_read=True,
    )

    assert len(read_messages.messages) == 1


@pytest.mark.asyncio
async def test_mailbox_wait_for_message() -> None:
    mailbox = AgentMailbox()

    async def delayed_send():
        await asyncio.sleep(0.02)
        await mailbox.send_message(
            sender_id="agent_a",
            recipient_id="agent_b",
            content="delayed message",
        )

    task = asyncio.create_task(delayed_send())

    message = await mailbox.wait_for_message(
        "agent_b",
        timeout=1,
    )

    await task

    assert message is not None
    assert message.content == "delayed message"


@pytest.mark.asyncio
async def test_mailbox_wait_timeout() -> None:
    mailbox = AgentMailbox()

    message = await mailbox.wait_for_message(
        "agent_missing",
        timeout=0.02,
    )

    assert message is None

    result = await mailbox.poll_messages(
        "agent_missing",
        timeout=0.02,
    )

    assert result.timed_out is True
    assert result.messages == []


@pytest.mark.asyncio
async def test_mailbox_broadcast_message() -> None:
    mailbox = AgentMailbox()

    messages = await mailbox.broadcast_message(
        sender_id="coordinator",
        recipient_ids=[
            "worker_1",
            "worker_2",
            "worker_3",
        ],
        content="广播任务",
        message_type="task",
        task_id="task_1",
    )

    assert len(messages) == 3
    assert {
        message.recipient_id
        for message in messages
    } == {
        "worker_1",
        "worker_2",
        "worker_3",
    }

    assert {
        message.thread_id
        for message in messages
    } == {
        messages[0].thread_id,
    }

    assert all(
        message.delivery_mode == MailboxDeliveryMode.BROADCAST
        for message in messages
    )

    assert len(mailbox.get_inbox("worker_1")) == 1
    assert len(mailbox.get_inbox("worker_2")) == 1
    assert len(mailbox.get_inbox("worker_3")) == 1


@pytest.mark.asyncio
async def test_mailbox_reply_keeps_thread() -> None:
    mailbox = AgentMailbox()

    request = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        subject="Question",
        content="你完成了吗？",
        message_type="request",
        task_id="task_1",
    )

    reply = await mailbox.reply_message(
        message_id=request.message_id,
        sender_id="agent_b",
        content="完成了。",
        message_type="response",
    )

    assert reply.recipient_id == "agent_a"
    assert reply.thread_id == request.thread_id
    assert reply.parent_message_id == request.message_id
    assert reply.task_id == request.task_id

    thread = mailbox.list_thread(request.thread_id)

    assert [
        message.message_id
        for message in thread
    ] == [
        request.message_id,
        reply.message_id,
    ]


@pytest.mark.asyncio
async def test_mailbox_ack_archive_and_delete() -> None:
    mailbox = AgentMailbox()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="ack me",
    )

    acked = await mailbox.acknowledge_message(
        message.message_id,
        agent_id="agent_b",
    )

    assert acked.status == MailboxMessageStatus.ACKED
    assert acked.acknowledged_at is not None

    archived = await mailbox.archive_message(
        message.message_id,
        agent_id="agent_b",
    )

    assert archived.status == MailboxMessageStatus.ARCHIVED

    visible = mailbox.get_inbox("agent_b")

    assert visible == []

    archived_visible = mailbox.get_inbox(
        "agent_b",
        include_archived=True,
        include_read=True,
    )

    assert len(archived_visible) == 1

    deleted = await mailbox.delete_message(
        message.message_id,
        agent_id="agent_b",
    )

    assert deleted is True
    assert mailbox.get_message(message.message_id).status == MailboxMessageStatus.DELETED

    assert mailbox.get_inbox(
        "agent_b",
        include_deleted=False,
    ) == []


@pytest.mark.asyncio
async def test_mailbox_hard_delete_removes_message() -> None:
    mailbox = AgentMailbox()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="delete me",
    )

    deleted = await mailbox.delete_message(
        message.message_id,
        agent_id="agent_b",
        hard_delete=True,
    )

    assert deleted is True
    assert mailbox.count_messages() == 0
    assert mailbox.get_inbox("agent_b") == []
    assert mailbox.get_outbox("agent_a") == []


@pytest.mark.asyncio
async def test_mailbox_permission_check() -> None:
    mailbox = AgentMailbox()

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="private",
    )

    with pytest.raises(MailboxPermissionError):
        await mailbox.mark_read(
            message.message_id,
            agent_id="agent_c",
        )


@pytest.mark.asyncio
async def test_mailbox_filters_by_type_task_and_thread() -> None:
    mailbox = AgentMailbox()

    task_message = await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="worker",
        content="task",
        message_type="task",
        task_id="task_1",
    )

    await mailbox.send_message(
        sender_id="coordinator",
        recipient_id="worker",
        content="note",
        message_type="note",
        task_id="task_2",
    )

    by_type = await mailbox.poll_messages(
        "worker",
        message_type="task",
    )

    assert [
        message.message_id
        for message in by_type.messages
    ] == [
        task_message.message_id,
    ]

    by_task = await mailbox.poll_messages(
        "worker",
        task_id="task_1",
    )

    assert [
        message.message_id
        for message in by_task.messages
    ] == [
        task_message.message_id,
    ]

    by_thread = await mailbox.poll_messages(
        "worker",
        thread_id=task_message.thread_id,
    )

    assert [
        message.message_id
        for message in by_thread.messages
    ] == [
        task_message.message_id,
    ]


@pytest.mark.asyncio
async def test_mailbox_events_are_emitted() -> None:
    events = []

    mailbox = AgentMailbox(
        event_handlers=[
            lambda event: events.append(event),
        ]
    )

    message = await mailbox.send_message(
        sender_id="agent_a",
        recipient_id="agent_b",
        content="hello",
    )

    await mailbox.mark_read(
        message.message_id,
        agent_id="agent_b",
    )
    await mailbox.acknowledge_message(
        message.message_id,
        agent_id="agent_b",
    )
    await mailbox.archive_message(
        message.message_id,
        agent_id="agent_b",
    )
    await mailbox.delete_message(
        message.message_id,
        agent_id="agent_b",
    )

    event_types = [
        event.event_type
        for event in events
    ]

    assert MailboxEventType.MESSAGE_SENT in event_types
    assert MailboxEventType.MESSAGE_DELIVERED in event_types
    assert MailboxEventType.MESSAGE_READ in event_types
    assert MailboxEventType.MESSAGE_ACKED in event_types
    assert MailboxEventType.MESSAGE_ARCHIVED in event_types
    assert MailboxEventType.MESSAGE_DELETED in event_types