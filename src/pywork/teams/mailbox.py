from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from typing import Any


class MailboxError(Exception):
    """Mailbox 基础异常。"""


class MailboxValidationError(MailboxError):
    """邮箱参数校验异常。"""


class MailboxNotFoundError(MailboxError):
    """消息不存在。"""


class MailboxPermissionError(MailboxError):
    """邮箱权限异常。"""


class MailboxMessageType(str, Enum):
    NOTE = "note"
    TASK = "task"
    RESULT = "result"
    REQUEST = "request"
    RESPONSE = "response"
    CONTROL = "control"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class MailboxMessageStatus(str, Enum):
    PENDING = "pending"
    DELIVERED = "delivered"
    READ = "read"
    ACKED = "acked"
    ARCHIVED = "archived"
    DELETED = "deleted"

    @property
    def is_visible(self) -> bool:
        return self not in {
            MailboxMessageStatus.ARCHIVED,
            MailboxMessageStatus.DELETED,
        }


class MailboxDeliveryMode(str, Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"


class MailboxEventType(str, Enum):
    MESSAGE_SENT = "message_sent"
    MESSAGE_DELIVERED = "message_delivered"
    MESSAGE_READ = "message_read"
    MESSAGE_ACKED = "message_acked"
    MESSAGE_ARCHIVED = "message_archived"
    MESSAGE_DELETED = "message_deleted"


MailboxEventHandler = Callable[["MailboxEvent"], Any | Awaitable[Any]]
MailboxMessagePredicate = Callable[["MailboxMessage"], bool]


def now_timestamp() -> float:
    return time.time()


def new_message_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_thread_id(prefix: str = "thread") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def safe_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Mapping):
        return {
            str(key): safe_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [
            safe_jsonable(item)
            for item in value
        ]

    if is_dataclass(value):
        return safe_jsonable(asdict(value))

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return safe_jsonable(value.to_dict())

    return str(value)


def normalize_message_type(value: MailboxMessageType | str | None) -> MailboxMessageType:
    if isinstance(value, MailboxMessageType):
        return value

    try:
        return MailboxMessageType(str(value or MailboxMessageType.NOTE.value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MailboxMessageType)
        raise MailboxValidationError(
            f"Invalid mailbox message type {value!r}. Valid types: {valid}"
        ) from exc


def normalize_message_status(value: MailboxMessageStatus | str) -> MailboxMessageStatus:
    if isinstance(value, MailboxMessageStatus):
        return value

    try:
        return MailboxMessageStatus(str(value).strip().lower())
    except ValueError as exc:
        valid = ", ".join(item.value for item in MailboxMessageStatus)
        raise MailboxValidationError(
            f"Invalid mailbox message status {value!r}. Valid statuses: {valid}"
        ) from exc


@dataclass(slots=True)
class MailboxMessage:
    """
    Agent 邮件消息。

    sender_id:
        发送方 Agent id。

    recipient_id:
        接收方 Agent id。广播时会为每个 recipient 创建一封消息。

    thread_id:
        对话线程 id，用于 request/response/result 串联。

    parent_message_id:
        回复某条消息时使用。
    """

    message_id: str
    sender_id: str
    recipient_id: str
    content: str

    subject: str = ""
    message_type: MailboxMessageType = MailboxMessageType.NOTE
    status: MailboxMessageStatus = MailboxMessageStatus.PENDING
    delivery_mode: MailboxDeliveryMode = MailboxDeliveryMode.DIRECT

    thread_id: str | None = None
    parent_message_id: str | None = None
    task_id: str | None = None

    payload: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    created_at: float = field(default_factory=now_timestamp)
    delivered_at: float | None = None
    read_at: float | None = None
    acknowledged_at: float | None = None
    archived_at: float | None = None
    deleted_at: float | None = None

    def mark_delivered(self) -> None:
        self.status = MailboxMessageStatus.DELIVERED
        self.delivered_at = self.delivered_at or now_timestamp()

    def mark_read(self) -> None:
        if self.status == MailboxMessageStatus.DELETED:
            return

        self.status = MailboxMessageStatus.READ
        self.read_at = self.read_at or now_timestamp()

    def mark_acked(self) -> None:
        if self.status == MailboxMessageStatus.DELETED:
            return

        self.status = MailboxMessageStatus.ACKED
        now = now_timestamp()
        self.read_at = self.read_at or now
        self.acknowledged_at = self.acknowledged_at or now

    def mark_archived(self) -> None:
        if self.status == MailboxMessageStatus.DELETED:
            return

        self.status = MailboxMessageStatus.ARCHIVED
        self.archived_at = self.archived_at or now_timestamp()

    def mark_deleted(self) -> None:
        self.status = MailboxMessageStatus.DELETED
        self.deleted_at = self.deleted_at or now_timestamp()

    def is_visible_to(
        self,
        agent_id: str,
    ) -> bool:
        return self.recipient_id == agent_id and self.status.is_visible

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "content": self.content,
            "subject": self.subject,
            "message_type": self.message_type.value,
            "status": self.status.value,
            "delivery_mode": self.delivery_mode.value,
            "thread_id": self.thread_id,
            "parent_message_id": self.parent_message_id,
            "task_id": self.task_id,
            "payload": safe_jsonable(self.payload),
            "metadata": safe_jsonable(self.metadata),
            "created_at": self.created_at,
            "delivered_at": self.delivered_at,
            "read_at": self.read_at,
            "acknowledged_at": self.acknowledged_at,
            "archived_at": self.archived_at,
            "deleted_at": self.deleted_at,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MailboxMessage:
        return cls(
            message_id=str(data["message_id"]),
            sender_id=str(data["sender_id"]),
            recipient_id=str(data["recipient_id"]),
            content=str(data.get("content") or ""),
            subject=str(data.get("subject") or ""),
            message_type=normalize_message_type(data.get("message_type")),
            status=normalize_message_status(data.get("status", MailboxMessageStatus.PENDING)),
            delivery_mode=MailboxDeliveryMode(str(data.get("delivery_mode") or MailboxDeliveryMode.DIRECT.value)),
            thread_id=data.get("thread_id"),
            parent_message_id=data.get("parent_message_id"),
            task_id=data.get("task_id"),
            payload=dict(data.get("payload") or {}),
            metadata=dict(data.get("metadata") or {}),
            created_at=float(data.get("created_at") or now_timestamp()),
            delivered_at=data.get("delivered_at"),
            read_at=data.get("read_at"),
            acknowledged_at=data.get("acknowledged_at"),
            archived_at=data.get("archived_at"),
            deleted_at=data.get("deleted_at"),
        )


@dataclass(slots=True)
class MailboxEvent:
    event_type: MailboxEventType
    message_id: str
    sender_id: str | None = None
    recipient_id: str | None = None
    thread_id: str | None = None
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "message_id": self.message_id,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "thread_id": self.thread_id,
            "task_id": self.task_id,
            "metadata": safe_jsonable(self.metadata),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class MailboxPollResult:
    agent_id: str
    messages: list[MailboxMessage]
    timed_out: bool = False
    waited_seconds: float = 0.0

    @property
    def has_messages(self) -> bool:
        return bool(self.messages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "messages": [
                message.to_dict()
                for message in self.messages
            ],
            "timed_out": self.timed_out,
            "waited_seconds": self.waited_seconds,
            "has_messages": self.has_messages,
        }


class AgentMailbox:
    """
    Agent 邮箱。

    这是内存实现，适合当前 Team / Swarm 的第一版协作。
    """

    def __init__(
        self,
        *,
        event_handlers: Sequence[MailboxEventHandler] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.metadata = metadata or {}
        self._messages: dict[str, MailboxMessage] = {}
        self._inbox: dict[str, list[str]] = {}
        self._outbox: dict[str, list[str]] = {}
        self._event_handlers: list[MailboxEventHandler] = list(event_handlers or [])
        self._condition = asyncio.Condition()

    def add_event_handler(
        self,
        handler: MailboxEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: MailboxEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: MailboxEvent,
    ) -> None:
        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    async def notify_waiters(self) -> None:
        async with self._condition:
            self._condition.notify_all()

    def count_messages(self) -> int:
        return len(self._messages)

    def get_message(
        self,
        message_id: str,
    ) -> MailboxMessage:
        message = self._messages.get(message_id)

        if message is None:
            raise MailboxNotFoundError(f"Mailbox message not found: {message_id}")

        return message

    def list_messages(
        self,
        *,
        include_deleted: bool = False,
    ) -> list[MailboxMessage]:
        messages = list(self._messages.values())

        if not include_deleted:
            messages = [
                message
                for message in messages
                if message.status != MailboxMessageStatus.DELETED
            ]

        messages.sort(key=lambda message: message.created_at)

        return messages

    async def send_message(
        self,
        *,
        sender_id: str,
        recipient_id: str,
        content: str,
        subject: str = "",
        message_type: MailboxMessageType | str = MailboxMessageType.NOTE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        task_id: str | None = None,
        delivery_mode: MailboxDeliveryMode = MailboxDeliveryMode.DIRECT,
        message_id: str | None = None,
    ) -> MailboxMessage:
        sender_id = str(sender_id).strip()
        recipient_id = str(recipient_id).strip()

        if not sender_id:
            raise MailboxValidationError("sender_id is required")

        if not recipient_id:
            raise MailboxValidationError("recipient_id is required")

        if content is None:
            raise MailboxValidationError("content is required")

        message = MailboxMessage(
            message_id=message_id or new_message_id(),
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=str(content),
            subject=subject,
            message_type=normalize_message_type(message_type),
            delivery_mode=delivery_mode,
            thread_id=thread_id or new_thread_id(),
            parent_message_id=parent_message_id,
            task_id=task_id,
            payload=dict(payload or {}),
            metadata={
                **self.metadata,
                **dict(metadata or {}),
            },
        )

        message.mark_delivered()

        self._messages[message.message_id] = message
        self._inbox.setdefault(recipient_id, []).append(message.message_id)
        self._outbox.setdefault(sender_id, []).append(message.message_id)

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_SENT,
                message_id=message.message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                thread_id=message.thread_id,
                task_id=task_id,
            )
        )

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_DELIVERED,
                message_id=message.message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                thread_id=message.thread_id,
                task_id=task_id,
            )
        )

        await self.notify_waiters()

        return message

    async def broadcast_message(
        self,
        *,
        sender_id: str,
        recipient_ids: Sequence[str],
        content: str,
        subject: str = "",
        message_type: MailboxMessageType | str = MailboxMessageType.NOTE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
        task_id: str | None = None,
    ) -> list[MailboxMessage]:
        recipients = [
            str(recipient_id).strip()
            for recipient_id in recipient_ids
            if str(recipient_id).strip()
        ]

        if not recipients:
            raise MailboxValidationError("recipient_ids must not be empty")

        shared_thread_id = thread_id or new_thread_id()

        messages: list[MailboxMessage] = []

        for recipient_id in recipients:
            message = await self.send_message(
                sender_id=sender_id,
                recipient_id=recipient_id,
                content=content,
                subject=subject,
                message_type=message_type,
                payload=payload,
                metadata=metadata,
                thread_id=shared_thread_id,
                task_id=task_id,
                delivery_mode=MailboxDeliveryMode.BROADCAST,
            )
            messages.append(message)

        return messages

    async def reply_message(
        self,
        *,
        message_id: str,
        sender_id: str,
        content: str,
        subject: str = "",
        message_type: MailboxMessageType | str = MailboxMessageType.RESPONSE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MailboxMessage:
        original = self.get_message(message_id)

        return await self.send_message(
            sender_id=sender_id,
            recipient_id=original.sender_id,
            content=content,
            subject=subject or f"Re: {original.subject}",
            message_type=message_type,
            payload=payload,
            metadata=metadata,
            thread_id=original.thread_id,
            parent_message_id=original.message_id,
            task_id=original.task_id,
        )

    def _select_messages(
        self,
        *,
        agent_id: str,
        source: str,
        include_read: bool = False,
        include_archived: bool = False,
        include_deleted: bool = False,
        message_type: MailboxMessageType | str | None = None,
        task_id: str | None = None,
        thread_id: str | None = None,
        predicate: MailboxMessagePredicate | None = None,
        limit: int | None = None,
    ) -> list[MailboxMessage]:
        if source == "inbox":
            message_ids = list(self._inbox.get(agent_id, []))
        elif source == "outbox":
            message_ids = list(self._outbox.get(agent_id, []))
        else:
            raise MailboxValidationError(f"Invalid mailbox source: {source}")

        normalized_type = (
            normalize_message_type(message_type)
            if message_type is not None
            else None
        )

        selected: list[MailboxMessage] = []

        for message_id in message_ids:
            message = self._messages.get(message_id)

            if message is None:
                continue

            if not include_deleted and message.status == MailboxMessageStatus.DELETED:
                continue

            if not include_archived and message.status == MailboxMessageStatus.ARCHIVED:
                continue

            if not include_read and source == "inbox" and message.status in {
                MailboxMessageStatus.READ,
                MailboxMessageStatus.ACKED,
            }:
                continue

            if normalized_type is not None and message.message_type != normalized_type:
                continue

            if task_id is not None and message.task_id != task_id:
                continue

            if thread_id is not None and message.thread_id != thread_id:
                continue

            if predicate is not None and not predicate(message):
                continue

            selected.append(message)

        selected.sort(key=lambda item: item.created_at)

        if limit is not None:
            selected = selected[:limit]

        return selected

    def get_inbox(
        self,
        agent_id: str,
        *,
        include_read: bool = False,
        include_archived: bool = False,
        include_deleted: bool = False,
        message_type: MailboxMessageType | str | None = None,
        task_id: str | None = None,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[MailboxMessage]:
        return self._select_messages(
            agent_id=agent_id,
            source="inbox",
            include_read=include_read,
            include_archived=include_archived,
            include_deleted=include_deleted,
            message_type=message_type,
            task_id=task_id,
            thread_id=thread_id,
            limit=limit,
        )

    def get_outbox(
        self,
        agent_id: str,
        *,
        include_archived: bool = True,
        include_deleted: bool = False,
        message_type: MailboxMessageType | str | None = None,
        task_id: str | None = None,
        thread_id: str | None = None,
        limit: int | None = None,
    ) -> list[MailboxMessage]:
        return self._select_messages(
            agent_id=agent_id,
            source="outbox",
            include_read=True,
            include_archived=include_archived,
            include_deleted=include_deleted,
            message_type=message_type,
            task_id=task_id,
            thread_id=thread_id,
            limit=limit,
        )

    async def poll_messages(
        self,
        agent_id: str,
        *,
        limit: int | None = None,
        include_read: bool = False,
        include_archived: bool = False,
        include_deleted: bool = False,
        message_type: MailboxMessageType | str | None = None,
        task_id: str | None = None,
        thread_id: str | None = None,
        predicate: MailboxMessagePredicate | None = None,
        timeout: float | None = None,
        mark_read: bool = False,
    ) -> MailboxPollResult:
        started_at = now_timestamp()

        def select() -> list[MailboxMessage]:
            return self._select_messages(
                agent_id=agent_id,
                source="inbox",
                include_read=include_read,
                include_archived=include_archived,
                include_deleted=include_deleted,
                message_type=message_type,
                task_id=task_id,
                thread_id=thread_id,
                predicate=predicate,
                limit=limit,
            )

        messages = select()

        if not messages and timeout is not None:
            deadline = now_timestamp() + timeout

            async with self._condition:
                while not messages:
                    remaining = deadline - now_timestamp()

                    if remaining <= 0:
                        break

                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=remaining,
                        )
                    except asyncio.TimeoutError:
                        break

                    messages = select()

        if mark_read:
            for message in messages:
                await self.mark_read(
                    message.message_id,
                    agent_id=agent_id,
                )

        waited = now_timestamp() - started_at

        return MailboxPollResult(
            agent_id=agent_id,
            messages=messages,
            timed_out=not messages and timeout is not None,
            waited_seconds=waited,
        )

    async def wait_for_message(
        self,
        agent_id: str,
        *,
        timeout: float | None = None,
        message_type: MailboxMessageType | str | None = None,
        task_id: str | None = None,
        thread_id: str | None = None,
        predicate: MailboxMessagePredicate | None = None,
        mark_read: bool = False,
    ) -> MailboxMessage | None:
        result = await self.poll_messages(
            agent_id,
            limit=1,
            message_type=message_type,
            task_id=task_id,
            thread_id=thread_id,
            predicate=predicate,
            timeout=timeout,
            mark_read=mark_read,
        )

        if not result.messages:
            return None

        return result.messages[0]

    def _require_recipient_permission(
        self,
        message: MailboxMessage,
        agent_id: str | None,
    ) -> None:
        if agent_id is None:
            return

        if message.recipient_id != agent_id:
            raise MailboxPermissionError(
                f"Agent {agent_id!r} cannot modify message {message.message_id!r}"
            )

    async def mark_read(
        self,
        message_id: str,
        *,
        agent_id: str | None = None,
    ) -> MailboxMessage:
        message = self.get_message(message_id)
        self._require_recipient_permission(message, agent_id)

        message.mark_read()

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_READ,
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipient_id=message.recipient_id,
                thread_id=message.thread_id,
                task_id=message.task_id,
            )
        )

        await self.notify_waiters()

        return message

    async def acknowledge_message(
        self,
        message_id: str,
        *,
        agent_id: str | None = None,
    ) -> MailboxMessage:
        message = self.get_message(message_id)
        self._require_recipient_permission(message, agent_id)

        message.mark_acked()

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_ACKED,
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipient_id=message.recipient_id,
                thread_id=message.thread_id,
                task_id=message.task_id,
            )
        )

        await self.notify_waiters()

        return message

    async def archive_message(
        self,
        message_id: str,
        *,
        agent_id: str | None = None,
    ) -> MailboxMessage:
        message = self.get_message(message_id)
        self._require_recipient_permission(message, agent_id)

        message.mark_archived()

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_ARCHIVED,
                message_id=message.message_id,
                sender_id=message.sender_id,
                recipient_id=message.recipient_id,
                thread_id=message.thread_id,
                task_id=message.task_id,
            )
        )

        await self.notify_waiters()

        return message

    async def delete_message(
        self,
        message_id: str,
        *,
        agent_id: str | None = None,
        hard_delete: bool = False,
    ) -> bool:
        message = self.get_message(message_id)
        self._require_recipient_permission(message, agent_id)

        sender_id = message.sender_id
        recipient_id = message.recipient_id

        if hard_delete:
            self._messages.pop(message_id, None)

            if recipient_id in self._inbox:
                self._inbox[recipient_id] = [
                    item
                    for item in self._inbox[recipient_id]
                    if item != message_id
                ]

            if sender_id in self._outbox:
                self._outbox[sender_id] = [
                    item
                    for item in self._outbox[sender_id]
                    if item != message_id
                ]
        else:
            message.mark_deleted()

        await self.emit_event(
            MailboxEvent(
                event_type=MailboxEventType.MESSAGE_DELETED,
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                thread_id=message.thread_id,
                task_id=message.task_id,
                metadata={
                    "hard_delete": hard_delete,
                },
            )
        )

        await self.notify_waiters()

        return True

    def list_thread(
        self,
        thread_id: str,
        *,
        include_deleted: bool = False,
    ) -> list[MailboxMessage]:
        messages = [
            message
            for message in self._messages.values()
            if message.thread_id == thread_id
        ]

        if not include_deleted:
            messages = [
                message
                for message in messages
                if message.status != MailboxMessageStatus.DELETED
            ]

        messages.sort(key=lambda message: message.created_at)

        return messages

    def clear(self) -> None:
        self._messages.clear()
        self._inbox.clear()
        self._outbox.clear()


def create_agent_mailbox(
    *,
    event_handlers: Sequence[MailboxEventHandler] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentMailbox:
    return AgentMailbox(
        event_handlers=event_handlers,
        metadata=metadata,
    )


__all__ = [
    "AgentMailbox",
    "MailboxDeliveryMode",
    "MailboxError",
    "MailboxEvent",
    "MailboxEventHandler",
    "MailboxEventType",
    "MailboxMessage",
    "MailboxMessagePredicate",
    "MailboxMessageStatus",
    "MailboxMessageType",
    "MailboxNotFoundError",
    "MailboxPermissionError",
    "MailboxPollResult",
    "MailboxValidationError",
    "create_agent_mailbox",
    "new_message_id",
    "new_thread_id",
    "normalize_message_status",
    "normalize_message_type",
]