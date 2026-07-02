from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.teams.mailbox import (
    AgentMailbox,
    MailboxMessage,
    MailboxMessageType,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


class SendMessageAction(str, Enum):
    SEND = "send"
    BROADCAST = "broadcast"
    REPLY = "reply"
    INBOX = "inbox"
    OUTBOX = "outbox"
    POLL = "poll"
    READ = "read"
    ACK = "ack"
    ARCHIVE = "archive"
    DELETE = "delete"


class SendMessageToolError(Exception):
    """send_message tool 基础异常。"""


class SendMessageRuntimeMissingError(SendMessageToolError):
    """运行时缺少 mailbox/team 等对象。"""


def normalize_action(value: str | None) -> SendMessageAction:
    text = str(value or SendMessageAction.SEND.value).strip().lower()

    aliases = {
        "send_message": "send",
        "message": "send",
        "dm": "send",
        "direct": "send",
        "broadcast_message": "broadcast",
        "reply_message": "reply",
        "list_inbox": "inbox",
        "list_outbox": "outbox",
        "poll_messages": "poll",
        "mark_read": "read",
        "acknowledge": "ack",
        "acknowledge_message": "ack",
        "delete_message": "delete",
        "archive_message": "archive",
    }

    text = aliases.get(text, text)

    try:
        return SendMessageAction(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in SendMessageAction)
        raise ToolValidationError(
            f"Invalid send_message action {value!r}. Valid actions: {valid}"
        ) from exc


def get_call_args(call: ToolCall) -> dict[str, Any]:
    args = getattr(call, "arguments", None)

    if args is None:
        return {}

    if isinstance(args, Mapping):
        return dict(args)

    raise ToolValidationError("Tool call arguments must be an object")


def get_call_id(call: ToolCall) -> str:
    return str(
        getattr(call, "call_id", None)
        or getattr(call, "id", None)
        or ""
    )


def make_result(
    call: ToolCall,
    *,
    tool_name: str,
    success: bool,
    content: str,
    data: dict[str, Any] | None = None,
    error: str | None = None,
) -> ToolResult:
    from pywork.schemas.tool_schema import ToolResultStatus

    return ToolResult(
        call_id=get_call_id(call),
        tool_name=tool_name,
        status=ToolResultStatus.SUCCESS if success else ToolResultStatus.ERROR,
        success=success,
        content=content,
        data=data or {},
        error=error,
    )


def context_metadata(context: ToolExecutionContext) -> dict[str, Any]:
    metadata = getattr(context, "metadata", None)

    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


def object_has_attr(value: Any, attr: str) -> bool:
    return hasattr(value, attr) and getattr(value, attr) is not None


def resolve_mailbox(context: ToolExecutionContext) -> AgentMailbox:
    """
    从 ToolExecutionContext 中解析 Mailbox。

    支持以下来源：
    - context.metadata["mailbox"]
    - context.metadata["team"].mailbox
    - context.metadata["swarm"].team.mailbox
    - context.metadata["teammate"].mailbox
    - context.metadata["agent"].mailbox

    注意：
    这里不要自动 create_agent_mailbox()。
    send_message 必须使用运行时共享 mailbox，否则消息会进入孤立邮箱。
    """

    metadata = context_metadata(context)

    mailbox = metadata.get("mailbox")

    if isinstance(mailbox, AgentMailbox):
        return mailbox

    team = metadata.get("team")

    if team is not None and object_has_attr(team, "mailbox"):
        team_mailbox = getattr(team, "mailbox")

        if isinstance(team_mailbox, AgentMailbox):
            return team_mailbox

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "team"):
        swarm_team = getattr(swarm, "team")

        if swarm_team is not None and object_has_attr(swarm_team, "mailbox"):
            swarm_mailbox = getattr(swarm_team, "mailbox")

            if isinstance(swarm_mailbox, AgentMailbox):
                return swarm_mailbox

    teammate = metadata.get("teammate") or metadata.get("agent")

    if teammate is not None and object_has_attr(teammate, "mailbox"):
        teammate_mailbox = getattr(teammate, "mailbox")

        if isinstance(teammate_mailbox, AgentMailbox):
            return teammate_mailbox

    raise SendMessageRuntimeMissingError(
        "send_message requires AgentMailbox in context.metadata['mailbox'], "
        "or a team/swarm/teammate object with mailbox."
    )

def has_message_runtime(context: ToolExecutionContext) -> bool:
    metadata = context_metadata(context)

    mailbox = metadata.get("mailbox")

    if isinstance(mailbox, AgentMailbox):
        return True

    team = metadata.get("team")

    if team is not None and object_has_attr(team, "mailbox"):
        team_mailbox = getattr(team, "mailbox")

        if isinstance(team_mailbox, AgentMailbox):
            return True

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "team"):
        swarm_team = getattr(swarm, "team")

        if swarm_team is not None and object_has_attr(swarm_team, "mailbox"):
            swarm_mailbox = getattr(swarm_team, "mailbox")

            if isinstance(swarm_mailbox, AgentMailbox):
                return True

    teammate = metadata.get("teammate") or metadata.get("agent")

    if teammate is not None and object_has_attr(teammate, "mailbox"):
        teammate_mailbox = getattr(teammate, "mailbox")

        if isinstance(teammate_mailbox, AgentMailbox):
            return True

    return False

def resolve_team(context: ToolExecutionContext) -> Any | None:
    metadata = context_metadata(context)

    team = metadata.get("team")

    if team is not None:
        return team

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "team"):
        return getattr(swarm, "team")

    return None


def resolve_sender_id(
    args: Mapping[str, Any],
    context: ToolExecutionContext,
) -> str:
    explicit = args.get("sender_id") or args.get("from") or args.get("from_id")

    if explicit:
        return str(explicit)

    metadata = context_metadata(context)

    for key in (
        "sender_id",
        "current_agent_id",
        "agent_id",
        "teammate_id",
        "worker_id",
        "team_id",
    ):
        value = metadata.get(key)

        if value:
            return str(value)

    teammate = metadata.get("teammate") or metadata.get("agent")

    if teammate is not None:
        teammate_id = getattr(teammate, "teammate_id", None) or getattr(teammate, "agent_id", None)

        if teammate_id:
            return str(teammate_id)

    team = resolve_team(context)

    if team is not None and getattr(team, "team_id", None):
        return str(getattr(team, "team_id"))

    return "agent"


def require_text_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    aliases: Sequence[str] = (),
) -> str:
    value = args.get(key)

    if value is None:
        for alias in aliases:
            value = args.get(alias)

            if value is not None:
                break

    text = str(value or "").strip()

    if not text:
        raise ToolValidationError(f"{key} is required")

    return text


def optional_text_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    aliases: Sequence[str] = (),
    default: str = "",
) -> str:
    value = args.get(key)

    if value is None:
        for alias in aliases:
            value = args.get(alias)

            if value is not None:
                break

    if value is None:
        return default

    return str(value)


def optional_dict_arg(
    args: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = args.get(key)

    if value is None:
        return {}

    if not isinstance(value, Mapping):
        raise ToolValidationError(f"{key} must be an object")

    return dict(value)


def optional_bool_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = args.get(key)

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on"}:
        return True

    if text in {"0", "false", "no", "n", "off"}:
        return False

    raise ToolValidationError(f"{key} must be a boolean")


def optional_int_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    default: int | None = None,
) -> int | None:
    value = args.get(key)

    if value is None:
        return default

    try:
        return int(value)
    except Exception as exc:
        raise ToolValidationError(f"{key} must be an integer") from exc


def optional_float_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    default: float | None = None,
) -> float | None:
    value = args.get(key)

    if value is None:
        return default

    try:
        return float(value)
    except Exception as exc:
        raise ToolValidationError(f"{key} must be a number") from exc


def normalize_recipient_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    if isinstance(value, Sequence):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    raise ToolValidationError("recipient_ids must be a list or comma-separated string")


def message_to_tool_data(message: MailboxMessage) -> dict[str, Any]:
    return message.to_dict()


def messages_to_tool_data(messages: Sequence[MailboxMessage]) -> list[dict[str, Any]]:
    return [
        message_to_tool_data(message)
        for message in messages
    ]


class SendMessageTool(BaseTool):
    name = "send_message"
    description = "Send, broadcast, reply, poll, and manage messages between agents through AgentMailbox."
    risk = ToolRiskLevel.LOW

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "send | broadcast | reply | inbox | outbox | poll | read | ack | archive | delete",
                "default": "send",
            },
            "sender_id": {
                "type": "string",
                "description": "Sender agent id. Optional when runtime metadata provides it.",
            },
            "recipient_id": {
                "type": "string",
                "description": "Recipient agent id for direct send.",
            },
            "recipient_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                },
                "description": "Recipients for broadcast.",
            },
            "role": {
                "type": "string",
                "description": "Optional Team role for broadcast through Team.",
            },
            "content": {
                "type": "string",
                "description": "Message content.",
            },
            "subject": {
                "type": "string",
                "description": "Message subject.",
            },
            "message_type": {
                "type": "string",
                "description": "note | task | result | request | response | control | error | heartbeat",
                "default": "note",
            },
            "message_id": {
                "type": "string",
                "description": "Message id for reply/read/ack/archive/delete.",
            },
            "thread_id": {
                "type": "string",
                "description": "Thread id.",
            },
            "parent_message_id": {
                "type": "string",
                "description": "Parent message id.",
            },
            "task_id": {
                "type": "string",
                "description": "Associated task id.",
            },
            "payload": {
                "type": "object",
                "description": "Structured payload.",
            },
            "metadata": {
                "type": "object",
                "description": "Extra metadata.",
            },
            "include_read": {
                "type": "boolean",
                "default": False,
            },
            "include_archived": {
                "type": "boolean",
                "default": False,
            },
            "include_deleted": {
                "type": "boolean",
                "default": False,
            },
            "mark_read": {
                "type": "boolean",
                "default": False,
            },
            "hard_delete": {
                "type": "boolean",
                "default": False,
            },
            "limit": {
                "type": "integer",
            },
            "timeout": {
                "type": "number",
            },
        },
        "required": [
            "action",
        ],
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)
        action = normalize_action(args.get("action"))

        if not has_message_runtime(context):
            error = (
                "send_message requires AgentMailbox in context.metadata['mailbox'], "
                "or a team/swarm/teammate object with mailbox."
            )

            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"send_message failed: {error}",
                data={
                    "action": action.value,
                    "error_type": "SendMessageRuntimeMissingError",
                },
                error=error,
            )

        try:
            if action == SendMessageAction.SEND:
                return await self.execute_send(call, context, args)

            if action == SendMessageAction.BROADCAST:
                return await self.execute_broadcast(call, context, args)

            if action == SendMessageAction.REPLY:
                return await self.execute_reply(call, context, args)

            if action == SendMessageAction.INBOX:
                return await self.execute_inbox(call, context, args)

            if action == SendMessageAction.OUTBOX:
                return await self.execute_outbox(call, context, args)

            if action == SendMessageAction.POLL:
                return await self.execute_poll(call, context, args)

            if action == SendMessageAction.READ:
                return await self.execute_mark_read(call, context, args)

            if action == SendMessageAction.ACK:
                return await self.execute_ack(call, context, args)

            if action == SendMessageAction.ARCHIVE:
                return await self.execute_archive(call, context, args)

            if action == SendMessageAction.DELETE:
                return await self.execute_delete(call, context, args)

            raise ToolValidationError(f"Unsupported action: {action}")

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"send_message failed: {exc}",
                data={
                    "action": action.value,
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )

    async def execute_send(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        sender_id = resolve_sender_id(args, context)
        recipient_id = require_text_arg(args, "recipient_id", aliases=("to", "to_id"))
        content = require_text_arg(args, "content", aliases=("message", "body"))

        message = await mailbox.send_message(
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            subject=optional_text_arg(args, "subject"),
            message_type=MailboxMessageType(
                str(args.get("message_type") or MailboxMessageType.NOTE.value).strip().lower()
            ),
            payload=optional_dict_arg(args, "payload"),
            metadata=optional_dict_arg(args, "metadata"),
            thread_id=args.get("thread_id"),
            parent_message_id=args.get("parent_message_id"),
            task_id=args.get("task_id"),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Message sent to {recipient_id}: {message.message_id}",
            data={
                "action": "send",
                "message": message_to_tool_data(message),
            },
        )

    async def execute_broadcast(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        team = resolve_team(context)
        mailbox = resolve_mailbox(context)
        sender_id = resolve_sender_id(args, context)
        content = require_text_arg(args, "content", aliases=("message", "body"))
        role = args.get("role")

        if team is not None and role:
            messages = await team.broadcast_message(
                role=str(role),
                subject=optional_text_arg(args, "subject"),
                content=content,
                message_type=str(args.get("message_type") or MailboxMessageType.NOTE.value),
                payload=optional_dict_arg(args, "payload"),
                metadata=optional_dict_arg(args, "metadata"),
            )
        else:
            recipient_ids = normalize_recipient_ids(
                args.get("recipient_ids") or args.get("recipients")
            )

            if not recipient_ids:
                raise ToolValidationError("recipient_ids is required for broadcast")

            messages = await mailbox.broadcast_message(
                sender_id=sender_id,
                recipient_ids=recipient_ids,
                content=content,
                subject=optional_text_arg(args, "subject"),
                message_type=str(args.get("message_type") or MailboxMessageType.NOTE.value),
                payload=optional_dict_arg(args, "payload"),
                metadata=optional_dict_arg(args, "metadata"),
                thread_id=args.get("thread_id"),
                task_id=args.get("task_id"),
            )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Broadcast delivered to {len(messages)} recipient(s).",
            data={
                "action": "broadcast",
                "messages": messages_to_tool_data(messages),
                "count": len(messages),
            },
        )

    async def execute_reply(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        sender_id = resolve_sender_id(args, context)
        message_id = require_text_arg(args, "message_id")
        content = require_text_arg(args, "content", aliases=("message", "body"))

        message = await mailbox.reply_message(
            message_id=message_id,
            sender_id=sender_id,
            content=content,
            subject=optional_text_arg(args, "subject"),
            message_type=str(args.get("message_type") or MailboxMessageType.RESPONSE.value),
            payload=optional_dict_arg(args, "payload"),
            metadata=optional_dict_arg(args, "metadata"),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Reply sent: {message.message_id}",
            data={
                "action": "reply",
                "message": message_to_tool_data(message),
            },
        )

    async def execute_inbox(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        agent_id = str(args.get("agent_id") or args.get("recipient_id") or resolve_sender_id(args, context))

        messages = mailbox.get_inbox(
            agent_id,
            include_read=optional_bool_arg(args, "include_read", default=False),
            include_archived=optional_bool_arg(args, "include_archived", default=False),
            include_deleted=optional_bool_arg(args, "include_deleted", default=False),
            message_type=args.get("message_type"),
            task_id=args.get("task_id"),
            thread_id=args.get("thread_id"),
            limit=optional_int_arg(args, "limit"),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Inbox for {agent_id}: {len(messages)} message(s).",
            data={
                "action": "inbox",
                "agent_id": agent_id,
                "messages": messages_to_tool_data(messages),
                "count": len(messages),
            },
        )

    async def execute_outbox(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        agent_id = str(args.get("agent_id") or args.get("sender_id") or resolve_sender_id(args, context))

        messages = mailbox.get_outbox(
            agent_id,
            include_archived=optional_bool_arg(args, "include_archived", default=True),
            include_deleted=optional_bool_arg(args, "include_deleted", default=False),
            message_type=args.get("message_type"),
            task_id=args.get("task_id"),
            thread_id=args.get("thread_id"),
            limit=optional_int_arg(args, "limit"),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Outbox for {agent_id}: {len(messages)} message(s).",
            data={
                "action": "outbox",
                "agent_id": agent_id,
                "messages": messages_to_tool_data(messages),
                "count": len(messages),
            },
        )

    async def execute_poll(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        agent_id = str(args.get("agent_id") or args.get("recipient_id") or resolve_sender_id(args, context))

        poll_result = await mailbox.poll_messages(
            agent_id,
            limit=optional_int_arg(args, "limit"),
            include_read=optional_bool_arg(args, "include_read", default=False),
            include_archived=optional_bool_arg(args, "include_archived", default=False),
            include_deleted=optional_bool_arg(args, "include_deleted", default=False),
            message_type=args.get("message_type"),
            task_id=args.get("task_id"),
            thread_id=args.get("thread_id"),
            timeout=optional_float_arg(args, "timeout"),
            mark_read=optional_bool_arg(args, "mark_read", default=False),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Polled {len(poll_result.messages)} message(s) for {agent_id}.",
            data={
                "action": "poll",
                "poll_result": poll_result.to_dict(),
            },
        )

    async def execute_mark_read(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        message_id = require_text_arg(args, "message_id")
        agent_id = args.get("agent_id") or args.get("recipient_id")

        message = await mailbox.mark_read(
            message_id,
            agent_id=str(agent_id) if agent_id else None,
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Message marked read: {message.message_id}",
            data={
                "action": "read",
                "message": message_to_tool_data(message),
            },
        )

    async def execute_ack(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        message_id = require_text_arg(args, "message_id")
        agent_id = args.get("agent_id") or args.get("recipient_id")

        message = await mailbox.acknowledge_message(
            message_id,
            agent_id=str(agent_id) if agent_id else None,
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Message acknowledged: {message.message_id}",
            data={
                "action": "ack",
                "message": message_to_tool_data(message),
            },
        )

    async def execute_archive(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        message_id = require_text_arg(args, "message_id")
        agent_id = args.get("agent_id") or args.get("recipient_id")

        message = await mailbox.archive_message(
            message_id,
            agent_id=str(agent_id) if agent_id else None,
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Message archived: {message.message_id}",
            data={
                "action": "archive",
                "message": message_to_tool_data(message),
            },
        )

    async def execute_delete(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
        args: Mapping[str, Any],
    ) -> ToolResult:
        mailbox = resolve_mailbox(context)
        message_id = require_text_arg(args, "message_id")
        agent_id = args.get("agent_id") or args.get("recipient_id")

        deleted = await mailbox.delete_message(
            message_id,
            agent_id=str(agent_id) if agent_id else None,
            hard_delete=optional_bool_arg(args, "hard_delete", default=False),
        )

        return make_result(
            call,
            tool_name=self.name,
            success=True,
            content=f"Message deleted: {message_id}",
            data={
                "action": "delete",
                "message_id": message_id,
                "deleted": deleted,
            },
        )


__all__ = [
    "SendMessageAction",
    "SendMessageRuntimeMissingError",
    "SendMessageTool",
    "SendMessageToolError",
    "has_message_runtime",
    "normalize_action",
    "resolve_mailbox",
    "resolve_sender_id",
    "resolve_team",
]