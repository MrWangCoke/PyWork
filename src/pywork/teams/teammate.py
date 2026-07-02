from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    SubAgentLLMCallable,
    SubAgentRunResult,
    SubAgentToolScope,
)
from pywork.subagents.manager import (
    SubAgentManager,
    create_default_subagent_manager,
)
from pywork.tasks.task import TaskRecord, TaskStatus
from pywork.teams.mailbox import (
    AgentMailbox,
    MailboxMessage,
    MailboxMessageStatus,
    MailboxMessageType,
    create_agent_mailbox,
    safe_jsonable,
)


class TeammateError(Exception):
    """Teammate 基础异常。"""


class TeammateBusyError(TeammateError):
    """Teammate 正在执行任务。"""


class TeammateStoppedError(TeammateError):
    """Teammate 已停止。"""


class TeammateStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"
    STOPPED = "stopped"
    FAILED = "failed"


class TeammateExecutionMode(str, Enum):
    DIRECT = "direct"
    TASK = "task"


class TeammateMessageAction(str, Enum):
    NONE = "none"
    ACK = "ack"
    EXECUTE_TASK = "execute_task"
    RESPOND_REQUEST = "respond_request"
    STOP = "stop"
    ERROR = "error"


def now_timestamp() -> float:
    return time.time()


def new_teammate_id(prefix: str = "teammate") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_teammate_run_id(prefix: str = "teammate_run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_execution_mode(
    value: TeammateExecutionMode | str | None,
) -> TeammateExecutionMode:
    if isinstance(value, TeammateExecutionMode):
        return value

    text = str(value or TeammateExecutionMode.DIRECT.value).strip().lower()

    try:
        return TeammateExecutionMode(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in TeammateExecutionMode)
        raise TeammateError(
            f"Invalid teammate execution mode {value!r}. Valid modes: {valid}"
        ) from exc


def normalize_teammate_role(value: str | None) -> str:
    text = str(value or "general").strip().lower()

    aliases = {
        "default": "general",
        "assistant": "general",
        "通用": "general",
        "默认": "general",
        "plan": "planner",
        "planning": "planner",
        "计划": "planner",
        "规划": "planner",
        "review": "reviewer",
        "code_review": "reviewer",
        "审查": "reviewer",
        "评审": "reviewer",
        "debug": "debugger",
        "diagnose": "debugger",
        "调试": "debugger",
        "排错": "debugger",
        "verify": "verifier",
        "test": "verifier",
        "tester": "verifier",
        "验证": "verifier",
        "测试": "verifier",
    }

    return aliases.get(text, text or "general")


def default_agent_for_teammate_role(role: str) -> str:
    normalized = normalize_teammate_role(role)

    if normalized in {
        "general",
        "planner",
        "reviewer",
        "debugger",
        "verifier",
    }:
        return normalized

    return "general"


@dataclass(slots=True)
class TeammateSpec:
    teammate_id: str = field(default_factory=new_teammate_id)
    name: str = ""
    role: str = "general"
    agent_name: str | None = None
    description: str = ""
    workspace_path: str | Path = "."

    tool_scope: SubAgentToolScope | None = None
    max_steps: int | None = None

    auto_ack_messages: bool = True
    auto_read_messages: bool = True
    reply_with_results: bool = True

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_role(self) -> str:
        return normalize_teammate_role(self.role)

    @property
    def resolved_agent_name(self) -> str:
        return self.agent_name or default_agent_for_teammate_role(self.role)

    @property
    def display_name(self) -> str:
        return self.name or self.teammate_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "name": self.name,
            "role": self.normalized_role,
            "agent_name": self.resolved_agent_name,
            "description": self.description,
            "workspace_path": str(self.workspace_path),
            "tool_scope": self.tool_scope.to_dict() if self.tool_scope else None,
            "max_steps": self.max_steps,
            "auto_ack_messages": self.auto_ack_messages,
            "auto_read_messages": self.auto_read_messages,
            "reply_with_results": self.reply_with_results,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class TeammateTaskResult:
    teammate_id: str
    agent_name: str
    task: str
    execution_mode: TeammateExecutionMode
    success: bool
    content: str = ""
    error: str | None = None

    run_id: str | None = None
    task_record_id: str | None = None

    subagent_result: SubAgentRunResult | None = None
    task_record: TaskRecord | None = None

    started_at: float = field(default_factory=now_timestamp)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at is None:
            return None

        return int((self.finished_at - self.started_at) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "execution_mode": self.execution_mode.value,
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "run_id": self.run_id,
            "task_record_id": self.task_record_id,
            "subagent_result": (
                self.subagent_result.to_dict()
                if self.subagent_result is not None
                else None
            ),
            "task_record": (
                self.task_record.to_dict()
                if self.task_record is not None
                else None
            ),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class TeammateMessageHandleResult:
    teammate_id: str
    message_id: str | None
    action: TeammateMessageAction
    handled: bool
    success: bool
    result_message_id: str | None = None
    task_result: TeammateTaskResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "message_id": self.message_id,
            "action": self.action.value,
            "handled": self.handled,
            "success": self.success,
            "result_message_id": self.result_message_id,
            "task_result": (
                self.task_result.to_dict()
                if self.task_result is not None
                else None
            ),
            "error": self.error,
            "metadata": safe_jsonable(self.metadata),
        }


class TeammateAgent:
    """
    Team 中的单个 Teammate Agent。

    职责：
    - 发送消息
    - 轮询邮箱
    - 处理 TASK / REQUEST / CONTROL 消息
    - 调用 SubAgentManager 执行子任务
    - 把结果回复给发送方
    """

    def __init__(
        self,
        *,
        spec: TeammateSpec | None = None,
        mailbox: AgentMailbox | None = None,
        manager: SubAgentManager | None = None,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec or TeammateSpec(
            workspace_path=workspace_path or ".",
        )
        self.metadata = metadata or {}

        self.mailbox = mailbox or create_agent_mailbox(
            metadata={
                "owner": "TeammateAgent",
                "teammate_id": self.spec.teammate_id,
            }
        )

        self.manager = manager or create_default_subagent_manager(
            llm=llm,
            tool_definitions=tool_definitions,
            workspace_path=workspace_path or self.spec.workspace_path,
            metadata={
                "owner": "TeammateAgent",
                "teammate_id": self.spec.teammate_id,
                **self.metadata,
            },
        )

        self.status = TeammateStatus.IDLE
        self.current_run_id: str | None = None
        self.current_task_record_id: str | None = None
        self.last_task_result: TeammateTaskResult | None = None
        self.last_message_result: TeammateMessageHandleResult | None = None
        self._stop_requested = False

    @property
    def teammate_id(self) -> str:
        return self.spec.teammate_id

    @property
    def name(self) -> str:
        return self.spec.display_name

    @property
    def role(self) -> str:
        return self.spec.normalized_role

    @property
    def agent_name(self) -> str:
        return self.spec.resolved_agent_name

    @property
    def is_busy(self) -> bool:
        return self.status == TeammateStatus.RUNNING

    @property
    def is_stopped(self) -> bool:
        return self.status == TeammateStatus.STOPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "teammate_id": self.teammate_id,
            "name": self.name,
            "role": self.role,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "current_run_id": self.current_run_id,
            "current_task_record_id": self.current_task_record_id,
            "spec": self.spec.to_dict(),
            "metadata": safe_jsonable(self.metadata),
        }

    async def send_message(
        self,
        *,
        recipient_id: str,
        content: str,
        subject: str = "",
        message_type: MailboxMessageType | str = MailboxMessageType.NOTE,
        payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        thread_id: str | None = None,
        parent_message_id: str | None = None,
        task_id: str | None = None,
    ) -> MailboxMessage:
        return await self.mailbox.send_message(
            sender_id=self.teammate_id,
            recipient_id=recipient_id,
            content=content,
            subject=subject,
            message_type=message_type,
            payload=payload,
            metadata={
                "sender_role": self.role,
                **dict(metadata or {}),
            },
            thread_id=thread_id,
            parent_message_id=parent_message_id,
            task_id=task_id,
        )

    async def poll_messages(
        self,
        *,
        limit: int | None = None,
        include_read: bool = False,
        timeout: float | None = None,
        mark_read: bool = False,
        message_type: MailboxMessageType | str | None = None,
    ):
        return await self.mailbox.poll_messages(
            self.teammate_id,
            limit=limit,
            include_read=include_read,
            timeout=timeout,
            mark_read=mark_read,
            message_type=message_type,
        )

    async def wait_for_message(
        self,
        *,
        timeout: float | None = None,
        message_type: MailboxMessageType | str | None = None,
        mark_read: bool = False,
    ) -> MailboxMessage | None:
        return await self.mailbox.wait_for_message(
            self.teammate_id,
            timeout=timeout,
            message_type=message_type,
            mark_read=mark_read,
        )

    def build_parent_messages_from_mailbox_message(
        self,
        message: MailboxMessage | None,
    ) -> list[dict[str, Any]]:
        if message is None:
            return []

        return [
            {
                "role": "user",
                "name": message.sender_id,
                "content": message.content,
                "metadata": {
                    "mailbox_message_id": message.message_id,
                    "sender_id": message.sender_id,
                    "recipient_id": message.recipient_id,
                    "message_type": message.message_type.value,
                    "thread_id": message.thread_id,
                    "task_id": message.task_id,
                    "payload": safe_jsonable(message.payload),
                },
            }
        ]

    async def execute_task(
        self,
        task: str,
        *,
        source_message: MailboxMessage | None = None,
        execution_mode: TeammateExecutionMode | str = TeammateExecutionMode.DIRECT,
        wait: bool = True,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        max_retries: int = 0,
        llm: SubAgentLLMCallable | None = None,
    ) -> TeammateTaskResult:
        if self.is_stopped:
            raise TeammateStoppedError(f"Teammate is stopped: {self.teammate_id}")

        if self.is_busy:
            raise TeammateBusyError(f"Teammate is busy: {self.teammate_id}")

        execution_mode = normalize_execution_mode(execution_mode)
        run_id = run_id or new_teammate_run_id()
        started_at = now_timestamp()

        self.status = TeammateStatus.RUNNING
        self.current_run_id = run_id

        try:
            if execution_mode == TeammateExecutionMode.TASK:
                result = await self._execute_task_backed(
                    task,
                    source_message=source_message,
                    wait=wait,
                    timeout_seconds=timeout_seconds,
                    metadata=metadata,
                    run_id=run_id,
                    task_id=task_id,
                    parent_task_id=parent_task_id,
                    max_retries=max_retries,
                    llm=llm,
                    started_at=started_at,
                )
            else:
                result = await self._execute_direct(
                    task,
                    source_message=source_message,
                    metadata=metadata,
                    run_id=run_id,
                    llm=llm,
                    started_at=started_at,
                )

            self.last_task_result = result
            return result

        except Exception as exc:
            finished_at = now_timestamp()
            result = TeammateTaskResult(
                teammate_id=self.teammate_id,
                agent_name=self.agent_name,
                task=task,
                execution_mode=execution_mode,
                success=False,
                error=str(exc),
                run_id=run_id,
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "error_type": type(exc).__name__,
                    "teammate_error": True,
                },
            )
            self.last_task_result = result
            self.status = TeammateStatus.FAILED
            return result

        finally:
            self.current_run_id = None
            self.current_task_record_id = None

            if self.status != TeammateStatus.STOPPED:
                self.status = TeammateStatus.IDLE

    async def _execute_direct(
        self,
        task: str,
        *,
        source_message: MailboxMessage | None,
        metadata: dict[str, Any] | None,
        run_id: str,
        llm: SubAgentLLMCallable | None,
        started_at: float,
    ) -> TeammateTaskResult:
        subagent_result = await self.manager.run_agent(
            self.agent_name,
            task,
            workspace_path=self.spec.workspace_path,
            parent_messages=self.build_parent_messages_from_mailbox_message(source_message),
            metadata={
                "teammate_id": self.teammate_id,
                "teammate_role": self.role,
                "source_message_id": source_message.message_id if source_message else None,
                **dict(metadata or {}),
            },
            tool_scope=self.spec.tool_scope,
            max_steps=self.spec.max_steps,
            llm=llm,
            run_id=run_id,
        )

        finished_at = now_timestamp()

        return TeammateTaskResult(
            teammate_id=self.teammate_id,
            agent_name=self.agent_name,
            task=task,
            execution_mode=TeammateExecutionMode.DIRECT,
            success=subagent_result.success,
            content=subagent_result.content,
            error=subagent_result.error,
            run_id=run_id,
            subagent_result=subagent_result,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "subagent_status": subagent_result.status.value,
            },
        )

    async def _execute_task_backed(
        self,
        task: str,
        *,
        source_message: MailboxMessage | None,
        wait: bool,
        timeout_seconds: float | None,
        metadata: dict[str, Any] | None,
        run_id: str,
        task_id: str | None,
        parent_task_id: str | None,
        max_retries: int,
        llm: SubAgentLLMCallable | None,
        started_at: float,
    ) -> TeammateTaskResult:
        output = await self.manager.run_agent_task(
            self.agent_name,
            task,
            workspace_path=self.spec.workspace_path,
            parent_messages=self.build_parent_messages_from_mailbox_message(source_message),
            metadata={
                "teammate_id": self.teammate_id,
                "teammate_role": self.role,
                "source_message_id": source_message.message_id if source_message else None,
                **dict(metadata or {}),
            },
            tool_scope=self.spec.tool_scope,
            max_steps=self.spec.max_steps,
            llm=llm,
            run_id=run_id,
            task_id=task_id,
            parent_task_id=parent_task_id,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            created_by=f"TeammateAgent:{self.teammate_id}",
            wait=False,
        )

        if isinstance(output, TaskRecord):
            task_record = output
            self.current_task_record_id = task_record.id
        else:
            self.current_task_record_id = output.task_id

            if wait:
                task_record = await output.wait(timeout=timeout_seconds)
            else:
                task_record = output.record

        finished_at = now_timestamp()

        content = ""
        error = task_record.error

        if task_record.result is not None:
            if isinstance(task_record.result.value, Mapping):
                content = str(task_record.result.value.get("content") or "")
            elif task_record.result.value is not None:
                content = str(task_record.result.value)

        return TeammateTaskResult(
            teammate_id=self.teammate_id,
            agent_name=self.agent_name,
            task=task,
            execution_mode=TeammateExecutionMode.TASK,
            success=task_record.status == TaskStatus.SUCCEEDED,
            content=content,
            error=error,
            run_id=run_id,
            task_record_id=task_record.id,
            task_record=task_record,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "task_status": task_record.status.value,
            },
        )

    async def handle_message(
        self,
        message: MailboxMessage,
        *,
        execution_mode: TeammateExecutionMode | str = TeammateExecutionMode.DIRECT,
        timeout_seconds: float | None = None,
    ) -> TeammateMessageHandleResult:
        try:
            if self.spec.auto_read_messages:
                await self.mailbox.mark_read(
                    message.message_id,
                    agent_id=self.teammate_id,
                )

            if message.message_type == MailboxMessageType.CONTROL:
                result = await self._handle_control_message(message)
                self.last_message_result = result
                return result

            if message.message_type == MailboxMessageType.TASK:
                result = await self._handle_task_message(
                    message,
                    execution_mode=execution_mode,
                    timeout_seconds=timeout_seconds,
                )
                self.last_message_result = result
                return result

            if message.message_type == MailboxMessageType.REQUEST:
                result = await self._handle_request_message(
                    message,
                    execution_mode=execution_mode,
                    timeout_seconds=timeout_seconds,
                )
                self.last_message_result = result
                return result

            if self.spec.auto_ack_messages:
                await self.mailbox.acknowledge_message(
                    message.message_id,
                    agent_id=self.teammate_id,
                )

            result = TeammateMessageHandleResult(
                teammate_id=self.teammate_id,
                message_id=message.message_id,
                action=TeammateMessageAction.ACK,
                handled=True,
                success=True,
                metadata={
                    "message_type": message.message_type.value,
                },
            )
            self.last_message_result = result
            return result

        except Exception as exc:
            error_result = TeammateMessageHandleResult(
                teammate_id=self.teammate_id,
                message_id=message.message_id,
                action=TeammateMessageAction.ERROR,
                handled=True,
                success=False,
                error=str(exc),
                metadata={
                    "error_type": type(exc).__name__,
                },
            )
            self.last_message_result = error_result
            return error_result

    async def _handle_task_message(
        self,
        message: MailboxMessage,
        *,
        execution_mode: TeammateExecutionMode | str,
        timeout_seconds: float | None,
    ) -> TeammateMessageHandleResult:
        task = str(message.payload.get("task") or message.content).strip()

        task_result = await self.execute_task(
            task,
            source_message=message,
            execution_mode=execution_mode,
            timeout_seconds=timeout_seconds,
            metadata={
                "mailbox_message_id": message.message_id,
                "mailbox_thread_id": message.thread_id,
                "mailbox_task_id": message.task_id,
            },
            parent_task_id=message.task_id,
        )

        result_message_id: str | None = None

        if self.spec.reply_with_results:
            reply = await self.mailbox.reply_message(
                message_id=message.message_id,
                sender_id=self.teammate_id,
                content=task_result.content or task_result.error or "",
                subject=f"Result: {message.subject}",
                message_type=(
                    MailboxMessageType.RESULT
                    if task_result.success
                    else MailboxMessageType.ERROR
                ),
                payload={
                    "task_result": task_result.to_dict(),
                },
                metadata={
                    "teammate_id": self.teammate_id,
                    "teammate_role": self.role,
                    "success": task_result.success,
                },
            )
            result_message_id = reply.message_id

        if self.spec.auto_ack_messages:
            await self.mailbox.acknowledge_message(
                message.message_id,
                agent_id=self.teammate_id,
            )

        return TeammateMessageHandleResult(
            teammate_id=self.teammate_id,
            message_id=message.message_id,
            action=TeammateMessageAction.EXECUTE_TASK,
            handled=True,
            success=task_result.success,
            result_message_id=result_message_id,
            task_result=task_result,
        )

    async def _handle_request_message(
        self,
        message: MailboxMessage,
        *,
        execution_mode: TeammateExecutionMode | str,
        timeout_seconds: float | None,
    ) -> TeammateMessageHandleResult:
        task = str(
            message.payload.get("task")
            or f"Respond to this request:\n{message.content}"
        ).strip()

        task_result = await self.execute_task(
            task,
            source_message=message,
            execution_mode=execution_mode,
            timeout_seconds=timeout_seconds,
            metadata={
                "mailbox_message_id": message.message_id,
                "mailbox_thread_id": message.thread_id,
                "mailbox_request": True,
            },
        )

        result_message_id: str | None = None

        if self.spec.reply_with_results:
            reply = await self.mailbox.reply_message(
                message_id=message.message_id,
                sender_id=self.teammate_id,
                content=task_result.content or task_result.error or "",
                subject=f"Response: {message.subject}",
                message_type=(
                    MailboxMessageType.RESPONSE
                    if task_result.success
                    else MailboxMessageType.ERROR
                ),
                payload={
                    "task_result": task_result.to_dict(),
                },
                metadata={
                    "teammate_id": self.teammate_id,
                    "teammate_role": self.role,
                    "success": task_result.success,
                },
            )
            result_message_id = reply.message_id

        if self.spec.auto_ack_messages:
            await self.mailbox.acknowledge_message(
                message.message_id,
                agent_id=self.teammate_id,
            )

        return TeammateMessageHandleResult(
            teammate_id=self.teammate_id,
            message_id=message.message_id,
            action=TeammateMessageAction.RESPOND_REQUEST,
            handled=True,
            success=task_result.success,
            result_message_id=result_message_id,
            task_result=task_result,
        )

    async def _handle_control_message(
        self,
        message: MailboxMessage,
    ) -> TeammateMessageHandleResult:
        command = str(
            message.payload.get("command")
            or message.content
            or ""
        ).strip().lower()

        if command in {
            "stop",
            "shutdown",
            "cancel",
            "停止",
            "关闭",
            "取消",
        }:
            await self.stop(reason=f"control message: {command}")

            if self.spec.auto_ack_messages:
                await self.mailbox.acknowledge_message(
                    message.message_id,
                    agent_id=self.teammate_id,
                )

            return TeammateMessageHandleResult(
                teammate_id=self.teammate_id,
                message_id=message.message_id,
                action=TeammateMessageAction.STOP,
                handled=True,
                success=True,
                metadata={
                    "command": command,
                },
            )

        if self.spec.auto_ack_messages:
            await self.mailbox.acknowledge_message(
                message.message_id,
                agent_id=self.teammate_id,
            )

        return TeammateMessageHandleResult(
            teammate_id=self.teammate_id,
            message_id=message.message_id,
            action=TeammateMessageAction.ACK,
            handled=True,
            success=True,
            metadata={
                "command": command,
            },
        )

    async def process_next_message(
        self,
        *,
        timeout: float | None = None,
        execution_mode: TeammateExecutionMode | str = TeammateExecutionMode.DIRECT,
        message_type: MailboxMessageType | str | None = None,
    ) -> TeammateMessageHandleResult:
        if self.is_stopped:
            return TeammateMessageHandleResult(
                teammate_id=self.teammate_id,
                message_id=None,
                action=TeammateMessageAction.NONE,
                handled=False,
                success=False,
                error="teammate stopped",
            )

        self.status = TeammateStatus.WAITING

        message = await self.wait_for_message(
            timeout=timeout,
            message_type=message_type,
            mark_read=False,
        )

        if message is None:
            if self.status != TeammateStatus.STOPPED:
                self.status = TeammateStatus.IDLE

            return TeammateMessageHandleResult(
                teammate_id=self.teammate_id,
                message_id=None,
                action=TeammateMessageAction.NONE,
                handled=False,
                success=True,
                metadata={
                    "timed_out": timeout is not None,
                },
            )

        return await self.handle_message(
            message,
            execution_mode=execution_mode,
            timeout_seconds=timeout,
        )

    async def run_loop(
        self,
        *,
        poll_timeout: float = 0.1,
        max_iterations: int | None = None,
        execution_mode: TeammateExecutionMode | str = TeammateExecutionMode.DIRECT,
    ) -> list[TeammateMessageHandleResult]:
        results: list[TeammateMessageHandleResult] = []
        iterations = 0
        self._stop_requested = False

        while not self._stop_requested and not self.is_stopped:
            if max_iterations is not None and iterations >= max_iterations:
                break

            iterations += 1

            result = await self.process_next_message(
                timeout=poll_timeout,
                execution_mode=execution_mode,
            )

            results.append(result)

        return results

    async def cancel_current(
        self,
        *,
        reason: str | None = None,
    ) -> bool:
        cancelled = False

        if self.current_task_record_id:
            await self.manager.cancel_agent_task(
                self.current_task_record_id,
                reason=reason or "teammate cancelled",
                wait=True,
            )
            cancelled = True

        if self.current_run_id:
            try:
                self.manager.abort_run(
                    self.current_run_id,
                    reason=reason or "teammate cancelled",
                )
                cancelled = True
            except Exception:
                pass

        return cancelled

    async def stop(
        self,
        *,
        reason: str | None = None,
    ) -> None:
        self._stop_requested = True
        await self.cancel_current(reason=reason or "teammate stopped")
        self.status = TeammateStatus.STOPPED


def create_teammate(
    *,
    teammate_id: str | None = None,
    name: str = "",
    role: str = "general",
    agent_name: str | None = None,
    description: str = "",
    workspace_path: str | Path = ".",
    mailbox: AgentMailbox | None = None,
    manager: SubAgentManager | None = None,
    llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    tool_scope: SubAgentToolScope | None = None,
    max_steps: int | None = None,
    auto_ack_messages: bool = True,
    auto_read_messages: bool = True,
    reply_with_results: bool = True,
    metadata: dict[str, Any] | None = None,
) -> TeammateAgent:
    spec = TeammateSpec(
        teammate_id=teammate_id or new_teammate_id(),
        name=name,
        role=role,
        agent_name=agent_name,
        description=description,
        workspace_path=workspace_path,
        tool_scope=tool_scope,
        max_steps=max_steps,
        auto_ack_messages=auto_ack_messages,
        auto_read_messages=auto_read_messages,
        reply_with_results=reply_with_results,
        metadata=metadata or {},
    )

    return TeammateAgent(
        spec=spec,
        mailbox=mailbox,
        manager=manager,
        llm=llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        metadata=metadata,
    )


__all__ = [
    "TeammateAgent",
    "TeammateBusyError",
    "TeammateError",
    "TeammateExecutionMode",
    "TeammateMessageAction",
    "TeammateMessageHandleResult",
    "TeammateSpec",
    "TeammateStatus",
    "TeammateStoppedError",
    "TeammateTaskResult",
    "create_teammate",
    "default_agent_for_teammate_role",
    "new_teammate_id",
    "new_teammate_run_id",
    "normalize_execution_mode",
    "normalize_teammate_role",
]