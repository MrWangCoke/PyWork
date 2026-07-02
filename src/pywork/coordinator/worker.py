from __future__ import annotations

import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.coordinator.context_modifier import (
    ContextModificationRequest,
    ContextModificationResult,
    WorkerContextModifier,
    create_default_context_modifier,
    normalize_worker_role,
)
from pywork.subagents.base import (
    SubAgentContext,
    SubAgentLLMCallable,
    SubAgentRunResult,
    SubAgentToolScope,
)
from pywork.subagents.manager import (
    SubAgentManager,
    create_default_subagent_manager,
)
from pywork.tasks.task import TaskRecord, TaskStatus, safe_jsonable


class WorkerError(Exception):
    """Worker 基础异常。"""


class WorkerBusyError(WorkerError):
    """Worker 正在运行其他任务。"""


class WorkerExecutionError(WorkerError):
    """Worker 执行异常。"""


class WorkerStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerExecutionMode(str, Enum):
    DIRECT = "direct"
    TASK = "task"


def now_timestamp() -> float:
    return time.time()


def new_worker_id(prefix: str = "worker") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_worker_run_id(prefix: str = "worker_run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_execution_mode(value: WorkerExecutionMode | str | None) -> WorkerExecutionMode:
    if isinstance(value, WorkerExecutionMode):
        return value

    text = str(value or WorkerExecutionMode.DIRECT.value).strip().lower()

    try:
        return WorkerExecutionMode(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in WorkerExecutionMode)
        raise WorkerExecutionError(
            f"Invalid worker execution mode {value!r}. Valid modes: {valid}"
        ) from exc


def default_agent_for_worker_role(worker_role: str) -> str:
    role = normalize_worker_role(worker_role)

    if role in {
        "general",
        "planner",
        "reviewer",
        "debugger",
        "verifier",
    }:
        return role

    return "general"


@dataclass(slots=True)
class WorkerSpec:
    """
    Worker 定义。

    worker_id:
        Worker 实例 id。

    worker_role:
        Worker 角色，例如 planner / reviewer / debugger / verifier。

    agent_name:
        实际交给 SubAgentManager 的 Agent 名称。
        如果为空，会根据 worker_role 自动映射。

    tool_scope:
        可选工具范围，用于限制 Worker 能看到/使用的工具。
    """

    worker_id: str = field(default_factory=new_worker_id)
    worker_role: str = "worker"
    agent_name: str | None = None
    description: str = ""
    workspace_path: str | Path = "."
    tool_scope: SubAgentToolScope | None = None
    max_steps: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_role(self) -> str:
        return normalize_worker_role(self.worker_role)

    @property
    def resolved_agent_name(self) -> str:
        return self.agent_name or default_agent_for_worker_role(self.worker_role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_role": self.normalized_role,
            "agent_name": self.resolved_agent_name,
            "description": self.description,
            "workspace_path": str(self.workspace_path),
            "tool_scope": self.tool_scope.to_dict() if self.tool_scope else None,
            "max_steps": self.max_steps,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True)
class WorkerRunRequest:
    """
    Worker 子任务执行请求。
    """

    task: str
    worker_id: str | None = None
    worker_role: str | None = None
    agent_name: str | None = None

    parent_task: str | None = None
    parent_messages: Sequence[Any] = field(default_factory=tuple)
    shared_memory: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)

    workspace_path: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    context: SubAgentContext | None = None
    tool_scope: SubAgentToolScope | None = None
    max_steps: int | None = None
    llm: SubAgentLLMCallable | None = None

    execution_mode: WorkerExecutionMode | str = WorkerExecutionMode.DIRECT
    wait: bool = True
    timeout_seconds: float | None = None
    max_retries: int = 0
    parent_task_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None

    context_profile_name: str | None = None
    max_messages: int | None = None
    max_chars_per_message: int | None = None
    max_total_chars: int | None = None
    recent_messages: int | None = None


@dataclass(slots=True)
class WorkerRunResult:
    worker_id: str
    worker_role: str
    agent_name: str
    task: str
    status: WorkerStatus
    execution_mode: WorkerExecutionMode
    success: bool
    content: str = ""
    error: str | None = None

    run_id: str | None = None
    task_record_id: str | None = None

    context_result: ContextModificationResult | None = None
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
            "worker_id": self.worker_id,
            "worker_role": self.worker_role,
            "agent_name": self.agent_name,
            "task": self.task,
            "status": self.status.value,
            "execution_mode": self.execution_mode.value,
            "success": self.success,
            "content": self.content,
            "error": self.error,
            "run_id": self.run_id,
            "task_record_id": self.task_record_id,
            "context_result": (
                self.context_result.to_dict()
                if self.context_result is not None
                else None
            ),
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


class CoordinatorWorker:
    """
    Coordinator 使用的 Worker Agent。

    它本身不直接调用 LLM，而是通过 SubAgentManager 调用对应的 SubAgent。
    """

    def __init__(
        self,
        *,
        spec: WorkerSpec | None = None,
        manager: SubAgentManager | None = None,
        context_modifier: WorkerContextModifier | None = None,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.spec = spec or WorkerSpec(
            workspace_path=workspace_path or ".",
        )
        self.metadata = metadata or {}

        self.manager = manager or create_default_subagent_manager(
            llm=llm,
            tool_definitions=tool_definitions,
            workspace_path=workspace_path or self.spec.workspace_path,
            metadata={
                "owner": "CoordinatorWorker",
                "worker_id": self.spec.worker_id,
                **self.metadata,
            },
        )

        self.context_modifier = context_modifier or create_default_context_modifier(
            metadata={
                "owner": "CoordinatorWorker",
                "worker_id": self.spec.worker_id,
            }
        )

        self.status = WorkerStatus.IDLE
        self.current_run_id: str | None = None
        self.current_task_record_id: str | None = None
        self.last_result: WorkerRunResult | None = None

    @property
    def worker_id(self) -> str:
        return self.spec.worker_id

    @property
    def worker_role(self) -> str:
        return self.spec.normalized_role

    @property
    def agent_name(self) -> str:
        return self.spec.resolved_agent_name

    @property
    def is_busy(self) -> bool:
        return self.status == WorkerStatus.RUNNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_role": self.worker_role,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "current_run_id": self.current_run_id,
            "current_task_record_id": self.current_task_record_id,
            "spec": self.spec.to_dict(),
            "metadata": safe_jsonable(self.metadata),
        }

    def build_context_request(
        self,
        request: WorkerRunRequest,
    ) -> ContextModificationRequest:
        worker_id = request.worker_id or self.worker_id
        worker_role = normalize_worker_role(
            request.worker_role or self.worker_role
        )

        workspace_path = (
            request.workspace_path
            or (
                request.context.workspace_path
                if request.context is not None
                else self.spec.workspace_path
            )
        )

        parent_messages = (
            request.parent_messages
            or (
                request.context.parent_messages
                if request.context is not None
                else []
            )
        )

        shared_memory = {
            **dict(
                request.context.working_memory
                if request.context is not None
                else {}
            ),
            **dict(request.shared_memory or {}),
        }

        metadata = {
            **self.metadata,
            **dict(self.spec.metadata),
            **dict(
                request.context.metadata
                if request.context is not None
                else {}
            ),
            **dict(request.metadata or {}),
        }

        return ContextModificationRequest(
            worker_id=worker_id,
            worker_role=worker_role,
            task=request.task,
            workspace_path=workspace_path,
            parent_task=request.parent_task,
            parent_messages=parent_messages,
            shared_memory=shared_memory,
            artifacts=request.artifacts,
            metadata=metadata,
            profile_name=request.context_profile_name,
            max_messages=request.max_messages,
            max_chars_per_message=request.max_chars_per_message,
            max_total_chars=request.max_total_chars,
            recent_messages=request.recent_messages,
        )

    def resolve_agent_name(
        self,
        request: WorkerRunRequest,
    ) -> str:
        requested = request.agent_name or self.agent_name

        return self.manager.resolve_agent_name(requested)

    def resolve_tool_scope(
        self,
        request: WorkerRunRequest,
    ) -> SubAgentToolScope | None:
        return request.tool_scope or self.spec.tool_scope

    def resolve_max_steps(
        self,
        request: WorkerRunRequest,
    ) -> int | None:
        return request.max_steps or self.spec.max_steps

    async def execute(
        self,
        request: WorkerRunRequest,
    ) -> WorkerRunResult:
        if self.is_busy:
            raise WorkerBusyError(f"Worker is already running: {self.worker_id}")

        execution_mode = normalize_execution_mode(request.execution_mode)
        agent_name = self.resolve_agent_name(request)
        run_id = request.run_id or new_worker_run_id()

        started_at = now_timestamp()

        self.status = WorkerStatus.RUNNING
        self.current_run_id = run_id

        context_result: ContextModificationResult | None = None

        try:
            context_request = self.build_context_request(request)
            context_result = self.context_modifier.modify(context_request)
            subagent_context = context_result.to_subagent_context()

            if execution_mode == WorkerExecutionMode.TASK:
                result = await self._execute_with_task_manager(
                    request,
                    agent_name=agent_name,
                    run_id=run_id,
                    subagent_context=subagent_context,
                    context_result=context_result,
                    started_at=started_at,
                )
            else:
                result = await self._execute_direct(
                    request,
                    agent_name=agent_name,
                    run_id=run_id,
                    subagent_context=subagent_context,
                    context_result=context_result,
                    started_at=started_at,
                )

            self.status = result.status
            self.last_result = result
            return result

        except Exception as exc:
            finished_at = now_timestamp()
            self.status = WorkerStatus.FAILED

            result = WorkerRunResult(
                worker_id=request.worker_id or self.worker_id,
                worker_role=normalize_worker_role(
                    request.worker_role or self.worker_role
                ),
                agent_name=agent_name,
                task=request.task,
                status=WorkerStatus.FAILED,
                execution_mode=execution_mode,
                success=False,
                error=str(exc),
                run_id=run_id,
                context_result=context_result,
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "error_type": type(exc).__name__,
                    "worker_error": True,
                },
            )

            self.last_result = result
            return result

        finally:
            self.current_run_id = None
            self.current_task_record_id = None

            if self.status in {
                WorkerStatus.COMPLETED,
                WorkerStatus.FAILED,
                WorkerStatus.CANCELLED,
            }:
                self.status = WorkerStatus.IDLE

    async def _execute_direct(
        self,
        request: WorkerRunRequest,
        *,
        agent_name: str,
        run_id: str,
        subagent_context: SubAgentContext,
        context_result: ContextModificationResult,
        started_at: float,
    ) -> WorkerRunResult:
        subagent_result = await self.manager.run_agent(
            agent_name,
            request.task,
            context=subagent_context,
            workspace_path=subagent_context.workspace_path,
            parent_messages=subagent_context.parent_messages,
            working_memory=subagent_context.working_memory,
            metadata={
                **dict(context_result.metadata),
                **dict(request.metadata or {}),
                "worker_id": self.worker_id,
                "worker_role": self.worker_role,
                "worker_run_id": run_id,
            },
            tool_scope=self.resolve_tool_scope(request),
            max_steps=self.resolve_max_steps(request),
            llm=request.llm,
            run_id=run_id,
        )

        finished_at = now_timestamp()

        status = (
            WorkerStatus.COMPLETED
            if subagent_result.success
            else WorkerStatus.FAILED
        )

        return WorkerRunResult(
            worker_id=self.worker_id,
            worker_role=self.worker_role,
            agent_name=agent_name,
            task=request.task,
            status=status,
            execution_mode=WorkerExecutionMode.DIRECT,
            success=subagent_result.success,
            content=subagent_result.content,
            error=subagent_result.error,
            run_id=run_id,
            context_result=context_result,
            subagent_result=subagent_result,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "worker_execution": "direct",
                "subagent_status": subagent_result.status.value,
            },
        )

    async def _execute_with_task_manager(
        self,
        request: WorkerRunRequest,
        *,
        agent_name: str,
        run_id: str,
        subagent_context: SubAgentContext,
        context_result: ContextModificationResult,
        started_at: float,
    ) -> WorkerRunResult:
        # 关键点：
        # 不要直接 wait=request.wait。
        # 必须先 wait=False 启动 Task，拿到 LocalTaskExecution，
        # 这样 cancel_current() 才能立刻知道 current_task_record_id。
        output = await self.manager.run_agent_task(
            agent_name,
            request.task,
            context=subagent_context,
            workspace_path=subagent_context.workspace_path,
            parent_messages=subagent_context.parent_messages,
            working_memory=subagent_context.working_memory,
            metadata={
                **dict(context_result.metadata),
                **dict(request.metadata or {}),
                "worker_id": self.worker_id,
                "worker_role": self.worker_role,
                "worker_run_id": run_id,
            },
            tool_scope=self.resolve_tool_scope(request),
            max_steps=self.resolve_max_steps(request),
            llm=request.llm,
            run_id=run_id,
            task_id=request.task_id,
            parent_task_id=request.parent_task_id,
            max_retries=request.max_retries,
            timeout_seconds=request.timeout_seconds,
            created_by=f"CoordinatorWorker:{self.worker_id}",
            wait=False,
        )

        if isinstance(output, TaskRecord):
            task_record = output
            self.current_task_record_id = task_record.id
        else:
            self.current_task_record_id = output.task_id

            if request.wait:
                task_record = await output.wait(
                    timeout=request.timeout_seconds,
                )
            else:
                task_record = output.record

        finished_at = now_timestamp()

        status = (
            WorkerStatus.COMPLETED
            if task_record.status == TaskStatus.SUCCEEDED
            else WorkerStatus.CANCELLED
            if task_record.status == TaskStatus.CANCELLED
            else WorkerStatus.FAILED
        )

        content = ""
        error = task_record.error

        if task_record.result is not None:
            if isinstance(task_record.result.value, Mapping):
                content = str(task_record.result.value.get("content") or "")
            elif task_record.result.value is not None:
                content = str(task_record.result.value)

        return WorkerRunResult(
            worker_id=self.worker_id,
            worker_role=self.worker_role,
            agent_name=agent_name,
            task=request.task,
            status=status,
            execution_mode=WorkerExecutionMode.TASK,
            success=task_record.status == TaskStatus.SUCCEEDED,
            content=content,
            error=error,
            run_id=run_id,
            task_record_id=task_record.id,
            context_result=context_result,
            task_record=task_record,
            started_at=started_at,
            finished_at=finished_at,
            metadata={
                "worker_execution": "task",
                "task_status": task_record.status.value,
            },
        )
    async def cancel_current(
        self,
        *,
        reason: str | None = None,
    ) -> bool:
        cancelled = False

        if self.current_task_record_id:
            await self.manager.cancel_agent_task(
                self.current_task_record_id,
                reason=reason or "worker cancelled",
                wait=True,
            )
            cancelled = True

        if self.current_run_id:
            try:
                self.manager.abort_run(
                    self.current_run_id,
                    reason=reason or "worker cancelled",
                )
                cancelled = True
            except Exception:
                pass

        if cancelled:
            self.status = WorkerStatus.CANCELLED

        return cancelled


def create_worker(
    *,
    worker_id: str | None = None,
    worker_role: str = "worker",
    agent_name: str | None = None,
    description: str = "",
    workspace_path: str | Path = ".",
    manager: SubAgentManager | None = None,
    context_modifier: WorkerContextModifier | None = None,
    llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    tool_scope: SubAgentToolScope | None = None,
    max_steps: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> CoordinatorWorker:
    spec = WorkerSpec(
        worker_id=worker_id or new_worker_id(),
        worker_role=worker_role,
        agent_name=agent_name,
        description=description,
        workspace_path=workspace_path,
        tool_scope=tool_scope,
        max_steps=max_steps,
        metadata=metadata or {},
    )

    return CoordinatorWorker(
        spec=spec,
        manager=manager,
        context_modifier=context_modifier,
        llm=llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        metadata=metadata,
    )


__all__ = [
    "CoordinatorWorker",
    "WorkerBusyError",
    "WorkerError",
    "WorkerExecutionError",
    "WorkerExecutionMode",
    "WorkerRunRequest",
    "WorkerRunResult",
    "WorkerSpec",
    "WorkerStatus",
    "create_worker",
    "default_agent_for_worker_role",
    "new_worker_id",
    "new_worker_run_id",
    "normalize_execution_mode",
]