from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    BaseSubAgent,
    SubAgentAbortSignal,
    SubAgentContext,
    SubAgentLLMCallable,
    SubAgentRunResult,
    SubAgentStatus,
    SubAgentToolScope,
    ToolDefinition,
    maybe_await,
    normalize_tool_name,
)
from pywork.tasks.local_task import LocalTaskExecution
from pywork.tasks.task import (
    TaskRecord,
    TaskResult,
    TaskStatus,
    TaskType,
    safe_jsonable,
)
from pywork.tasks.task_manager import (
    TaskManager,
    TaskManagerConfig,
    create_task_manager,
)
from pywork.subagents.debugger import DebuggerSubAgent
from pywork.subagents.general import GeneralSubAgent
from pywork.subagents.planner import PlannerSubAgent
from pywork.subagents.reviewer import ReviewerSubAgent
from pywork.subagents.verifier import VerifierSubAgent


class SubAgentManagerError(Exception):
    """SubAgentManager 基础异常。"""


class SubAgentNotFoundError(SubAgentManagerError):
    """找不到指定 SubAgent。"""


class SubAgentAlreadyRegisteredError(SubAgentManagerError):
    """SubAgent 已经注册。"""


class SubAgentDisabledError(SubAgentManagerError):
    """SubAgent 被禁用。"""


class SubAgentManagerEventType(str, Enum):
    REGISTERED = "registered"
    UNREGISTERED = "unregistered"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"
    CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class SubAgentManagerConfig:
    """
    SubAgentManager 配置。

    max_history_records:
        保留最近多少条运行记录。

    max_concurrent_agents:
        run_many(concurrent=True) 时默认最大并发数。

    default_workspace_path:
        没有显式传 workspace 时使用。

    abort_on_first_failure:
        并发/顺序运行多个 SubAgent 时，一个失败后是否中止剩余任务。
    """

    max_history_records: int = 200
    max_concurrent_agents: int = 4
    default_workspace_path: str | Path = "."
    abort_on_first_failure: bool = False


@dataclass(slots=True)
class SubAgentSpec:
    """
    SubAgent 注册信息。
    """

    name: str
    agent_cls: type[BaseSubAgent]
    aliases: tuple[str, ...] = ()
    description: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_name(self) -> str:
        return normalize_tool_name(self.name)

    def normalized_aliases(self) -> tuple[str, ...]:
        return tuple(
            normalize_tool_name(alias)
            for alias in self.aliases
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.normalized_name(),
            "class_name": self.agent_cls.__name__,
            "aliases": list(self.normalized_aliases()),
            "description": self.description,
            "enabled": self.enabled,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SubAgentTaskRequest:
    """
    运行 SubAgent 的请求。

    agent_name:
        可以是正式名字，也可以是 alias。

    context:
        如果传入，manager 会复制其中的信息，避免直接污染外部 context。

    tool_scope:
        单次运行覆盖默认工具范围。

    llm:
        单次运行覆盖 manager 默认 llm。
    """

    agent_name: str
    task: str
    context: SubAgentContext | None = None
    workspace_path: str | Path | None = None
    parent_messages: list[Any] | None = None
    working_memory: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_scope: SubAgentToolScope | None = None
    max_steps: int | None = None
    llm: SubAgentLLMCallable | None = None
    run_id: str | None = None


@dataclass(slots=True)
class ManagedSubAgentRun:
    """
    manager 内部追踪的一次运行。
    """

    run_id: str
    agent_name: str
    task: str
    abort_signal: SubAgentAbortSignal
    started_at: float
    status: SubAgentStatus = SubAgentStatus.RUNNING
    finished_at: float | None = None
    result: SubAgentRunResult | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at is None:
            return None

        return int((self.finished_at - self.started_at) * 1000)

    def finish(
        self,
        result: SubAgentRunResult,
    ) -> None:
        self.result = result
        self.status = result.status
        self.error = result.error
        self.finished_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": dict(self.metadata),
            "result": self.result.to_dict() if self.result else None,
        }


@dataclass(slots=True, frozen=True)
class SubAgentManagerEvent:
    event_type: SubAgentManagerEventType
    run_id: str | None = None
    agent_name: str | None = None
    task: str | None = None
    status: SubAgentStatus | None = None
    content: str | None = None
    error: str | None = None
    result: SubAgentRunResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "task": self.task,
            "status": self.status.value if self.status else None,
            "content": self.content,
            "error": self.error,
            "result": self.result.to_dict() if self.result else None,
            "metadata": dict(self.metadata),
            "timestamp": self.timestamp,
        }


SubAgentManagerEventHandler = Callable[[SubAgentManagerEvent], Any]


def new_run_id(prefix: str = "subagent_run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def normalize_task_request(
    request: SubAgentTaskRequest | Mapping[str, Any],
) -> SubAgentTaskRequest:
    if isinstance(request, SubAgentTaskRequest):
        return request

    return SubAgentTaskRequest(
        agent_name=str(request["agent_name"]),
        task=str(request["task"]),
        context=request.get("context"),
        workspace_path=request.get("workspace_path"),
        parent_messages=request.get("parent_messages"),
        working_memory=request.get("working_memory"),
        metadata=dict(request.get("metadata") or {}),
        tool_scope=request.get("tool_scope"),
        max_steps=request.get("max_steps"),
        llm=request.get("llm"),
        run_id=request.get("run_id"),
    )


def clone_context_for_run(
    *,
    task: str,
    source: SubAgentContext | None,
    workspace_path: str | Path,
    parent_messages: list[Any] | None,
    working_memory: dict[str, Any] | None,
    metadata: dict[str, Any],
    abort_signal: SubAgentAbortSignal,
) -> SubAgentContext:
    if source is None:
        return SubAgentContext(
            task=task,
            workspace_path=workspace_path,
            parent_messages=list(parent_messages or []),
            working_memory=dict(working_memory or {}),
            metadata=dict(metadata),
            abort_signal=abort_signal,
        )

    return SubAgentContext(
        task=task,
        workspace_path=workspace_path or source.workspace_path,
        parent_messages=list(
            parent_messages
            if parent_messages is not None
            else source.parent_messages
        ),
        working_memory=dict(
            working_memory
            if working_memory is not None
            else source.working_memory
        ),
        metadata={
            **dict(source.metadata),
            **dict(metadata),
        },
        abort_signal=abort_signal,
    )


class SubAgentManager:
    """
    SubAgent 管理器。

    第一版完整职责：
    - 注册/注销 Agent
    - alias 解析
    - 统一创建 Agent 实例
    - 统一传入 llm/tool_definitions/context
    - 运行单个 Agent
    - 顺序/并发运行多个 Agent
    - 运行状态追踪
    - 历史记录
    - 事件回调
    - abort 控制

    它暂时不负责：
    - 自动判断用户任务应该交给哪个 Agent
    - planner -> implement -> verifier -> reviewer 的流程编排
    - 接入 RuntimeGraph
    - 接入 TUI 展示

    这些后面放到 router.py / coordinator.py / runtime integration。
    """

    def __init__(
        self,
        *,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        workspace_path: str | Path | None = None,
        config: SubAgentManagerConfig | None = None,
        task_manager: TaskManager | None = None,
        task_manager_config: TaskManagerConfig | None = None,
        auto_register_defaults: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.tool_definitions = [
            dict(definition)
            for definition in (tool_definitions or [])
        ]
        self.config = config or SubAgentManagerConfig()
        self.workspace_path = (
            Path(workspace_path)
            if workspace_path is not None
            else Path(self.config.default_workspace_path)
        )
        self.metadata = metadata or {}

        self.task_manager = task_manager or create_task_manager(
            config=task_manager_config,
            metadata={
                "owner": "SubAgentManager",
                **dict(metadata or {}),
            },
        )

        self._specs: dict[str, SubAgentSpec] = {}
        self._aliases: dict[str, str] = {}
        self._active_runs: dict[str, ManagedSubAgentRun] = {}
        self._history: list[ManagedSubAgentRun] = []
        self._event_handlers: list[SubAgentManagerEventHandler] = []

        if auto_register_defaults:
            self.register_default_agents()

    # ---------------------------------------------------------------------
    # event
    # ---------------------------------------------------------------------

    def add_event_handler(
        self,
        handler: SubAgentManagerEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: SubAgentManagerEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: SubAgentManagerEvent,
    ) -> None:
        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    # ---------------------------------------------------------------------
    # registry
    # ---------------------------------------------------------------------

    def register_default_agents(self) -> None:
        self.register_agent(
            GeneralSubAgent,
            aliases=("default", "assistant", "general_agent"),
            replace=True,
        )
        self.register_agent(
            PlannerSubAgent,
            aliases=("plan", "planning"),
            replace=True,
        )
        self.register_agent(
            ReviewerSubAgent,
            aliases=("review", "code_review"),
            replace=True,
        )
        self.register_agent(
            DebuggerSubAgent,
            aliases=("debug", "diagnose", "diagnostic"),
            replace=True,
        )
        self.register_agent(
            VerifierSubAgent,
            aliases=("verify", "test", "tester"),
            replace=True,
        )

    def register_agent(
        self,
        agent_cls: type[BaseSubAgent],
        *,
        name: str | None = None,
        aliases: Sequence[str] | None = None,
        description: str | None = None,
        enabled: bool = True,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
    ) -> SubAgentSpec:
        agent_name = normalize_tool_name(name or agent_cls.name)

        if not replace and agent_name in self._specs:
            raise SubAgentAlreadyRegisteredError(
                f"SubAgent already registered: {agent_name}"
            )

        normalized_aliases = tuple(
            normalize_tool_name(alias)
            for alias in (aliases or ())
        )

        for alias in normalized_aliases:
            existing = self._aliases.get(alias)

            if existing is not None and existing != agent_name and not replace:
                raise SubAgentAlreadyRegisteredError(
                    f"SubAgent alias already registered: {alias} -> {existing}"
                )

        old_spec = self._specs.get(agent_name)

        if old_spec is not None:
            for alias in old_spec.normalized_aliases():
                if self._aliases.get(alias) == agent_name:
                    del self._aliases[alias]

        spec = SubAgentSpec(
            name=agent_name,
            agent_cls=agent_cls,
            aliases=normalized_aliases,
            description=description or getattr(agent_cls, "description", ""),
            enabled=enabled,
            metadata=metadata or {},
        )

        self._specs[agent_name] = spec

        for alias in normalized_aliases:
            self._aliases[alias] = agent_name

        return spec

    def unregister_agent(
        self,
        name_or_alias: str,
    ) -> SubAgentSpec:
        agent_name = self.resolve_agent_name(name_or_alias)
        spec = self._specs.pop(agent_name)

        for alias in spec.normalized_aliases():
            if self._aliases.get(alias) == agent_name:
                del self._aliases[alias]

        return spec

    def enable_agent(
        self,
        name_or_alias: str,
    ) -> None:
        spec = self.get_spec(name_or_alias)
        spec.enabled = True

    def disable_agent(
        self,
        name_or_alias: str,
    ) -> None:
        spec = self.get_spec(name_or_alias)
        spec.enabled = False

    def has_agent(
        self,
        name_or_alias: str,
    ) -> bool:
        try:
            self.resolve_agent_name(name_or_alias)
            return True
        except SubAgentNotFoundError:
            return False

    def resolve_agent_name(
        self,
        name_or_alias: str,
    ) -> str:
        normalized = normalize_tool_name(name_or_alias)

        if normalized in self._specs:
            return normalized

        if normalized in self._aliases:
            return self._aliases[normalized]

        raise SubAgentNotFoundError(
            f"Unknown SubAgent: {name_or_alias}"
        )

    def get_spec(
        self,
        name_or_alias: str,
    ) -> SubAgentSpec:
        return self._specs[self.resolve_agent_name(name_or_alias)]

    def list_agents(
        self,
        *,
        include_disabled: bool = False,
    ) -> list[dict[str, Any]]:
        specs = sorted(
            self._specs.values(),
            key=lambda item: item.normalized_name(),
        )

        return [
            spec.to_dict()
            for spec in specs
            if include_disabled or spec.enabled
        ]

    # ---------------------------------------------------------------------
    # context / agent factory
    # ---------------------------------------------------------------------

    def set_llm(
        self,
        llm: SubAgentLLMCallable | None,
    ) -> None:
        self.llm = llm

    def set_tool_definitions(
        self,
        tool_definitions: Sequence[ToolDefinition],
    ) -> None:
        self.tool_definitions = [
            dict(definition)
            for definition in tool_definitions
        ]

    def create_agent(
        self,
        name_or_alias: str,
        *,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseSubAgent:
        spec = self.get_spec(name_or_alias)

        if not spec.enabled:
            raise SubAgentDisabledError(
                f"SubAgent is disabled: {spec.name}"
            )

        agent_metadata = {
            "manager": "SubAgentManager",
            "registered_name": spec.name,
            **dict(spec.metadata),
            **dict(metadata or {}),
        }

        return spec.agent_cls(
            llm=llm or self.llm,
            tool_definitions=tool_definitions or self.tool_definitions,
            tool_scope=tool_scope,
            max_steps=max_steps,
            metadata=agent_metadata,
        )

    def build_context(
        self,
        *,
        task: str,
        source: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        abort_signal: SubAgentAbortSignal | None = None,
    ) -> SubAgentContext:
        return clone_context_for_run(
            task=task,
            source=source,
            workspace_path=workspace_path or self.workspace_path,
            parent_messages=parent_messages,
            working_memory=working_memory,
            metadata=metadata or {},
            abort_signal=abort_signal or SubAgentAbortSignal(),
        )

    # ---------------------------------------------------------------------
    # active / history
    # ---------------------------------------------------------------------

    def get_active_runs(self) -> list[dict[str, Any]]:
        return [
            run.to_dict()
            for run in self._active_runs.values()
        ]

    def get_history(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = self._history[-limit:] if limit else self._history

        return [
            record.to_dict()
            for record in records
        ]

    def clear_history(self) -> None:
        self._history.clear()

    def _append_history(
        self,
        record: ManagedSubAgentRun,
    ) -> None:
        self._history.append(record)

        limit = self.config.max_history_records

        if limit > 0 and len(self._history) > limit:
            del self._history[: len(self._history) - limit]

    # ---------------------------------------------------------------------
    # abort
    # ---------------------------------------------------------------------

    def abort_run(
        self,
        run_id: str,
        reason: str | None = None,
    ) -> bool:
        record = self._active_runs.get(run_id)

        if record is None:
            return False

        record.abort_signal.abort(
            reason or f"run aborted: {run_id}"
        )

        return True

    def abort_agent(
        self,
        name_or_alias: str,
        reason: str | None = None,
    ) -> int:
        agent_name = self.resolve_agent_name(name_or_alias)
        count = 0

        for record in self._active_runs.values():
            if record.agent_name == agent_name:
                record.abort_signal.abort(
                    reason or f"agent aborted: {agent_name}"
                )
                count += 1

        return count

    def abort_all(
        self,
        reason: str | None = None,
    ) -> int:
        count = 0

        for record in self._active_runs.values():
            record.abort_signal.abort(
                reason or "all subagent runs aborted"
            )
            count += 1

        return count

    # ---------------------------------------------------------------------
    # run single
    # ---------------------------------------------------------------------

    async def run_agent(
        self,
        name_or_alias: str,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        llm: SubAgentLLMCallable | None = None,
        run_id: str | None = None,
    ) -> SubAgentRunResult:
        agent_name = self.resolve_agent_name(name_or_alias)
        run_id = run_id or new_run_id(agent_name)
        abort_signal = SubAgentAbortSignal()

        run_metadata = {
            "run_id": run_id,
            "manager": "SubAgentManager",
            "agent_name": agent_name,
            **dict(metadata or {}),
        }

        run_context = self.build_context(
            task=task,
            source=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            working_memory=working_memory,
            metadata=run_metadata,
            abort_signal=abort_signal,
        )

        agent = self.create_agent(
            agent_name,
            llm=llm,
            tool_scope=tool_scope,
            max_steps=max_steps,
            metadata=run_metadata,
        )

        record = ManagedSubAgentRun(
            run_id=run_id,
            agent_name=agent_name,
            task=task,
            abort_signal=abort_signal,
            started_at=time.time(),
            metadata=run_metadata,
        )

        self._active_runs[run_id] = record

        await self.emit_event(
            SubAgentManagerEvent(
                event_type=SubAgentManagerEventType.STARTED,
                run_id=run_id,
                agent_name=agent_name,
                task=task,
                status=SubAgentStatus.RUNNING,
                metadata=run_metadata,
            )
        )

        try:
            result = await agent.run(
                task,
                context=run_context,
                abort_signal=abort_signal,
                metadata=run_metadata,
            )

            result.metadata.setdefault("manager", {})
            result.metadata["manager"] = {
                "run_id": run_id,
                "agent_name": agent_name,
                "duration_ms": None,
                **dict(result.metadata.get("manager") or {}),
            }

            record.finish(result)
            result.metadata["manager"]["duration_ms"] = record.duration_ms

            self._append_history(record)

            event_type = SubAgentManagerEventType.COMPLETED

            if result.status == SubAgentStatus.ABORTED:
                event_type = SubAgentManagerEventType.ABORTED
            elif result.status == SubAgentStatus.FAILED:
                event_type = SubAgentManagerEventType.FAILED

            await self.emit_event(
                SubAgentManagerEvent(
                    event_type=event_type,
                    run_id=run_id,
                    agent_name=agent_name,
                    task=task,
                    status=result.status,
                    content=result.content,
                    error=result.error,
                    result=result,
                    metadata={
                        **run_metadata,
                        "duration_ms": record.duration_ms,
                    },
                )
            )

            return result

        finally:
            self._active_runs.pop(run_id, None)

    # ---------------------------------------------------------------------
    # run many
    # ---------------------------------------------------------------------

    async def run_many(
        self,
        requests: Sequence[SubAgentTaskRequest | Mapping[str, Any]],
        *,
        concurrent: bool = False,
        max_concurrency: int | None = None,
        abort_on_first_failure: bool | None = None,
    ) -> list[SubAgentRunResult]:
        normalized_requests = [
            normalize_task_request(request)
            for request in requests
        ]

        if not concurrent:
            return await self.run_sequence(
                normalized_requests,
                abort_on_first_failure=abort_on_first_failure,
            )

        return await self.run_parallel(
            normalized_requests,
            max_concurrency=max_concurrency,
            abort_on_first_failure=abort_on_first_failure,
        )

    async def run_sequence(
        self,
        requests: Sequence[SubAgentTaskRequest | Mapping[str, Any]],
        *,
        abort_on_first_failure: bool | None = None,
    ) -> list[SubAgentRunResult]:
        normalized_requests = [
            normalize_task_request(request)
            for request in requests
        ]

        should_abort = (
            self.config.abort_on_first_failure
            if abort_on_first_failure is None
            else abort_on_first_failure
        )

        results: list[SubAgentRunResult] = []

        for request in normalized_requests:
            result = await self.run_agent(
                request.agent_name,
                request.task,
                context=request.context,
                workspace_path=request.workspace_path,
                parent_messages=request.parent_messages,
                working_memory=request.working_memory,
                metadata=request.metadata,
                tool_scope=request.tool_scope,
                max_steps=request.max_steps,
                llm=request.llm,
                run_id=request.run_id,
            )

            results.append(result)

            if should_abort and not result.success:
                self.abort_all(
                    reason=f"sequence aborted after failure in {request.agent_name}"
                )
                break

        return results

    async def run_parallel(
        self,
        requests: Sequence[SubAgentTaskRequest | Mapping[str, Any]],
        *,
        max_concurrency: int | None = None,
        abort_on_first_failure: bool | None = None,
    ) -> list[SubAgentRunResult]:
        normalized_requests = [
            normalize_task_request(request)
            for request in requests
        ]

        concurrency = max_concurrency or self.config.max_concurrent_agents
        concurrency = max(1, concurrency)

        should_abort = (
            self.config.abort_on_first_failure
            if abort_on_first_failure is None
            else abort_on_first_failure
        )

        semaphore = asyncio.Semaphore(concurrency)

        async def run_one(
            request: SubAgentTaskRequest,
        ) -> SubAgentRunResult:
            async with semaphore:
                result = await self.run_agent(
                    request.agent_name,
                    request.task,
                    context=request.context,
                    workspace_path=request.workspace_path,
                    parent_messages=request.parent_messages,
                    working_memory=request.working_memory,
                    metadata=request.metadata,
                    tool_scope=request.tool_scope,
                    max_steps=request.max_steps,
                    llm=request.llm,
                    run_id=request.run_id,
                )

                if should_abort and not result.success:
                    self.abort_all(
                        reason=f"parallel aborted after failure in {request.agent_name}"
                    )

                return result

        tasks = [
            asyncio.create_task(run_one(request))
            for request in normalized_requests
        ]

        return list(await asyncio.gather(*tasks))

    # ---------------------------------------------------------------------
    # result helpers
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # task integration
    # ---------------------------------------------------------------------

    def tool_scope_to_payload(
        self,
        tool_scope: SubAgentToolScope | None,
    ) -> dict[str, Any] | None:
        if tool_scope is None:
            return None

        return tool_scope.to_dict()

    def tool_scope_from_payload(
        self,
        payload: dict[str, Any] | None,
    ) -> SubAgentToolScope | None:
        if not payload:
            return None

        allowed_tools = payload.get("allowed_tools")
        denied_tools = payload.get("denied_tools") or []

        return SubAgentToolScope(
            allowed_tools=(
                frozenset(str(item) for item in allowed_tools)
                if isinstance(allowed_tools, list)
                else None
            ),
            denied_tools=frozenset(str(item) for item in denied_tools),
            permission_mode=str(payload.get("permission_mode") or "readonly"),
        )

    def build_task_payload_from_request(
        self,
        request: SubAgentTaskRequest,
    ) -> dict[str, Any]:
        context = request.context

        workspace_path = (
            request.workspace_path
            or (context.workspace_path if context is not None else self.workspace_path)
        )

        parent_messages = (
            request.parent_messages
            if request.parent_messages is not None
            else (context.parent_messages if context is not None else [])
        )

        working_memory = (
            request.working_memory
            if request.working_memory is not None
            else (context.working_memory if context is not None else {})
        )

        return {
            "agent_name": request.agent_name,
            "task": request.task,
            "workspace_path": str(workspace_path),
            "parent_messages": safe_jsonable(list(parent_messages or [])),
            "working_memory": safe_jsonable(dict(working_memory or {})),
            "metadata": safe_jsonable(dict(request.metadata or {})),
            "tool_scope": self.tool_scope_to_payload(request.tool_scope),
            "max_steps": request.max_steps,
            "run_id": request.run_id,
        }

    def build_request_from_task_record(
        self,
        record: TaskRecord,
    ) -> SubAgentTaskRequest:
        payload = dict(record.payload or {})

        agent_name = str(payload.get("agent_name") or record.agent_id or "general")
        task = str(payload.get("task") or record.name)

        return SubAgentTaskRequest(
            agent_name=agent_name,
            task=task,
            workspace_path=payload.get("workspace_path") or self.workspace_path,
            parent_messages=list(payload.get("parent_messages") or []),
            working_memory=dict(payload.get("working_memory") or {}),
            metadata={
                **dict(payload.get("metadata") or {}),
                "task_id": record.id,
                "parent_task_id": record.parent_id,
            },
            tool_scope=self.tool_scope_from_payload(
                payload.get("tool_scope")
                if isinstance(payload.get("tool_scope"), dict)
                else None
            ),
            max_steps=payload.get("max_steps"),
            run_id=payload.get("run_id") or f"{record.id}_subagent_run",
        )

    def make_agent_task_runner(
        self,
        request: SubAgentTaskRequest,
    ):
        async def runner(record: TaskRecord) -> TaskResult:
            task_request = self.build_request_from_task_record(record)

            # 保留创建 Task 时传入的临时对象，例如 llm/tool_scope。
            if request.llm is not None:
                task_request.llm = request.llm

            if request.tool_scope is not None:
                task_request.tool_scope = request.tool_scope

            if request.context is not None:
                task_request.context = request.context

            result = await self.run_agent(
                task_request.agent_name,
                task_request.task,
                context=task_request.context,
                workspace_path=task_request.workspace_path,
                parent_messages=task_request.parent_messages,
                working_memory=task_request.working_memory,
                metadata={
                    **dict(task_request.metadata or {}),
                    "task_id": record.id,
                    "task_retry_count": record.retry_count,
                },
                tool_scope=task_request.tool_scope,
                max_steps=task_request.max_steps,
                llm=task_request.llm,
                run_id=task_request.run_id,
            )

            value = result.to_dict()

            metadata = {
                "task_id": record.id,
                "agent_name": result.name,
                "agent_role": result.role,
                "subagent_status": result.status.value,
                "subagent_success": result.success,
            }

            if result.success:
                return TaskResult.success_result(
                    value,
                    metadata=metadata,
                )

            return TaskResult.failure_result(
                result.error or f"SubAgent {result.name} failed",
                error_type=f"SubAgentStatus:{result.status.value}",
                value=value,
                metadata=metadata,
            )

        return runner

    async def create_agent_task(
        self,
        agent_name: str,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        llm: SubAgentLLMCallable | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        created_by: str | None = None,
    ) -> TaskRecord:
        resolved_agent_name = self.resolve_agent_name(agent_name)

        request = SubAgentTaskRequest(
            agent_name=resolved_agent_name,
            task=task,
            context=context,
            workspace_path=workspace_path or self.workspace_path,
            parent_messages=parent_messages,
            working_memory=working_memory,
            metadata={
                **dict(metadata or {}),
                "subagent_task": True,
            },
            tool_scope=tool_scope,
            max_steps=max_steps,
            llm=llm,
            run_id=run_id,
        )

        payload = self.build_task_payload_from_request(request)

        record = await self.task_manager.create_task(
            f"SubAgent {resolved_agent_name}: {task}",
            task_type=TaskType.SUBAGENT,
            payload=payload,
            parent_id=parent_task_id,
            agent_id=resolved_agent_name,
            metadata={
                "subagent_manager": True,
                "agent_name": resolved_agent_name,
                **dict(metadata or {}),
            },
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            created_by=created_by or "SubAgentManager",
            task_id=task_id,
            runner=self.make_agent_task_runner(request),
        )

        return record

    async def start_agent_task(
        self,
        task_id: str,
    ) -> LocalTaskExecution:
        record = await self.task_manager.get_task(task_id)
        request = self.build_request_from_task_record(record)

        self.task_manager.set_runner(
            task_id,
            self.make_agent_task_runner(request),
        )

        return await self.task_manager.start_task(
            task_id,
            agent_id=record.agent_id,
            task_name=record.name,
        )

    async def run_agent_task(
        self,
        agent_name: str,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        working_memory: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        llm: SubAgentLLMCallable | None = None,
        run_id: str | None = None,
        task_id: str | None = None,
        parent_task_id: str | None = None,
        max_retries: int = 0,
        timeout_seconds: float | None = None,
        created_by: str | None = None,
        wait: bool = True,
    ) -> TaskRecord | LocalTaskExecution:
        record = await self.create_agent_task(
            agent_name,
            task,
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            working_memory=working_memory,
            metadata=metadata,
            tool_scope=tool_scope,
            max_steps=max_steps,
            llm=llm,
            run_id=run_id,
            task_id=task_id,
            parent_task_id=parent_task_id,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds,
            created_by=created_by,
        )

        execution = await self.start_agent_task(record.id)

        if not wait:
            return execution

        return await execution.wait(
            timeout=timeout_seconds,
        )

    async def run_many_agent_tasks(
        self,
        requests: Sequence[SubAgentTaskRequest | Mapping[str, Any]],
        *,
        concurrent: bool = False,
        max_concurrency: int | None = None,
        parent_task_id: str | None = None,
        wait: bool = True,
    ) -> list[TaskRecord | LocalTaskExecution]:
        normalized_requests = [
            normalize_task_request(request)
            for request in requests
        ]

        async def create_and_maybe_start(
            request: SubAgentTaskRequest,
        ) -> TaskRecord | LocalTaskExecution:
            return await self.run_agent_task(
                request.agent_name,
                request.task,
                context=request.context,
                workspace_path=request.workspace_path,
                parent_messages=request.parent_messages,
                working_memory=request.working_memory,
                metadata=request.metadata,
                tool_scope=request.tool_scope,
                max_steps=request.max_steps,
                llm=request.llm,
                run_id=request.run_id,
                parent_task_id=parent_task_id,
                wait=wait,
            )

        if not concurrent:
            results: list[TaskRecord | LocalTaskExecution] = []

            for request in normalized_requests:
                results.append(
                    await create_and_maybe_start(request)
                )

            return results

        concurrency = max_concurrency or self.config.max_concurrent_agents
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_one(
            request: SubAgentTaskRequest,
        ) -> TaskRecord | LocalTaskExecution:
            async with semaphore:
                return await create_and_maybe_start(request)

        return list(
            await asyncio.gather(
                *[
                    run_one(request)
                    for request in normalized_requests
                ]
            )
        )

    async def get_agent_task(
        self,
        task_id: str,
    ) -> TaskRecord:
        return await self.task_manager.get_task(task_id)

    async def list_agent_tasks(
        self,
        *,
        status: TaskStatus | str | None = None,
        parent_task_id: str | None = None,
        agent_id: str | None = None,
        limit: int | None = None,
    ) -> list[TaskRecord]:
        return await self.task_manager.list_tasks(
            status=status,
            parent_id=parent_task_id,
            agent_id=agent_id,
            limit=limit,
        )

    async def cancel_agent_task(
        self,
        task_id: str,
        *,
        reason: str | None = None,
        wait: bool = True,
    ) -> TaskRecord:
        return await self.task_manager.cancel_task(
            task_id,
            reason=reason,
            wait=wait,
        )

    async def retry_agent_task(
        self,
        task_id: str,
        *,
        wait: bool = True,
        timeout: float | None = None,
    ) -> LocalTaskExecution | TaskRecord:
        record = await self.task_manager.get_task(task_id)
        request = self.build_request_from_task_record(record)

        self.task_manager.set_runner(
            task_id,
            self.make_agent_task_runner(request),
        )

        return await self.task_manager.retry_task(
            task_id,
            agent_id=record.agent_id,
            wait=wait,
            timeout=timeout,
        )

    async def watch_agent_task(
        self,
        task_id: str,
        *,
        poll_interval: float | None = None,
        timeout: float | None = None,
        include_duplicates: bool = False,
    ) -> AsyncIterator[TaskRecord]:
        async for record in self.task_manager.watch_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            include_duplicates=include_duplicates,
        ):
            yield record

    def summarize_results(
        self,
        results: Sequence[SubAgentRunResult],
    ) -> str:
        lines: list[str] = []

        for result in results:
            status = result.status.value
            lines.append(
                f"- {result.name}: {status}"
            )

            if result.error:
                lines.append(
                    f"  error: {result.error}"
                )

            if result.content:
                first_line = result.content.strip().splitlines()[0]
                lines.append(
                    f"  summary: {first_line}"
                )

        return "\n".join(lines)


def create_default_subagent_manager(
    *,
    llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[ToolDefinition] | None = None,
    workspace_path: str | Path | None = None,
    config: SubAgentManagerConfig | None = None,
    task_manager: TaskManager | None = None,
    task_manager_config: TaskManagerConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> SubAgentManager:
    return SubAgentManager(
        llm=llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        config=config,
        task_manager=task_manager,
        task_manager_config=task_manager_config,
        auto_register_defaults=True,
        metadata=metadata,
    )