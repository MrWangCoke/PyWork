from __future__ import annotations

import inspect
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Protocol

from pywork.runtime.state import AgentMessage, AgentState, create_agent_state


SubAgentMessageLike = AgentMessage | dict[str, Any]
ToolDefinition = dict[str, Any]


class SubAgentStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    ABORTED = "aborted"
    FAILED = "failed"


class SubAgentError(Exception):
    """SubAgent 基础异常。"""


class SubAgentAbortError(SubAgentError):
    """SubAgent 被中止。"""


class SubAgentValidationError(SubAgentError):
    """SubAgent 参数异常。"""


@dataclass(slots=True)
class SubAgentAbortSignal:
    """
    SubAgent 中止信号。

    作用：
    - 主 Agent 可以调用 abort()
    - 子 Agent 在关键步骤调用 raise_if_aborted()
    - 后续接入 manager/coordinator 后，可以统一取消所有子 Agent
    """

    reason: str | None = None
    _aborted: bool = False

    @property
    def aborted(self) -> bool:
        return self._aborted

    def abort(self, reason: str | None = None) -> None:
        self._aborted = True
        self.reason = reason or self.reason or "subagent aborted"

    def reset(self) -> None:
        self._aborted = False
        self.reason = None

    def raise_if_aborted(self) -> None:
        if self._aborted:
            raise SubAgentAbortError(self.reason or "subagent aborted")


@dataclass(slots=True, frozen=True)
class SubAgentToolScope:
    """
    子 Agent 独立工具/权限范围。

    allowed_tools:
        None 表示不做白名单限制。
        frozenset 表示只允许这些工具。

    denied_tools:
        黑名单，优先级高于 allowed_tools。

    permission_mode:
        子 Agent 默认权限模式。
        planner/reviewer 通常 readonly。
        debugger/verifier 可以 default，但仍然要经过 PermissionGate。
    """

    allowed_tools: frozenset[str] | None = None
    denied_tools: frozenset[str] = field(default_factory=frozenset)
    permission_mode: str = "readonly"

    def allows(self, tool_name: str) -> bool:
        normalized = normalize_tool_name(tool_name)

        if normalized in self.denied_tools:
            return False

        if self.allowed_tools is None:
            return True

        return normalized in self.allowed_tools

    def filter_tool_names(self, tool_names: Sequence[str]) -> list[str]:
        return [
            tool_name
            for tool_name in tool_names
            if self.allows(tool_name)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": (
                sorted(self.allowed_tools)
                if self.allowed_tools is not None
                else None
            ),
            "denied_tools": sorted(self.denied_tools),
            "permission_mode": self.permission_mode,
        }


@dataclass(slots=True)
class SubAgentContext:
    """
    子 Agent 隔离上下文。

    parent_messages:
        从主 Agent 复制过来的只读上下文。
        子 Agent 会拷贝这些消息进入自己的 AgentState，
        但不会修改主 Agent 的 AgentState。

    working_memory:
        子 Agent 私有临时记忆。
    """

    task: str
    workspace_path: str | Path = "."
    parent_messages: list[SubAgentMessageLike] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    abort_signal: SubAgentAbortSignal = field(default_factory=SubAgentAbortSignal)

    def resolved_workspace_path(self) -> Path:
        return Path(self.workspace_path).expanduser().resolve()


@dataclass(slots=True, frozen=True)
class SubAgentRunRequest:
    task: str
    context: SubAgentContext
    max_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SubAgentRunResult:
    agent_id: str
    name: str
    role: str
    status: SubAgentStatus
    content: str
    state: AgentState
    steps: int
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == SubAgentStatus.COMPLETED

    @property
    def aborted(self) -> bool:
        return self.status == SubAgentStatus.ABORTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "status": self.status.value,
            "content": self.content,
            "state": self.state.to_dict(),
            "steps": self.steps,
            "error": self.error,
            "metadata": self.metadata,
            "success": self.success,
            "aborted": self.aborted,
        }


@dataclass(slots=True, frozen=True)
class SubAgentLLMResponse:
    content: str
    raw: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SubAgentLLMCallable(Protocol):
    def __call__(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[ToolDefinition] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...


def normalize_tool_name(tool_name: str) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def new_subagent_id(prefix: str = "subagent") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def agent_message_to_dict(message: SubAgentMessageLike) -> dict[str, Any]:
    if isinstance(message, AgentMessage):
        return {
            "role": message.role,
            "content": message.content,
            "name": message.name,
            "tool_call_id": message.tool_call_id,
            "metadata": dict(message.metadata),
        }

    if isinstance(message, dict):
        return {
            "role": str(message.get("role", "user")),
            "content": str(message.get("content", "")),
            "name": message.get("name"),
            "tool_call_id": message.get("tool_call_id"),
            "metadata": dict(message.get("metadata", {}))
            if isinstance(message.get("metadata", {}), dict)
            else {},
        }

    return {
        "role": "user",
        "content": str(message),
        "metadata": {},
    }


def add_dict_message_to_agent_state(
    state: AgentState,
    message: dict[str, Any],
) -> None:
    role = str(message.get("role", "user"))
    content = str(message.get("content", ""))

    if role not in {"system", "user", "assistant", "tool", "error"}:
        role = "user"

    state.add_message(
        role,  # type: ignore[arg-type]
        content,
        name=message.get("name"),
        tool_call_id=message.get("tool_call_id"),
        metadata=message.get("metadata") or {},
    )


def copy_messages_to_agent_state(
    state: AgentState,
    messages: Sequence[SubAgentMessageLike],
) -> None:
    for message in messages:
        add_dict_message_to_agent_state(
            state,
            agent_message_to_dict(message),
        )


def get_tool_definition_name(definition: Mapping[str, Any]) -> str:
    """
    兼容两种工具定义结构：

    OpenAI style:
        {"type": "function", "function": {"name": "..."}}

    Simple style:
        {"name": "..."}
    """

    function = definition.get("function")

    if isinstance(function, Mapping):
        return str(function.get("name", ""))

    return str(definition.get("name", ""))


def filter_tool_definitions_by_scope(
    definitions: Sequence[ToolDefinition],
    scope: SubAgentToolScope,
) -> list[ToolDefinition]:
    filtered: list[ToolDefinition] = []

    for definition in definitions:
        tool_name = get_tool_definition_name(definition)

        if not tool_name:
            continue

        if scope.allows(tool_name):
            filtered.append(dict(definition))

    return filtered


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def normalize_llm_response(value: Any) -> SubAgentLLMResponse:
    if isinstance(value, SubAgentLLMResponse):
        return value

    if isinstance(value, str):
        return SubAgentLLMResponse(
            content=value,
            raw=value,
        )

    if isinstance(value, dict):
        content = (
            value.get("content")
            or value.get("text")
            or value.get("message")
            or ""
        )

        metadata = value.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        return SubAgentLLMResponse(
            content=str(content),
            raw=value,
            metadata=dict(metadata),
        )

    content = (
        getattr(value, "content", None)
        or getattr(value, "text", None)
        or str(value)
    )

    metadata = getattr(value, "metadata", None)

    if not isinstance(metadata, dict):
        metadata = {}

    return SubAgentLLMResponse(
        content=str(content),
        raw=value,
        metadata=dict(metadata),
    )


class BaseSubAgent:
    """
    SubAgent 基类。

    第一版能力：
    - 使用独立 AgentState
    - 隔离 parent_messages
    - 独立工具白名单/黑名单
    - 独立 permission_mode
    - 支持 abort signal
    - 支持传入 fake/real LLM callable 跑一次任务

    这一版暂时不做：
    - 多轮工具循环
    - 子 Agent 调用 PermissionGate 执行工具
    - manager/coordinator 调度

    这些放到后续任务实现。
    """

    name: ClassVar[str] = "base"
    role: ClassVar[str] = "base"
    description: ClassVar[str] = "Base subagent"

    default_system_prompt: ClassVar[str] = (
        "You are a PyWork subagent. "
        "Complete the assigned task within your isolated context and tool scope."
    )

    default_allowed_tools: ClassVar[frozenset[str] | None] = None
    default_denied_tools: ClassVar[frozenset[str]] = frozenset()
    default_permission_mode: ClassVar[str] = "readonly"
    default_max_steps: ClassVar[int] = 1

    def __init__(
        self,
        *,
        llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[ToolDefinition] | None = None,
        tool_scope: SubAgentToolScope | None = None,
        max_steps: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.llm = llm
        self.tool_definitions = [
            dict(definition)
            for definition in (tool_definitions or [])
        ]

        self.tool_scope = tool_scope or SubAgentToolScope(
            allowed_tools=self.default_allowed_tools,
            denied_tools=self.default_denied_tools,
            permission_mode=self.default_permission_mode,
        )

        self.max_steps = max_steps or self.default_max_steps
        self.metadata = metadata or {}

    def get_system_prompt(self) -> str:
        return self.default_system_prompt

    def get_tool_definitions(self) -> list[ToolDefinition]:
        return filter_tool_definitions_by_scope(
            self.tool_definitions,
            self.tool_scope,
        )

    def create_state(
        self,
        *,
        context: SubAgentContext,
    ) -> AgentState:
        agent_id = new_subagent_id(self.name)

        state = create_agent_state(
            system_prompt=self.get_system_prompt(),
            max_iterations=self.max_steps,
            metadata={
                "agent_id": agent_id,
                "subagent_name": self.name,
                "subagent_role": self.role,
                "subagent_status": SubAgentStatus.CREATED.value,
                "description": self.description,
                "permission_mode": self.tool_scope.permission_mode,
                "workspace_path": str(context.resolved_workspace_path()),
                "tool_scope": self.tool_scope.to_dict(),
                **self.metadata,
            },
        )

        if context.parent_messages:
            state.add_system_message(
                "Parent context follows. Treat it as read-only background information.",
                metadata={
                    "source": "subagent.parent_context",
                },
            )

            copy_messages_to_agent_state(
                state,
                context.parent_messages,
            )

        return state

    def set_state_status(
        self,
        state: AgentState,
        status: SubAgentStatus,
    ) -> None:
        state.metadata["subagent_status"] = status.value

    def build_request(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        abort_signal: SubAgentAbortSignal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentRunRequest:
        context = context or SubAgentContext(task=task)

        context.task = task

        if abort_signal is not None:
            context.abort_signal = abort_signal

        return SubAgentRunRequest(
            task=task,
            context=context,
            max_steps=self.max_steps,
            metadata=metadata or {},
        )

    def build_llm_messages(
        self,
        state: AgentState,
    ) -> list[dict[str, Any]]:
        return [
            message.to_dict()
            for message in state.messages
            if message.role in {"system", "user", "assistant", "tool"}
        ]

    async def run(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        abort_signal: SubAgentAbortSignal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentRunResult:
        request = self.build_request(
            task,
            context=context,
            abort_signal=abort_signal,
            metadata=metadata,
        )

        state = self.create_state(
            context=request.context,
        )

        agent_id = str(state.metadata["agent_id"])

        try:
            request.context.abort_signal.raise_if_aborted()

            self.set_state_status(
                state,
                SubAgentStatus.RUNNING,
            )
            state.set_thinking()
            state.add_user_message(
                request.task,
                metadata={
                    "source": "subagent.task",
                    **request.metadata,
                },
            )
            state.next_iteration()

            if self.llm is None:
                raise SubAgentValidationError(
                    f"{self.name} has no llm callable configured"
                )

            request.context.abort_signal.raise_if_aborted()

            response = await maybe_await(
                self.llm(
                    self.build_llm_messages(state),
                    tools=self.get_tool_definitions(),
                    metadata={
                        "agent_id": agent_id,
                        "agent_name": self.name,
                        "agent_role": self.role,
                        "permission_mode": self.tool_scope.permission_mode,
                        "workspace_path": str(request.context.resolved_workspace_path()),
                        "tool_scope": self.tool_scope.to_dict(),
                        **request.metadata,
                    },
                )
            )

            request.context.abort_signal.raise_if_aborted()

            llm_response = normalize_llm_response(response)

            state.add_assistant_message(
                llm_response.content,
                metadata={
                    "source": "subagent.llm",
                    **llm_response.metadata,
                },
            )
            state.set_finished()
            self.set_state_status(
                state,
                SubAgentStatus.COMPLETED,
            )

            return SubAgentRunResult(
                agent_id=agent_id,
                name=self.name,
                role=self.role,
                status=SubAgentStatus.COMPLETED,
                content=llm_response.content,
                state=state,
                steps=state.iteration,
                metadata={
                    "tool_scope": self.tool_scope.to_dict(),
                    "llm_metadata": llm_response.metadata,
                },
            )

        except SubAgentAbortError as exc:
            state.set_cancelled(str(exc))
            self.set_state_status(
                state,
                SubAgentStatus.ABORTED,
            )

            return SubAgentRunResult(
                agent_id=agent_id,
                name=self.name,
                role=self.role,
                status=SubAgentStatus.ABORTED,
                content="",
                state=state,
                steps=state.iteration,
                error=str(exc),
                metadata={
                    "tool_scope": self.tool_scope.to_dict(),
                },
            )

        except Exception as exc:
            state.set_error(str(exc))
            self.set_state_status(
                state,
                SubAgentStatus.FAILED,
            )

            return SubAgentRunResult(
                agent_id=agent_id,
                name=self.name,
                role=self.role,
                status=SubAgentStatus.FAILED,
                content="",
                state=state,
                steps=state.iteration,
                error=str(exc),
                metadata={
                    "tool_scope": self.tool_scope.to_dict(),
                    "error_type": type(exc).__name__,
                },
            )