from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.subagents.base import (
    SubAgentContext,
    SubAgentLLMCallable,
    SubAgentRunResult,
)
from pywork.subagents.manager import (
    SubAgentManager,
    SubAgentTaskRequest,
    create_default_subagent_manager,
)
from pywork.subagents.router import (
    LLMSubAgentRouter,
    SubAgentRouteResult,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


class AgentToolAction(str, Enum):
    LIST_AGENTS = "list_agents"
    DESCRIBE_AGENT = "describe_agent"
    CREATE_AGENT = "create_agent"

    ROUTE = "route"
    RUN = "run"
    ROUTE_AND_RUN = "route_and_run"
    RUN_MANY = "run_many"

    ACTIVE_RUNS = "active_runs"
    HISTORY = "history"

    ABORT_RUN = "abort_run"
    ABORT_AGENT = "abort_agent"
    ABORT_ALL = "abort_all"


class AgentToolError(Exception):
    """agent_tool 基础异常。"""


class AgentToolMissingManagerError(AgentToolError):
    """缺少可用 SubAgentManager。"""


class AgentToolMissingRouterError(AgentToolError):
    """缺少可用 SubAgent Router。"""


@dataclass(slots=True)
class AgentToolRuntimeObjects:
    manager: SubAgentManager
    router: LLMSubAgentRouter | None = None


def normalize_action(value: Any) -> AgentToolAction:
    if isinstance(value, AgentToolAction):
        return value

    try:
        return AgentToolAction(str(value).strip())
    except ValueError as exc:
        valid = ", ".join(action.value for action in AgentToolAction)
        raise ToolValidationError(
            f"Invalid agent action {value!r}. Valid actions: {valid}"
        ) from exc


def get_optional_str(
    arguments: Mapping[str, Any],
    key: str,
) -> str | None:
    value = arguments.get(key)

    if value is None:
        return None

    text = str(value).strip()

    return text or None


def get_required_str(
    arguments: Mapping[str, Any],
    key: str,
) -> str:
    value = get_optional_str(arguments, key)

    if value is None:
        raise ToolValidationError(f"Missing required argument {key!r}")

    return value


def get_bool_arg(
    arguments: Mapping[str, Any],
    key: str,
    default: bool = False,
) -> bool:
    value = arguments.get(key, default)

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    return bool(value)


def get_int_arg(
    arguments: Mapping[str, Any],
    key: str,
    default: int | None = None,
) -> int | None:
    value = arguments.get(key)

    if value is None:
        return default

    if isinstance(value, bool):
        raise ToolValidationError(f"Argument {key!r} must be an integer")

    try:
        return int(value)
    except Exception as exc:
        raise ToolValidationError(f"Argument {key!r} must be an integer") from exc


def get_dict_arg(
    arguments: Mapping[str, Any],
    key: str,
) -> dict[str, Any]:
    value = arguments.get(key)

    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ToolValidationError(f"Argument {key!r} must be an object")

    return dict(value)


def get_list_arg(
    arguments: Mapping[str, Any],
    key: str,
) -> list[Any]:
    value = arguments.get(key)

    if value is None:
        return []

    if not isinstance(value, list):
        raise ToolValidationError(f"Argument {key!r} must be an array")

    return list(value)


def get_context_metadata_value(
    context: ToolExecutionContext,
    *keys: str,
) -> Any:
    for key in keys:
        if key in context.metadata:
            return context.metadata[key]

    return None


def get_tool_definitions_from_context(
    context: ToolExecutionContext,
) -> list[dict[str, Any]]:
    tool_definitions = get_context_metadata_value(
        context,
        "subagent_tool_definitions",
        "tool_definitions",
    )

    if isinstance(tool_definitions, list):
        return [
            dict(definition)
            for definition in tool_definitions
            if isinstance(definition, dict)
        ]

    registry = get_context_metadata_value(
        context,
        "tool_registry",
        "registry",
    )

    if registry is not None and hasattr(registry, "list_definitions"):
        definitions = registry.list_definitions()

        if isinstance(definitions, list):
            return [
                dict(definition)
                for definition in definitions
                if isinstance(definition, dict)
            ]

    return []


def get_llm_from_context(
    context: ToolExecutionContext,
) -> SubAgentLLMCallable | None:
    value = get_context_metadata_value(
        context,
        "subagent_llm",
        "runtime_llm",
        "llm",
    )

    if callable(value):
        return value

    return None


def get_router_llm_from_context(
    context: ToolExecutionContext,
) -> SubAgentLLMCallable | None:
    value = get_context_metadata_value(
        context,
        "subagent_router_llm",
        "router_llm",
    )

    if callable(value):
        return value

    return None


def get_parent_messages_from_args_or_context(
    arguments: Mapping[str, Any],
    context: ToolExecutionContext,
) -> list[Any]:
    explicit = arguments.get("parent_messages")

    if isinstance(explicit, list):
        return list(explicit)

    value = get_context_metadata_value(
        context,
        "parent_messages",
        "agent_messages",
    )

    if isinstance(value, list):
        return list(value)

    agent_state = get_context_metadata_value(
        context,
        "agent_state",
    )

    messages = getattr(agent_state, "messages", None)

    if isinstance(messages, list):
        return list(messages)

    return []


def result_to_data(result: SubAgentRunResult) -> dict[str, Any]:
    return result.to_dict()


def route_to_data(route: SubAgentRouteResult) -> dict[str, Any]:
    return route.to_dict()


def render_json_preview(data: Any) -> str:
    return json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def summarize_run_result(result: SubAgentRunResult) -> str:
    status = result.status.value

    lines = [
        f"SubAgent `{result.name}` finished with status `{status}`.",
    ]

    if result.error:
        lines.append(f"Error: {result.error}")

    if result.content:
        lines.append("")
        lines.append(result.content.strip())

    return "\n".join(lines)


def summarize_many_results(results: Sequence[SubAgentRunResult]) -> str:
    lines = [
        f"Ran {len(results)} SubAgent task(s).",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(
            f"{index}. `{result.name}` → `{result.status.value}`"
        )

        if result.error:
            lines.append(f"   error: {result.error}")

        if result.content:
            first_line = result.content.strip().splitlines()[0]
            lines.append(f"   summary: {first_line}")

    return "\n".join(lines)


def normalize_task_request_from_mapping(
    item: Mapping[str, Any],
    *,
    default_workspace_path: str | Path | None = None,
    default_parent_messages: list[Any] | None = None,
    default_metadata: dict[str, Any] | None = None,
) -> SubAgentTaskRequest:
    agent_name = str(item.get("agent_name") or "").strip()
    task = str(item.get("task") or "").strip()

    if not agent_name:
        raise ToolValidationError("Each task item must include agent_name")

    if not task:
        raise ToolValidationError("Each task item must include task")

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    return SubAgentTaskRequest(
        agent_name=agent_name,
        task=task,
        workspace_path=item.get("workspace_path") or default_workspace_path,
        parent_messages=(
            list(item["parent_messages"])
            if isinstance(item.get("parent_messages"), list)
            else list(default_parent_messages or [])
        ),
        working_memory=(
            dict(item["working_memory"])
            if isinstance(item.get("working_memory"), dict)
            else None
        ),
        metadata={
            **dict(default_metadata or {}),
            **dict(metadata),
        },
        run_id=(
            str(item["run_id"])
            if item.get("run_id") is not None
            else None
        ),
    )


class AgentTool(BaseTool):
    """
    主 Agent 调用 SubAgent 的工具。

    支持：
    - list_agents
    - describe_agent
    - create_agent
    - route
    - run
    - route_and_run
    - run_many
    - active_runs
    - history
    - abort_run
    - abort_agent
    - abort_all
    """

    name = "agent"
    description = (
        "Create, route, run, inspect, and stop PyWork SubAgents "
        "such as planner, reviewer, debugger, verifier, and general."
    )
    risk_level = ToolRiskLevel.LOW

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": (
                    "Action to perform. One of: list_agents, describe_agent, "
                    "create_agent, route, run, route_and_run, run_many, "
                    "active_runs, history, abort_run, abort_agent, abort_all."
                ),
            },
            "agent_name": {
                "type": "string",
                "description": (
                    "SubAgent name or alias, for example planner, reviewer, "
                    "debugger, verifier, general."
                ),
            },
            "task": {
                "type": "string",
                "description": "Task to give to the SubAgent or router.",
            },
            "tasks": {
                "type": "array",
                "description": (
                    "Task list for run_many. Each item should contain "
                    "agent_name and task."
                ),
            },
            "route": {
                "type": "boolean",
                "description": (
                    "When true, route the task with LLM router before running."
                ),
            },
            "concurrent": {
                "type": "boolean",
                "description": "Run many tasks concurrently.",
            },
            "max_concurrency": {
                "type": "integer",
                "description": "Max concurrency for run_many.",
            },
            "run_id": {
                "type": "string",
                "description": "Run id to abort or assign.",
            },
            "reason": {
                "type": "string",
                "description": "Abort reason or additional explanation.",
            },
            "include_disabled": {
                "type": "boolean",
                "description": "Include disabled agents in list_agents.",
            },
            "history_limit": {
                "type": "integer",
                "description": "Max history records to return.",
            },
            "parent_messages": {
                "type": "array",
                "description": (
                    "Optional parent messages copied into SubAgent context."
                ),
            },
            "metadata": {
                "type": "object",
                "description": "Extra metadata attached to the SubAgent run.",
            },
        },
        "required": [
            "action",
        ],
    }

    def __init__(
        self,
        *,
        manager: SubAgentManager | None = None,
        router: LLMSubAgentRouter | None = None,
        llm: SubAgentLLMCallable | None = None,
        router_llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.manager = manager
        self.router = router
        self.llm = llm
        self.router_llm = router_llm
        self.tool_definitions = [
            dict(definition)
            for definition in (tool_definitions or [])
        ]
        self._fallback_runtime: AgentToolRuntimeObjects | None = None

        super().__init__()

    def get_runtime_objects(
        self,
        context: ToolExecutionContext,
    ) -> AgentToolRuntimeObjects:
        manager_from_context = get_context_metadata_value(
            context,
            "subagent_manager",
        )

        if isinstance(manager_from_context, SubAgentManager):
            manager = manager_from_context
        elif self.manager is not None:
            manager = self.manager
        else:
            manager = self.get_or_create_fallback_manager(context)

        router_from_context = get_context_metadata_value(
            context,
            "subagent_router",
        )

        if isinstance(router_from_context, LLMSubAgentRouter):
            router = router_from_context
        elif self.router is not None:
            router = self.router
        else:
            router = self.get_or_create_router(
                manager,
                context,
            )

        return AgentToolRuntimeObjects(
            manager=manager,
            router=router,
        )

    def get_or_create_fallback_manager(
        self,
        context: ToolExecutionContext,
    ) -> SubAgentManager:
        llm = self.llm or get_llm_from_context(context)
        tool_definitions = (
            get_tool_definitions_from_context(context)
            or self.tool_definitions
        )

        if self._fallback_runtime is None:
            manager = create_default_subagent_manager(
                llm=llm,
                tool_definitions=tool_definitions,
                workspace_path=context.workspace_path,
                metadata={
                    "source": "agent_tool.fallback_manager",
                },
            )

            self._fallback_runtime = AgentToolRuntimeObjects(
                manager=manager,
                router=None,
            )

            return manager

        manager = self._fallback_runtime.manager

        if manager.llm is None and llm is not None:
            manager.set_llm(llm)

        if not manager.tool_definitions and tool_definitions:
            manager.set_tool_definitions(tool_definitions)

        return manager

    def get_or_create_router(
        self,
        manager: SubAgentManager,
        context: ToolExecutionContext,
    ) -> LLMSubAgentRouter | None:
        if (
            self._fallback_runtime is not None
            and self._fallback_runtime.manager is manager
            and self._fallback_runtime.router is not None
        ):
            return self._fallback_runtime.router

        router_llm = (
            self.router_llm
            or get_router_llm_from_context(context)
            or manager.llm
        )

        if router_llm is None:
            return None

        router = LLMSubAgentRouter(
            manager=manager,
            llm=router_llm,
            metadata={
                "source": "agent_tool",
            },
        )

        if (
            self._fallback_runtime is not None
            and self._fallback_runtime.manager is manager
        ):
            self._fallback_runtime.router = router

        return router

    def require_router(
        self,
        runtime: AgentToolRuntimeObjects,
    ) -> LLMSubAgentRouter:
        if runtime.router is None:
            raise AgentToolMissingRouterError(
                "Agent router is unavailable. Provide subagent_router, "
                "subagent_router_llm, router_llm, subagent_llm, or llm "
                "in ToolExecutionContext.metadata."
            )

        return runtime.router

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        action = normalize_action(call.arguments.get("action"))
        runtime = self.get_runtime_objects(context)

        metadata = {
            "source": "agent_tool",
            "tool_call_id": call.call_id,
            **get_dict_arg(call.arguments, "metadata"),
        }

        parent_messages = get_parent_messages_from_args_or_context(
            call.arguments,
            context,
        )

        workspace_path = Path(context.workspace_path)

        if action == AgentToolAction.LIST_AGENTS:
            return self._success(
                call,
                content="Available SubAgents listed.",
                data={
                    "action": action.value,
                    "agents": runtime.manager.list_agents(
                        include_disabled=get_bool_arg(
                            call.arguments,
                            "include_disabled",
                            False,
                        )
                    ),
                },
            )

        if action in {
            AgentToolAction.DESCRIBE_AGENT,
            AgentToolAction.CREATE_AGENT,
        }:
            return await self.execute_describe_or_create(
                call,
                runtime,
                action,
            )

        if action == AgentToolAction.ROUTE:
            return await self.execute_route(
                call,
                runtime,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )

        if action == AgentToolAction.RUN:
            return await self.execute_run(
                call,
                runtime,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )

        if action == AgentToolAction.ROUTE_AND_RUN:
            return await self.execute_route_and_run(
                call,
                runtime,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )

        if action == AgentToolAction.RUN_MANY:
            return await self.execute_run_many(
                call,
                runtime,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )

        if action == AgentToolAction.ACTIVE_RUNS:
            return self._success(
                call,
                content="Active SubAgent runs listed.",
                data={
                    "action": action.value,
                    "active_runs": runtime.manager.get_active_runs(),
                },
            )

        if action == AgentToolAction.HISTORY:
            return self._success(
                call,
                content="SubAgent run history listed.",
                data={
                    "action": action.value,
                    "history": runtime.manager.get_history(
                        limit=get_int_arg(
                            call.arguments,
                            "history_limit",
                            None,
                        )
                    ),
                },
            )

        if action == AgentToolAction.ABORT_RUN:
            run_id = get_required_str(call.arguments, "run_id")
            reason = get_optional_str(call.arguments, "reason")

            aborted = runtime.manager.abort_run(
                run_id,
                reason=reason,
            )

            return self._success(
                call,
                content=(
                    f"Abort signal sent to run `{run_id}`."
                    if aborted
                    else f"No active SubAgent run found for `{run_id}`."
                ),
                data={
                    "action": action.value,
                    "run_id": run_id,
                    "aborted": aborted,
                },
            )

        if action == AgentToolAction.ABORT_AGENT:
            agent_name = get_required_str(call.arguments, "agent_name")
            reason = get_optional_str(call.arguments, "reason")

            count = runtime.manager.abort_agent(
                agent_name,
                reason=reason,
            )

            return self._success(
                call,
                content=f"Abort signal sent to {count} `{agent_name}` run(s).",
                data={
                    "action": action.value,
                    "agent_name": agent_name,
                    "aborted_count": count,
                },
            )

        if action == AgentToolAction.ABORT_ALL:
            reason = get_optional_str(call.arguments, "reason")

            count = runtime.manager.abort_all(
                reason=reason,
            )

            return self._success(
                call,
                content=f"Abort signal sent to {count} SubAgent run(s).",
                data={
                    "action": action.value,
                    "aborted_count": count,
                },
            )

        raise ToolValidationError(f"Unsupported agent action: {action.value}")

    async def execute_describe_or_create(
        self,
        call: ToolCall,
        runtime: AgentToolRuntimeObjects,
        action: AgentToolAction,
    ) -> ToolResult:
        agent_name = get_required_str(call.arguments, "agent_name")
        spec = runtime.manager.get_spec(agent_name)

        agent = runtime.manager.create_agent(agent_name)

        data = {
            "action": action.value,
            "agent": spec.to_dict(),
            "instance": {
                "name": agent.name,
                "role": agent.role,
                "description": agent.description,
                "permission_mode": agent.tool_scope.permission_mode,
                "tool_scope": agent.tool_scope.to_dict(),
                "tool_definitions": [
                    definition.get("name")
                    or (
                        definition.get("function", {})
                        if isinstance(definition.get("function"), dict)
                        else {}
                    ).get("name")
                    for definition in agent.get_tool_definitions()
                ],
            },
        }

        content = (
            f"SubAgent `{agent.name}` is available."
            if action == AgentToolAction.DESCRIBE_AGENT
            else f"SubAgent `{agent.name}` created for inspection."
        )

        return self._success(
            call,
            content=content,
            data=data,
        )

    async def execute_route(
        self,
        call: ToolCall,
        runtime: AgentToolRuntimeObjects,
        *,
        workspace_path: str | Path,
        parent_messages: list[Any],
        metadata: dict[str, Any],
    ) -> ToolResult:
        task = get_required_str(call.arguments, "task")
        router = self.require_router(runtime)

        route = await router.route(
            task,
            context=SubAgentContext(
                task=task,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            ),
            metadata=metadata,
        )

        data = {
            "action": AgentToolAction.ROUTE.value,
            "route": route_to_data(route),
        }

        return self._success(
            call,
            content=(
                f"Task routed to `{route.agent_name}` "
                f"with confidence `{route.confidence_label.value}`."
            ),
            data=data,
        )

    async def execute_run(
        self,
        call: ToolCall,
        runtime: AgentToolRuntimeObjects,
        *,
        workspace_path: str | Path,
        parent_messages: list[Any],
        metadata: dict[str, Any],
    ) -> ToolResult:
        task = get_required_str(call.arguments, "task")

        if get_bool_arg(call.arguments, "route", False):
            return await self.execute_route_and_run(
                call,
                runtime,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )

        agent_name = get_required_str(call.arguments, "agent_name")

        result = await runtime.manager.run_agent(
            agent_name,
            task,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata=metadata,
            run_id=get_optional_str(call.arguments, "run_id"),
        )

        data = {
            "action": AgentToolAction.RUN.value,
            "result": result_to_data(result),
        }

        return self._success(
            call,
            content=summarize_run_result(result),
            data=data,
        )

    async def execute_route_and_run(
        self,
        call: ToolCall,
        runtime: AgentToolRuntimeObjects,
        *,
        workspace_path: str | Path,
        parent_messages: list[Any],
        metadata: dict[str, Any],
    ) -> ToolResult:
        task = get_required_str(call.arguments, "task")
        router = self.require_router(runtime)

        route, output = await router.route_and_run(
            task,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata=metadata,
            concurrent_pipeline=get_bool_arg(
                call.arguments,
                "concurrent",
                False,
            ),
        )

        if isinstance(output, list):
            results = output
            content = summarize_many_results(results)
            output_data: Any = [
                result_to_data(result)
                for result in results
            ]
        else:
            content = summarize_run_result(output)
            output_data = result_to_data(output)

        data = {
            "action": AgentToolAction.ROUTE_AND_RUN.value,
            "route": route_to_data(route),
            "result": output_data,
        }

        return self._success(
            call,
            content=content,
            data=data,
        )

    async def execute_run_many(
        self,
        call: ToolCall,
        runtime: AgentToolRuntimeObjects,
        *,
        workspace_path: str | Path,
        parent_messages: list[Any],
        metadata: dict[str, Any],
    ) -> ToolResult:
        raw_tasks = get_list_arg(call.arguments, "tasks")

        if not raw_tasks:
            raise ToolValidationError("run_many requires non-empty tasks array")

        requests: list[SubAgentTaskRequest] = []

        for item in raw_tasks:
            if not isinstance(item, Mapping):
                raise ToolValidationError("Each run_many task must be an object")

            requests.append(
                normalize_task_request_from_mapping(
                    item,
                    default_workspace_path=workspace_path,
                    default_parent_messages=parent_messages,
                    default_metadata=metadata,
                )
            )

        results = await runtime.manager.run_many(
            requests,
            concurrent=get_bool_arg(
                call.arguments,
                "concurrent",
                False,
            ),
            max_concurrency=get_int_arg(
                call.arguments,
                "max_concurrency",
                None,
            ),
        )

        data = {
            "action": AgentToolAction.RUN_MANY.value,
            "results": [
                result_to_data(result)
                for result in results
            ],
        }

        return self._success(
            call,
            content=summarize_many_results(results),
            data=data,
        )

    def _success(
        self,
        call: ToolCall,
        *,
        content: str,
        data: dict[str, Any],
    ) -> ToolResult:
        return ToolResult.success_result(
            call=call,
            content=content,
            data=data,
            metadata={
                "tool": self.name,
            },
        )

    def render_result(
        self,
        result: ToolResult,
    ) -> str:
        if result.content:
            return result.content

        if result.success:
            return render_json_preview(result.data)

        return result.error or "agent tool failed"