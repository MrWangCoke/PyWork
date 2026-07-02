from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.subagents.manager import (
    SubAgentManager,
    SubAgentTaskRequest,
    create_default_subagent_manager,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


COORDINATOR_ACTION_RUN = "run"
COORDINATOR_STRATEGY_PARALLEL = "parallel"
COORDINATOR_STRATEGY_SEQUENTIAL = "sequential"
COORDINATOR_EXECUTION_MODE_TASK = "task"
COORDINATOR_EXECUTION_MODE_DIRECT = "direct"


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


def infer_agent_name(task: str) -> str:
    lowered = task.lower()

    if any(word in lowered for word in ["review", "code review", "审查", "审核", "代码审查"]):
        return "reviewer"

    if any(word in lowered for word in ["test", "pytest", "verify", "验证", "测试", "运行测试"]):
        return "verifier"

    if any(word in lowered for word in ["debug", "diagnose", "调试", "排查", "修 bug", "修bug"]):
        return "debugger"

    if any(word in lowered for word in ["plan", "planning", "规划", "计划", "拆解", "方案"]):
        return "planner"

    return "general"


def normalize_strategy(value: Any) -> str:
    text = str(value or COORDINATOR_STRATEGY_PARALLEL).strip().lower()

    aliases = {
        "concurrent": "parallel",
        "并行": "parallel",
        "并发": "parallel",
        "同时": "parallel",
        "serial": "sequential",
        "sequence": "sequential",
        "顺序": "sequential",
    }

    text = aliases.get(text, text)

    if text not in {
        COORDINATOR_STRATEGY_PARALLEL,
        COORDINATOR_STRATEGY_SEQUENTIAL,
    }:
        raise ToolValidationError(
            "strategy must be 'parallel' or 'sequential'"
        )

    return text


def normalize_execution_mode(value: Any) -> str:
    text = str(value or COORDINATOR_EXECUTION_MODE_TASK).strip().lower()

    aliases = {
        "background": "task",
        "task_manager": "task",
        "后台": "task",
        "direct": "direct",
        "直接": "direct",
    }

    text = aliases.get(text, text)

    if text not in {
        COORDINATOR_EXECUTION_MODE_TASK,
        COORDINATOR_EXECUTION_MODE_DIRECT,
    }:
        raise ToolValidationError(
            "execution_mode must be 'task' or 'direct'"
        )

    return text


def normalize_step(
    item: Any,
    *,
    index: int,
    default_metadata: dict[str, Any] | None = None,
) -> SubAgentTaskRequest:
    metadata = {
        "coordinator": "CoordinatorTool",
        "worker_index": index,
        **dict(default_metadata or {}),
    }

    if isinstance(item, str):
        task = item.strip()

        if not task:
            raise ToolValidationError("task step cannot be empty")

        agent_name = infer_agent_name(task)

        return SubAgentTaskRequest(
            agent_name=agent_name,
            task=task,
            metadata=metadata,
        )

    if not isinstance(item, Mapping):
        raise ToolValidationError(
            "Each coordinator step must be a string or object"
        )

    task = str(
        item.get("task")
        or item.get("title")
        or item.get("description")
        or ""
    ).strip()

    if not task:
        raise ToolValidationError(
            "Each coordinator step object must include 'task'"
        )

    step_metadata = item.get("metadata")

    if not isinstance(step_metadata, dict):
        step_metadata = {}

    agent_name = str(
        item.get("agent_name")
        or item.get("agent")
        or item.get("role")
        or infer_agent_name(task)
    ).strip()

    return SubAgentTaskRequest(
        agent_name=agent_name,
        task=task,
        workspace_path=item.get("workspace_path"),
        parent_messages=(
            list(item["parent_messages"])
            if isinstance(item.get("parent_messages"), list)
            else None
        ),
        working_memory=(
            dict(item["working_memory"])
            if isinstance(item.get("working_memory"), dict)
            else None
        ),
        metadata={
            **metadata,
            **dict(step_metadata),
            "worker_id": str(item.get("worker_id") or f"worker_{index}"),
        },
        run_id=(
            str(item["run_id"])
            if item.get("run_id") is not None
            else None
        ),
    )


def normalize_steps(
    arguments: Mapping[str, Any],
) -> list[SubAgentTaskRequest]:
    raw_steps = arguments.get("steps")

    if raw_steps is None:
        raw_steps = arguments.get("tasks")

    if not isinstance(raw_steps, list) or not raw_steps:
        raise ToolValidationError(
            "CoordinatorTool requires a non-empty 'steps' or 'tasks' array"
        )

    default_metadata = get_dict_arg(arguments, "metadata")

    return [
        normalize_step(
            item,
            index=index,
            default_metadata=default_metadata,
        )
        for index, item in enumerate(raw_steps, start=1)
    ]


def object_to_data(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, list | tuple):
        return [
            object_to_data(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): object_to_data(item)
            for key, item in value.items()
        }

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            return object_to_data(to_dict())
        except Exception:
            pass

    model_dump = getattr(value, "model_dump", None)

    if callable(model_dump):
        try:
            return object_to_data(model_dump())
        except Exception:
            pass

    return str(value)


def summarize_worker_data(
    workers: Sequence[dict[str, Any]],
) -> str:
    lines: list[str] = []

    for index, worker in enumerate(workers, start=1):
        agent_name = (
            worker.get("agent_name")
            or worker.get("name")
            or worker.get("agent")
            or "-"
        )
        status = (
            worker.get("status")
            or worker.get("task_status")
            or worker.get("subagent_status")
            or "-"
        )
        task_id = (
            worker.get("id")
            or worker.get("task_id")
            or worker.get("run_id")
            or "-"
        )

        lines.append(
            f"{index}. `{agent_name}` → `{status}` ({task_id})"
        )

        result = worker.get("result")

        if isinstance(result, dict):
            content = str(result.get("content") or "").strip()

            if content:
                first_line = content.splitlines()[0]
                lines.append(f"   summary: {first_line}")

        error = worker.get("error")

        if error:
            lines.append(f"   error: {error}")

    return "\n".join(lines)


def get_context_manager(
    context: ToolExecutionContext,
) -> SubAgentManager:
    metadata = context.metadata or {}

    manager = metadata.get("subagent_manager") or metadata.get("manager")

    if isinstance(manager, SubAgentManager):
        return manager

    task_manager = metadata.get("task_manager")
    tool_definitions = metadata.get("tool_definitions")
    registry = metadata.get("tool_registry") or metadata.get("registry")

    if not isinstance(tool_definitions, list):
        list_definitions = getattr(registry, "list_definitions", None)

        if callable(list_definitions):
            tool_definitions = list_definitions()
        else:
            tool_definitions = []

    manager = create_default_subagent_manager(
        workspace_path=context.workspace_path,
        task_manager=task_manager,
        tool_definitions=tool_definitions,
        metadata={
            "source": "coordinator_tool.fallback_manager",
        },
    )

    metadata["subagent_manager"] = manager
    metadata["manager"] = manager
    metadata["task_manager"] = manager.task_manager

    return manager


class CoordinatorTool(BaseTool):
    """
    Coordinator 工具。

    用于：
    - 把多步任务拆成多个 worker
    - 并行或顺序执行
    - 默认使用 TaskManager 后台任务模式，方便 TUI TaskPanel 展示进度
    """

    name = "coordinator"
    description = (
        "Coordinate multiple SubAgent worker tasks. Use this when the user asks "
        "to run several tasks in parallel or sequentially and return a summary."
    )
    risk_level = ToolRiskLevel.LOW

    input_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Action to perform. Currently supports: run.",
            },
            "strategy": {
                "type": "string",
                "description": "parallel or sequential. Defaults to parallel.",
            },
            "execution_mode": {
                "type": "string",
                "description": (
                    "task or direct. Defaults to task so TaskPanel can show progress."
                ),
            },
            "steps": {
                "type": "array",
                "description": (
                    "Worker steps. Each item can be a string or an object with "
                    "agent_name/task/metadata."
                ),
            },
            "tasks": {
                "type": "array",
                "description": "Alias of steps.",
            },
            "wait": {
                "type": "boolean",
                "description": "Whether to wait for task-backed workers to finish.",
            },
            "max_concurrency": {
                "type": "integer",
                "description": "Maximum number of concurrent workers.",
            },
            "parent_task_id": {
                "type": "string",
                "description": "Optional parent TaskManager task id.",
            },
            "metadata": {
                "type": "object",
                "description": "Extra metadata for worker tasks.",
            },
        },
        "required": [
            "action",
        ],
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        action = str(call.arguments.get("action") or "").strip().lower()

        if action != COORDINATOR_ACTION_RUN:
            raise ToolValidationError(
                "CoordinatorTool currently supports only action='run'"
            )

        strategy = normalize_strategy(call.arguments.get("strategy"))
        execution_mode = normalize_execution_mode(
            call.arguments.get("execution_mode")
        )
        concurrent = strategy == COORDINATOR_STRATEGY_PARALLEL
        max_concurrency = get_int_arg(
            call.arguments,
            "max_concurrency",
            None,
        )
        wait = get_bool_arg(
            call.arguments,
            "wait",
            True,
        )
        parent_task_id = call.arguments.get("parent_task_id")
        parent_task_id = str(parent_task_id) if parent_task_id else None

        manager = get_context_manager(context)
        requests = normalize_steps(call.arguments)

        if execution_mode == COORDINATOR_EXECUTION_MODE_TASK:
            workers = await self.run_task_backed_workers(
                manager,
                requests,
                concurrent=concurrent,
                max_concurrency=max_concurrency,
                parent_task_id=parent_task_id,
                wait=wait,
            )
        else:
            workers = await self.run_direct_workers(
                manager,
                requests,
                concurrent=concurrent,
                max_concurrency=max_concurrency,
            )

        worker_data = [
            object_to_data(worker)
            for worker in workers
        ]

        content = (
            f"Coordinator finished {len(worker_data)} worker task(s) "
            f"with strategy `{strategy}` and execution mode `{execution_mode}`.\n\n"
            f"{summarize_worker_data(worker_data)}"
        )

        return ToolResult.success_result(
            call=call,
            content=content,
            data={
                "action": action,
                "strategy": strategy,
                "execution_mode": execution_mode,
                "concurrent": concurrent,
                "max_concurrency": max_concurrency,
                "wait": wait,
                "worker_count": len(worker_data),
                "workers": worker_data,
                "task_manager_visible": execution_mode == COORDINATOR_EXECUTION_MODE_TASK,
            },
        )

    async def run_task_backed_workers(
        self,
        manager: SubAgentManager,
        requests: list[SubAgentTaskRequest],
        *,
        concurrent: bool,
        max_concurrency: int | None,
        parent_task_id: str | None,
        wait: bool,
    ) -> list[Any]:
        try:
            return await manager.run_many_agent_tasks(
                requests,
                concurrent=concurrent,
                max_concurrency=max_concurrency,
                parent_task_id=parent_task_id,
                wait=wait,
            )
        except TypeError:
            # 兼容旧签名。
            return await manager.run_many_agent_tasks(
                requests,
                concurrent=concurrent,
                parent_task_id=parent_task_id,
                wait=wait,
            )

    async def run_direct_workers(
        self,
        manager: SubAgentManager,
        requests: list[SubAgentTaskRequest],
        *,
        concurrent: bool,
        max_concurrency: int | None,
    ) -> list[Any]:
        return await manager.run_many(
            requests,
            concurrent=concurrent,
            max_concurrency=max_concurrency,
        )

    def render_result(
        self,
        result: ToolResult,
    ) -> str:
        if result.content:
            return result.content

        return json.dumps(
            result.data,
            ensure_ascii=False,
            indent=2,
            default=str,
        )