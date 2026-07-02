from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tasks.task import (
    TaskSpec,
    TaskStatus,
    TaskType,
    normalize_task_status,
    normalize_task_type,
    safe_jsonable,
)
from pywork.teams.team import (
    Team,
    TeamSharedTask,
    TeamTaskStatus,
    normalize_team_task_priority,
    normalize_team_task_status,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


TASK_CREATE_RISK = getattr(ToolRiskLevel, "MEDIUM", ToolRiskLevel.LOW)
TASK_STOP_RISK = getattr(ToolRiskLevel, "MEDIUM", ToolRiskLevel.LOW)


class TaskToolError(Exception):
    """task_tools 基础异常。"""


class TaskRuntimeMissingError(TaskToolError):
    """缺少 task_manager/team/subagent_manager 等运行时对象。"""


class TaskNotFoundError(TaskToolError):
    """找不到任务。"""


class TaskCreateTarget(str, Enum):
    AUTO = "auto"
    TASK_MANAGER = "task_manager"
    TEAM = "team"
    SUBAGENT = "subagent"


class TaskListTarget(str, Enum):
    AUTO = "auto"
    TASK_MANAGER = "task_manager"
    TEAM = "team"


class TaskOutputTarget(str, Enum):
    AUTO = "auto"
    TASK_MANAGER = "task_manager"
    TEAM = "team"


class TaskStopTarget(str, Enum):
    AUTO = "auto"
    TASK_MANAGER = "task_manager"
    TEAM = "team"
    SUBAGENT = "subagent"


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


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


def normalize_target(
    value: str | None,
    enum_cls: type[Enum],
) -> Enum:
    text = str(value or "auto").strip().lower()

    aliases = {
        "tm": "task_manager",
        "manager": "task_manager",
        "tasks": "task_manager",
        "shared_task": "team",
        "shared_tasks": "team",
        "team_task": "team",
        "agent": "subagent",
        "sub_agent": "subagent",
    }

    text = aliases.get(text, text)

    try:
        return enum_cls(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in enum_cls)
        raise ToolValidationError(
            f"Invalid target {value!r}. Valid targets: {valid}"
        ) from exc


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


def require_text_arg(
    args: Mapping[str, Any],
    key: str,
    *,
    aliases: Sequence[str] = (),
) -> str:
    text = optional_text_arg(
        args,
        key,
        aliases=aliases,
        default="",
    ).strip()

    if not text:
        raise ToolValidationError(f"{key} is required")

    return text


def require_task_id(args: Mapping[str, Any]) -> str:
    task_id = (
        args.get("task_id")
        or args.get("id")
        or args.get("task_record_id")
        or args.get("team_task_id")
    )

    text = str(task_id or "").strip()

    if not text:
        raise ToolValidationError("task_id is required")

    return text


def normalize_common_status(value: str | None) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    aliases = {
        "todo": "pending",
        "created": "pending",
        "queue": "queued",
        "enqueue": "queued",
        "start": "running",
        "started": "running",
        "in_progress": "running",
        "progress": "running",
        "done": "succeeded",
        "complete": "succeeded",
        "completed": "succeeded",
        "success": "succeeded",
        "ok": "succeeded",
        "error": "failed",
        "fail": "failed",
        "cancel": "cancelled",
        "canceled": "cancelled",
        "stop": "cancelled",
        "stopped": "cancelled",
        "abort": "aborted",
        "retry": "retrying",
    }

    return aliases.get(text, text)


def normalize_team_status_from_common(status: str | None) -> str | None:
    if status is None:
        return None

    mapping = {
        "pending": TeamTaskStatus.PENDING.value,
        "queued": TeamTaskStatus.PENDING.value,
        "retrying": TeamTaskStatus.RUNNING.value,
        "running": TeamTaskStatus.RUNNING.value,
        "succeeded": TeamTaskStatus.SUCCEEDED.value,
        "failed": TeamTaskStatus.FAILED.value,
        "cancelled": TeamTaskStatus.CANCELLED.value,
        "aborted": TeamTaskStatus.CANCELLED.value,
        "assigned": TeamTaskStatus.ASSIGNED.value,
        "dispatched": TeamTaskStatus.DISPATCHED.value,
    }

    return mapping.get(status, status)


def resolve_team(context: ToolExecutionContext) -> Team | None:
    metadata = context_metadata(context)

    team = metadata.get("team")

    if isinstance(team, Team):
        return team

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "team"):
        swarm_team = getattr(swarm, "team")

        if isinstance(swarm_team, Team):
            return swarm_team

    return None


def resolve_subagent_manager(context: ToolExecutionContext) -> Any | None:
    metadata = context_metadata(context)

    manager = (
        metadata.get("subagent_manager")
        or metadata.get("manager")
    )

    if manager is not None:
        return manager

    team = resolve_team(context)

    if team is not None and object_has_attr(team, "roster"):
        roster = getattr(team, "roster")

        if roster is not None and object_has_attr(roster, "manager"):
            return getattr(roster, "manager")

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "manager"):
        return getattr(swarm, "manager")

    return None


def resolve_task_manager(context: ToolExecutionContext) -> Any | None:
    metadata = context_metadata(context)

    task_manager = metadata.get("task_manager")

    if task_manager is not None:
        return task_manager

    manager = resolve_subagent_manager(context)

    if manager is not None and object_has_attr(manager, "task_manager"):
        return getattr(manager, "task_manager")

    return None


def has_any_task_runtime(context: ToolExecutionContext) -> bool:
    return (
        resolve_task_manager(context) is not None
        or resolve_team(context) is not None
        or resolve_subagent_manager(context) is not None
    )


def task_record_to_data(task: Any) -> dict[str, Any]:
    to_dict = getattr(task, "to_dict", None)

    if callable(to_dict):
        return safe_jsonable(to_dict())

    return {
        "id": getattr(task, "id", None),
        "name": getattr(task, "name", None),
        "task_type": safe_jsonable(getattr(task, "task_type", None)),
        "status": safe_jsonable(getattr(task, "status", None)),
        "payload": safe_jsonable(getattr(task, "payload", None)),
        "result": safe_jsonable(getattr(task, "result", None)),
        "error": getattr(task, "error", None),
        "parent_id": getattr(task, "parent_id", None),
        "agent_id": getattr(task, "agent_id", None),
        "metadata": safe_jsonable(getattr(task, "metadata", None)),
    }


def team_task_to_data(task: TeamSharedTask) -> dict[str, Any]:
    return safe_jsonable(task.to_dict())


def execution_to_data(execution: Any) -> dict[str, Any]:
    if isinstance(execution, Mapping):
        return safe_jsonable(dict(execution))

    to_dict = getattr(execution, "to_dict", None)

    if callable(to_dict):
        return safe_jsonable(to_dict())

    record = getattr(execution, "record", None)

    return {
        "task_id": getattr(execution, "task_id", None)
        or getattr(execution, "id", None)
        or getattr(record, "id", None),
        "done": getattr(execution, "done", None),
        "cancelled": getattr(execution, "cancelled", None),
        "record": task_record_to_data(record) if record is not None else None,
    }


async def get_task_record(
    task_manager: Any,
    task_id: str,
) -> Any | None:
    get_task = getattr(task_manager, "get_task", None)

    if callable(get_task):
        task = await maybe_await(get_task(task_id))

        if task is not None:
            return task

    records = getattr(task_manager, "_records", None)

    if isinstance(records, Mapping):
        return records.get(task_id)

    return None


async def persist_task_record(
    task_manager: Any,
    task: Any,
) -> None:
    task_id = getattr(task, "id", None)

    records = getattr(task_manager, "_records", None)

    if isinstance(records, dict) and task_id:
        records[str(task_id)] = task

    update_task = getattr(task_manager, "update_task", None)

    if callable(update_task):
        await maybe_await(update_task(task))
        return

    register_task = getattr(task_manager, "register_task", None)

    if callable(register_task):
        await maybe_await(register_task(task))
        return

    persist_record = getattr(task_manager, "_persist_record", None)

    if callable(persist_record):
        await maybe_await(persist_record(task))
        return


def set_task_id_if_possible(
    task: Any,
    task_id: str | None,
) -> Any:
    if not task_id:
        return task

    try:
        setattr(task, "id", task_id)
    except Exception:
        pass

    return task


async def register_task_record(
    task_manager: Any,
    task: Any,
) -> Any:
    register_task = getattr(task_manager, "register_task", None)

    if callable(register_task):
        await maybe_await(register_task(task))
        return task

    records = getattr(task_manager, "_records", None)

    if isinstance(records, dict):
        task_id = getattr(task, "id", None)

        if task_id:
            records[str(task_id)] = task
            return task

    update_task = getattr(task_manager, "update_task", None)

    if callable(update_task):
        await maybe_await(update_task(task))
        return task

    raise TaskRuntimeMissingError(
        "task_manager does not support register_task/update_task/_records"
    )


async def create_task_manager_record(
    task_manager: Any,
    args: Mapping[str, Any],
    context: ToolExecutionContext,
) -> Any:
    name = optional_text_arg(
        args,
        "name",
        aliases=("title",),
        default="",
    ).strip()

    task_text = optional_text_arg(
        args,
        "task",
        aliases=("description", "content"),
        default="",
    ).strip()

    if not name:
        name = task_text.splitlines()[0][:80] if task_text else "Task"

    payload = optional_dict_arg(args, "payload")

    if task_text and "task" not in payload:
        payload["task"] = task_text

    metadata = optional_dict_arg(args, "metadata")

    task_type_value = args.get("task_type") or TaskType.GENERIC.value

    try:
        task_type = normalize_task_type(task_type_value)
    except Exception:
        task_type = TaskType.GENERIC

    spec = TaskSpec(
        name=name,
        task_type=task_type,
        payload=payload,
        parent_id=args.get("parent_id") or args.get("parent_task_id"),
        agent_id=args.get("agent_id") or args.get("agent_name"),
        metadata=metadata,
        max_retries=optional_int_arg(args, "max_retries", default=0) or 0,
        timeout_seconds=optional_float_arg(args, "timeout_seconds"),
        created_by=args.get("created_by") or args.get("sender_id"),
    )

    create_task = getattr(task_manager, "create_task", None)
    task_id = args.get("task_id") or args.get("id")

    if callable(create_task):
        attempts = [
            lambda: create_task(
                spec=spec,
                task_id=str(task_id) if task_id else None,
            ),
            lambda: create_task(
                name=spec.name,
                task_type=spec.task_type,
                payload=spec.payload,
                parent_id=spec.parent_id,
                agent_id=spec.agent_id,
                metadata=spec.metadata,
                max_retries=spec.max_retries,
                timeout_seconds=spec.timeout_seconds,
                created_by=spec.created_by,
                task_id=str(task_id) if task_id else None,
            ),
        ]

        for attempt in attempts:
            try:
                task = await maybe_await(attempt())
                task = set_task_id_if_possible(task, str(task_id) if task_id else None)
                await persist_task_record(task_manager, task)
                return task
            except TypeError:
                continue

    task = spec.to_record()
    task = set_task_id_if_possible(task, str(task_id) if task_id else None)

    await register_task_record(
        task_manager,
        task,
    )

    return task


async def maybe_start_task_manager_record(
    task_manager: Any,
    task: Any,
    args: Mapping[str, Any],
) -> Any | None:
    start = optional_bool_arg(args, "start", default=False)

    if not start:
        return None

    runner_name = optional_text_arg(args, "runner_name", default="").strip()
    runner = None

    if runner_name:
        get_runner = getattr(task_manager, "get_runner", None)

        if callable(get_runner):
            runner = await maybe_await(get_runner(runner_name))

    if runner is None:
        task_runners = args.get("task_runners")

        if isinstance(task_runners, Mapping) and runner_name:
            runner = task_runners.get(runner_name)

    task_id = getattr(task, "id", None)

    start_task = getattr(task_manager, "start_task", None)

    if callable(start_task):
        attempts = [
            lambda: start_task(
                task_id,
                runner=runner,
            ),
            lambda: start_task(
                task_id,
            ),
        ]

        for attempt in attempts:
            try:
                return await maybe_await(attempt())
            except TypeError:
                continue

    run_task = getattr(task_manager, "run_task", None)

    if callable(run_task):
        attempts = [
            lambda: run_task(
                record=task,
                runner=runner,
            ),
            lambda: run_task(
                task_id,
                runner=runner,
            ),
        ]

        for attempt in attempts:
            try:
                return await maybe_await(attempt())
            except TypeError:
                continue

    return None


async def start_subagent_task(
    subagent_manager: Any,
    args: Mapping[str, Any],
    context: ToolExecutionContext,
) -> Any:
    agent_name = require_text_arg(
        args,
        "agent_name",
        aliases=("agent", "role"),
    )
    task_text = require_text_arg(
        args,
        "task",
        aliases=("description", "content"),
    )

    metadata = optional_dict_arg(args, "metadata")
    workspace_path = args.get("workspace_path") or getattr(context, "workspace_path", ".")

    try:
        from pywork.subagents.manager import SubAgentTaskRequest

        request = SubAgentTaskRequest(
            agent_name=agent_name,
            task=task_text,
            workspace_path=workspace_path,
            metadata=metadata,
            max_steps=optional_int_arg(args, "max_steps"),
        )
    except Exception:
        request = None

    run_agent_task = getattr(subagent_manager, "run_agent_task", None)

    if not callable(run_agent_task):
        raise TaskRuntimeMissingError("subagent_manager does not support run_agent_task")

    wait = optional_bool_arg(args, "wait", default=False)

    attempts = []

    if request is not None:
        attempts.extend(
            [
                lambda: run_agent_task(request, wait=wait),
                lambda: run_agent_task(request=request, wait=wait),
            ]
        )

    attempts.extend(
        [
            lambda: run_agent_task(
                agent_name=agent_name,
                task=task_text,
                workspace_path=workspace_path,
                metadata=metadata,
                wait=wait,
            ),
            lambda: run_agent_task(
                agent_name,
                task_text,
                wait=wait,
            ),
        ]
    )

    last_type_error: TypeError | None = None

    for attempt in attempts:
        try:
            return await maybe_await(attempt())
        except TypeError as exc:
            last_type_error = exc
            continue

    if last_type_error is not None:
        raise last_type_error

    raise TaskRuntimeMissingError("failed to start subagent task")


async def list_task_manager_records(
    task_manager: Any,
    args: Mapping[str, Any],
) -> list[Any]:
    status_text = normalize_common_status(
        str(args["status"])
        if args.get("status") is not None
        else None
    )

    normalized_status: Any = None

    if status_text is not None:
        try:
            normalized_status = normalize_task_status(status_text)
        except Exception:
            normalized_status = status_text

    parent_id = args.get("parent_id") or args.get("parent_task_id")
    agent_id = args.get("agent_id") or args.get("agent_name")
    limit = optional_int_arg(args, "limit")

    list_tasks = getattr(task_manager, "list_tasks", None)

    if callable(list_tasks):
        attempts = [
            lambda: list_tasks(
                status=normalized_status,
                parent_id=parent_id,
                agent_id=agent_id,
                limit=limit,
            ),
            lambda: list_tasks(
                status=normalized_status,
                limit=limit,
            ),
            lambda: list_tasks(),
        ]

        for attempt in attempts:
            try:
                records = await maybe_await(attempt())

                if records is None:
                    return []

                records = list(records)
                break
            except TypeError:
                continue
        else:
            records = []
    else:
        records_obj = getattr(task_manager, "_records", {})

        if isinstance(records_obj, Mapping):
            records = list(records_obj.values())
        else:
            records = []

    if normalized_status is not None:
        records = [
            record
            for record in records
            if safe_jsonable(getattr(record, "status", None)) == safe_jsonable(normalized_status)
        ]

    if parent_id is not None:
        records = [
            record
            for record in records
            if getattr(record, "parent_id", None) == parent_id
        ]

    if agent_id is not None:
        records = [
            record
            for record in records
            if getattr(record, "agent_id", None) == agent_id
        ]

    if limit is not None:
        records = records[:limit]

    return records


def list_team_tasks(
    team: Team,
    args: Mapping[str, Any],
) -> list[TeamSharedTask]:
    status_text = normalize_common_status(
        str(args["status"])
        if args.get("status") is not None
        else None
    )
    team_status = normalize_team_status_from_common(status_text)

    role = args.get("role")
    assigned_to = args.get("assigned_to") or args.get("teammate_id")
    limit = optional_int_arg(args, "limit")
    include_terminal = optional_bool_arg(args, "include_terminal", default=True)

    normalized_status = None

    if team_status is not None:
        try:
            normalized_status = normalize_team_task_status(team_status)
        except Exception:
            normalized_status = team_status

    return team.list_shared_tasks(
        status=normalized_status,
        assigned_to=str(assigned_to) if assigned_to is not None else None,
        role=str(role) if role is not None else None,
        include_terminal=include_terminal,
        limit=limit,
    )


async def cancel_task_manager_task(
    task_manager: Any,
    task_id: str,
    reason: str,
) -> tuple[bool, Any | None]:
    cancel_task = getattr(task_manager, "cancel_task", None)

    if callable(cancel_task):
        attempts = [
            lambda: cancel_task(task_id, reason=reason),
            lambda: cancel_task(task_id),
        ]

        for attempt in attempts:
            try:
                output = await maybe_await(attempt())
                return True, output
            except TypeError:
                continue

    stop_task = getattr(task_manager, "stop_task", None)

    if callable(stop_task):
        attempts = [
            lambda: stop_task(task_id, reason=reason),
            lambda: stop_task(task_id),
        ]

        for attempt in attempts:
            try:
                output = await maybe_await(attempt())
                return True, output
            except TypeError:
                continue

    task = await get_task_record(
        task_manager,
        task_id,
    )

    if task is None:
        return False, None

    mark_cancelled = getattr(task, "mark_cancelled", None)

    if callable(mark_cancelled):
        try:
            mark_cancelled(reason)
        except TypeError:
            mark_cancelled()
    else:
        try:
            setattr(task, "status", normalize_task_status(TaskStatus.CANCELLED.value))
        except Exception:
            setattr(task, "status", TaskStatus.CANCELLED.value)

        setattr(task, "error", reason)

    await persist_task_record(
        task_manager,
        task,
    )

    return True, task


async def cancel_subagent_task(
    subagent_manager: Any,
    task_id: str,
    reason: str,
) -> tuple[bool, Any | None]:
    cancel_agent_task = getattr(subagent_manager, "cancel_agent_task", None)

    if callable(cancel_agent_task):
        attempts = [
            lambda: cancel_agent_task(task_id, reason=reason),
            lambda: cancel_agent_task(task_id),
        ]

        for attempt in attempts:
            try:
                output = await maybe_await(attempt())
                return True, output
            except TypeError:
                continue

    abort_run = getattr(subagent_manager, "abort_run", None)

    if callable(abort_run):
        attempts = [
            lambda: abort_run(task_id, reason=reason),
            lambda: abort_run(task_id),
        ]

        for attempt in attempts:
            try:
                output = await maybe_await(attempt())
                return True, output
            except TypeError:
                continue

    return False, None


class TaskCreateTool(BaseTool):
    name = "task_create"
    description = "Create a TaskRecord, TeamSharedTask, or subagent-backed background task."
    risk = TASK_CREATE_RISK

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "auto | task_manager | team | subagent",
                "default": "auto",
            },
            "name": {
                "type": "string",
            },
            "title": {
                "type": "string",
            },
            "task": {
                "type": "string",
            },
            "description": {
                "type": "string",
            },
            "content": {
                "type": "string",
            },
            "task_id": {
                "type": "string",
            },
            "task_type": {
                "type": "string",
                "description": "generic | subagent | tool | runtime | user",
                "default": "generic",
            },
            "agent_name": {
                "type": "string",
                "description": "SubAgent name when target=subagent.",
            },
            "role": {
                "type": "string",
                "description": "Team role or subagent role.",
            },
            "assigned_to": {
                "type": "string",
            },
            "priority": {
                "type": "string",
                "description": "low | normal | high | urgent",
                "default": "normal",
            },
            "payload": {
                "type": "object",
            },
            "metadata": {
                "type": "object",
            },
            "parent_id": {
                "type": "string",
            },
            "agent_id": {
                "type": "string",
            },
            "max_retries": {
                "type": "integer",
            },
            "timeout_seconds": {
                "type": "number",
            },
            "start": {
                "type": "boolean",
                "default": False,
            },
            "wait": {
                "type": "boolean",
                "default": False,
            },
            "runner_name": {
                "type": "string",
            },
        },
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)

        try:
            if not has_any_task_runtime(context):
                raise TaskRuntimeMissingError(
                    "task_create requires context.metadata['task_manager'], "
                    "context.metadata['team'], or context.metadata['subagent_manager']."
                )

            target = normalize_target(
                args.get("target"),
                TaskCreateTarget,
            )

            if target == TaskCreateTarget.AUTO:
                if args.get("agent_name") and resolve_subagent_manager(context) is not None:
                    target = TaskCreateTarget.SUBAGENT
                elif resolve_task_manager(context) is not None:
                    target = TaskCreateTarget.TASK_MANAGER
                elif resolve_team(context) is not None:
                    target = TaskCreateTarget.TEAM
                else:
                    target = TaskCreateTarget.SUBAGENT

            if target == TaskCreateTarget.SUBAGENT:
                manager = resolve_subagent_manager(context)

                if manager is None:
                    raise TaskRuntimeMissingError("subagent_manager is required")

                output = await start_subagent_task(
                    manager,
                    args,
                    context,
                )

                return make_result(
                    call,
                    tool_name=self.name,
                    success=True,
                    content="Subagent task created.",
                    data={
                        "target": "subagent",
                        "task": execution_to_data(output),
                    },
                )

            if target == TaskCreateTarget.TASK_MANAGER:
                task_manager = resolve_task_manager(context)

                if task_manager is None:
                    raise TaskRuntimeMissingError("task_manager is required")

                task = await create_task_manager_record(
                    task_manager,
                    args,
                    context,
                )
                execution = await maybe_start_task_manager_record(
                    task_manager,
                    task,
                    args,
                )

                return make_result(
                    call,
                    tool_name=self.name,
                    success=True,
                    content=f"Task created: {getattr(task, 'id', None)}",
                    data={
                        "target": "task_manager",
                        "task": task_record_to_data(task),
                        "execution": (
                            execution_to_data(execution)
                            if execution is not None
                            else None
                        ),
                    },
                )

            if target == TaskCreateTarget.TEAM:
                team = resolve_team(context)

                if team is None:
                    raise TaskRuntimeMissingError("team is required")

                title = optional_text_arg(
                    args,
                    "title",
                    aliases=("name",),
                    default="",
                ).strip()
                description = optional_text_arg(
                    args,
                    "description",
                    aliases=("task", "content"),
                    default="",
                ).strip()

                if not title:
                    title = description.splitlines()[0][:80] if description else "Team task"

                task = team.create_shared_task(
                    title,
                    description=description,
                    role=(
                        str(args["role"])
                        if args.get("role") is not None
                        else None
                    ),
                    assigned_to=(
                        str(args["assigned_to"])
                        if args.get("assigned_to") is not None
                        else None
                    ),
                    parent_task_id=(
                        str(args["parent_id"])
                        if args.get("parent_id") is not None
                        else None
                    ),
                    created_by=(
                        str(args["created_by"])
                        if args.get("created_by") is not None
                        else None
                    ),
                    priority=normalize_team_task_priority(args.get("priority")),
                    payload=optional_dict_arg(args, "payload"),
                    metadata=optional_dict_arg(args, "metadata"),
                    task_id=(
                        str(args["task_id"])
                        if args.get("task_id") is not None
                        else None
                    ),
                )

                return make_result(
                    call,
                    tool_name=self.name,
                    success=True,
                    content=f"Team task created: {task.task_id}",
                    data={
                        "target": "team",
                        "task": team_task_to_data(task),
                    },
                )

            raise ToolValidationError(f"Unsupported target: {target.value}")

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"task_create failed: {exc}",
                data={
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


class TaskListTool(BaseTool):
    name = "task_list"
    description = "List TaskManager records and/or Team shared tasks."
    risk = ToolRiskLevel.LOW

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "auto | task_manager | team",
                "default": "auto",
            },
            "status": {
                "type": "string",
            },
            "parent_id": {
                "type": "string",
            },
            "agent_id": {
                "type": "string",
            },
            "agent_name": {
                "type": "string",
            },
            "role": {
                "type": "string",
            },
            "assigned_to": {
                "type": "string",
            },
            "include_terminal": {
                "type": "boolean",
                "default": True,
            },
            "limit": {
                "type": "integer",
            },
        },
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)

        try:
            if not has_any_task_runtime(context):
                raise TaskRuntimeMissingError(
                    "task_list requires context.metadata['task_manager'] or context.metadata['team']."
                )

            target = normalize_target(
                args.get("target"),
                TaskListTarget,
            )

            sources: list[str] = []
            task_records: list[dict[str, Any]] = []
            team_tasks: list[dict[str, Any]] = []

            task_manager = resolve_task_manager(context)
            team = resolve_team(context)

            if target in {TaskListTarget.AUTO, TaskListTarget.TASK_MANAGER} and task_manager is not None:
                records = await list_task_manager_records(
                    task_manager,
                    args,
                )
                task_records = [
                    task_record_to_data(record)
                    for record in records
                ]
                sources.append("task_manager")

            if target in {TaskListTarget.AUTO, TaskListTarget.TEAM} and team is not None:
                tasks = list_team_tasks(
                    team,
                    args,
                )
                team_tasks = [
                    team_task_to_data(task)
                    for task in tasks
                ]
                sources.append("team")

            if not sources:
                raise TaskRuntimeMissingError(f"No runtime source available for target={target.value}")

            count = len(task_records) + len(team_tasks)

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Listed {count} task(s).",
                data={
                    "sources": sources,
                    "count": count,
                    "task_records": task_records,
                    "team_tasks": team_tasks,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"task_list failed: {exc}",
                data={
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


class TaskOutputTool(BaseTool):
    name = "task_output"
    description = "Get output, status, result, and metadata for one task."
    risk = ToolRiskLevel.LOW

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "auto | task_manager | team",
                "default": "auto",
            },
            "task_id": {
                "type": "string",
            },
        },
        "required": [
            "task_id",
        ],
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)
        task_id = ""

        try:
            if not has_any_task_runtime(context):
                raise TaskRuntimeMissingError(
                    "task_output requires context.metadata['task_manager'] or context.metadata['team']."
                )

            task_id = require_task_id(args)
            target = normalize_target(
                args.get("target"),
                TaskOutputTarget,
            )

            task_record_data = None
            team_task_data = None
            sources: list[str] = []

            task_manager = resolve_task_manager(context)

            if target in {TaskOutputTarget.AUTO, TaskOutputTarget.TASK_MANAGER} and task_manager is not None:
                record = await get_task_record(
                    task_manager,
                    task_id,
                )

                if record is not None:
                    task_record_data = task_record_to_data(record)
                    sources.append("task_manager")

            team = resolve_team(context)

            if target in {TaskOutputTarget.AUTO, TaskOutputTarget.TEAM} and team is not None:
                team_task = team.get_shared_task(task_id)

                if team_task is not None:
                    team_task_data = team_task_to_data(team_task)
                    sources.append("team")

            if not sources:
                raise TaskNotFoundError(f"task not found: {task_id}")

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Task output: {task_id}",
                data={
                    "task_id": task_id,
                    "sources": sources,
                    "task_record": task_record_data,
                    "team_task": team_task_data,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"task_output failed: {exc}",
                data={
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


class TaskStopTool(BaseTool):
    name = "task_stop"
    description = "Stop or cancel a running TaskManager task, Team shared task, or SubAgent task."
    risk = TASK_STOP_RISK

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "auto | task_manager | team | subagent",
                "default": "auto",
            },
            "task_id": {
                "type": "string",
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "task_id",
        ],
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)
        task_id = ""

        try:
            if not has_any_task_runtime(context):
                raise TaskRuntimeMissingError(
                    "task_stop requires context.metadata['task_manager'], "
                    "context.metadata['team'], or context.metadata['subagent_manager']."
                )

            task_id = require_task_id(args)
            reason = optional_text_arg(
                args,
                "reason",
                aliases=("message",),
                default="task stopped",
            )
            target = normalize_target(
                args.get("target"),
                TaskStopTarget,
            )

            cancelled_targets: list[str] = []
            outputs: dict[str, Any] = {}

            if target in {TaskStopTarget.AUTO, TaskStopTarget.SUBAGENT}:
                manager = resolve_subagent_manager(context)

                if manager is not None:
                    cancelled, output = await cancel_subagent_task(
                        manager,
                        task_id,
                        reason,
                    )

                    if cancelled:
                        cancelled_targets.append("subagent")
                        outputs["subagent"] = safe_jsonable(output)

            if target in {TaskStopTarget.AUTO, TaskStopTarget.TASK_MANAGER}:
                task_manager = resolve_task_manager(context)

                if task_manager is not None:
                    cancelled, output = await cancel_task_manager_task(
                        task_manager,
                        task_id,
                        reason,
                    )

                    if cancelled:
                        cancelled_targets.append("task_manager")
                        outputs["task_manager"] = safe_jsonable(
                            task_record_to_data(output)
                            if output is not None and not isinstance(output, bool)
                            else output
                        )

            if target in {TaskStopTarget.AUTO, TaskStopTarget.TEAM}:
                team = resolve_team(context)

                if team is not None:
                    team_task = team.get_shared_task(task_id)

                    if team_task is not None:
                        team_task.mark_cancelled(reason)
                        cancelled_targets.append("team")
                        outputs["team"] = team_task_to_data(team_task)

            if not cancelled_targets:
                raise TaskNotFoundError(f"task not found or not cancellable: {task_id}")

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Task stopped: {task_id}",
                data={
                    "task_id": task_id,
                    "reason": reason,
                    "cancelled_targets": cancelled_targets,
                    "outputs": outputs,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"task_stop failed: {exc}",
                data={
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


__all__ = [
    "TaskCreateTarget",
    "TaskCreateTool",
    "TaskListTarget",
    "TaskListTool",
    "TaskNotFoundError",
    "TaskOutputTarget",
    "TaskOutputTool",
    "TaskRuntimeMissingError",
    "TaskStopTarget",
    "TaskStopTool",
    "TaskToolError",
    "cancel_task_manager_task",
    "create_task_manager_record",
    "get_task_record",
    "has_any_task_runtime",
    "list_task_manager_records",
    "resolve_subagent_manager",
    "resolve_task_manager",
    "resolve_team",
    "start_subagent_task",
]