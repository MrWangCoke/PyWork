from __future__ import annotations

import inspect
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tasks.task import (
    TaskResult,
    TaskStatus,
    normalize_task_status,
)
from pywork.teams.team import (
    Team,
    TeamSharedTask,
    TeamTaskStatus,
    normalize_team_task_status,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


class TaskUpdateAction(str, Enum):
    UPDATE = "update"
    STATUS = "status"
    METADATA = "metadata"
    RESULT = "result"
    FAIL = "fail"
    CANCEL = "cancel"


class TaskUpdateToolError(Exception):
    """task_update tool 基础异常。"""


class TaskUpdateRuntimeMissingError(TaskUpdateToolError):
    """缺少 task_manager/team 等运行时对象。"""


class TaskUpdateNotFoundError(TaskUpdateToolError):
    """找不到对应 task。"""


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


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def safe_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
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

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            return safe_jsonable(to_dict())
        except Exception:
            pass

    return str(value)


def context_metadata(context: ToolExecutionContext) -> dict[str, Any]:
    metadata = getattr(context, "metadata", None)

    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


def object_has_attr(value: Any, attr: str) -> bool:
    return hasattr(value, attr) and getattr(value, attr) is not None


def normalize_action(value: str | None) -> TaskUpdateAction:
    text = str(value or TaskUpdateAction.UPDATE.value).strip().lower()

    aliases = {
        "set_status": "status",
        "mark": "status",
        "mark_status": "status",
        "update_status": "status",
        "update_metadata": "metadata",
        "set_metadata": "metadata",
        "complete": "result",
        "succeed": "result",
        "success": "result",
        "succeeded": "result",
        "failed": "fail",
        "error": "fail",
        "cancelled": "cancel",
        "canceled": "cancel",
        "stop": "cancel",
        "stopped": "cancel",
    }

    text = aliases.get(text, text)

    try:
        return TaskUpdateAction(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in TaskUpdateAction)
        raise ToolValidationError(
            f"Invalid task_update action {value!r}. Valid actions: {valid}"
        ) from exc


def normalize_common_status(
    status: str | None,
    *,
    action: TaskUpdateAction,
) -> str | None:
    if status is None:
        if action == TaskUpdateAction.RESULT:
            return "succeeded"

        if action == TaskUpdateAction.FAIL:
            return "failed"

        if action == TaskUpdateAction.CANCEL:
            return "cancelled"

        return None

    text = str(status).strip().lower()

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


def resolve_task_manager(context: ToolExecutionContext) -> Any | None:
    metadata = context_metadata(context)

    task_manager = metadata.get("task_manager")

    if task_manager is not None:
        return task_manager

    manager = metadata.get("manager") or metadata.get("subagent_manager")

    if manager is not None and object_has_attr(manager, "task_manager"):
        return getattr(manager, "task_manager")

    team = resolve_team(context)

    if team is not None and object_has_attr(team, "roster"):
        roster = getattr(team, "roster")

        if roster is not None and object_has_attr(roster, "manager"):
            roster_manager = getattr(roster, "manager")

            if roster_manager is not None and object_has_attr(roster_manager, "task_manager"):
                return getattr(roster_manager, "task_manager")

    swarm = metadata.get("swarm")

    if swarm is not None and object_has_attr(swarm, "team"):
        swarm_team = getattr(swarm, "team")

        if swarm_team is not None and object_has_attr(swarm_team, "roster"):
            roster = getattr(swarm_team, "roster")

            if roster is not None and object_has_attr(roster, "manager"):
                roster_manager = getattr(roster, "manager")

                if roster_manager is not None and object_has_attr(roster_manager, "task_manager"):
                    return getattr(roster_manager, "task_manager")

    return None


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


def has_task_runtime(context: ToolExecutionContext) -> bool:
    return resolve_task_manager(context) is not None or resolve_team(context) is not None


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

    storage = getattr(task_manager, "storage", None)

    if storage is not None:
        storage_update = getattr(storage, "update_task", None)

        if callable(storage_update):
            await maybe_await(storage_update(task))


def build_success_task_result(
    value: Any,
    *,
    metadata: dict[str, Any] | None = None,
) -> TaskResult:
    try:
        return TaskResult.success_result(
            value=value,
            metadata=metadata or {},
        )
    except TypeError:
        try:
            return TaskResult.success_result(value)
        except TypeError:
            return TaskResult(
                success=True,
                value=value,
                metadata=metadata or {},
            )


def build_failure_task_result(
    error: str,
    *,
    value: Any = None,
    error_type: str | None = None,
    traceback: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskResult:
    try:
        return TaskResult.failure_result(
            error=error,
            error_type=error_type,
            traceback=traceback,
            metadata=metadata or {},
        )
    except TypeError:
        try:
            return TaskResult.failure_result(error)
        except TypeError:
            return TaskResult(
                success=False,
                value=value,
                error=error,
                error_type=error_type,
                traceback=traceback,
                metadata=metadata or {},
            )


def extract_result_payload(args: Mapping[str, Any]) -> Any:
    if "result" in args:
        return args["result"]

    if "output" in args:
        return args["output"]

    if "value" in args:
        return args["value"]

    if "content" in args:
        return {
            "content": args["content"],
        }

    return {}


def extract_error_text(args: Mapping[str, Any]) -> str:
    return str(
        args.get("error")
        or args.get("reason")
        or args.get("message")
        or "task failed"
    )


def merge_metadata(
    target: Any,
    metadata: dict[str, Any],
    *,
    replace: bool = False,
) -> None:
    if not metadata:
        return

    current = getattr(target, "metadata", None)

    if replace or not isinstance(current, dict):
        setattr(target, "metadata", dict(metadata))
    else:
        current.update(metadata)


def touch_task(task: Any) -> None:
    touch = getattr(task, "touch", None)

    if callable(touch):
        touch()


def call_method_if_exists(
    target: Any,
    name: str,
    *args: Any,
    **kwargs: Any,
) -> bool:
    method = getattr(target, name, None)

    if not callable(method):
        return False

    attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
        (args, kwargs),
        (args, {}),
        ((), kwargs),
        ((), {}),
    ]

    for call_args, call_kwargs in attempts:
        try:
            method(*call_args, **call_kwargs)
            return True
        except TypeError:
            continue

    return False


def set_record_status_directly(
    task: Any,
    status: str,
) -> None:
    try:
        normalized_status = normalize_task_status(status)
    except Exception:
        normalized_status = status

    set_status = getattr(task, "set_status", None)

    if callable(set_status):
        try:
            set_status(normalized_status)
            return
        except TypeError:
            set_status(str(status))
            return

    setattr(task, "status", normalized_status)
    touch_task(task)


def set_team_task_status_directly(
    task: TeamSharedTask,
    status: str,
) -> None:
    try:
        normalized_status = normalize_team_task_status(status)
    except Exception:
        normalized_status = status

    task.status = normalized_status
    task.touch()


def update_task_record_status(
    task: Any,
    status: str | None,
    args: Mapping[str, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    if status is None:
        merge_metadata(
            task,
            metadata,
            replace=optional_bool_arg(args, "replace_metadata", default=False),
        )
        touch_task(task)
        return

    result_payload = extract_result_payload(args)
    error_text = extract_error_text(args)

    if status == TaskStatus.QUEUED.value:
        if not call_method_if_exists(task, "mark_queued"):
            set_record_status_directly(task, status)

    elif status == TaskStatus.RUNNING.value:
        if not call_method_if_exists(task, "mark_running"):
            set_record_status_directly(task, status)

    elif status == TaskStatus.RETRYING.value:
        if not call_method_if_exists(task, "mark_retrying"):
            set_record_status_directly(task, status)

    elif status == TaskStatus.SUCCEEDED.value:
        task_result = build_success_task_result(
            result_payload,
            metadata=metadata,
        )

        if not call_method_if_exists(task, "mark_succeeded", task_result):
            set_record_status_directly(task, status)
            setattr(task, "result", task_result)
            setattr(task, "error", None)

    elif status == TaskStatus.FAILED.value:
        task_result = build_failure_task_result(
            error_text,
            value=result_payload,
            error_type=(
                str(args["error_type"])
                if args.get("error_type") is not None
                else None
            ),
            traceback=(
                str(args["traceback"])
                if args.get("traceback") is not None
                else None
            ),
            metadata=metadata,
        )

        if not call_method_if_exists(task, "mark_failed", error_text, result=task_result):
            set_record_status_directly(task, status)
            setattr(task, "result", task_result)
            setattr(task, "error", error_text)

    elif status == TaskStatus.CANCELLED.value:
        reason = str(args.get("reason") or args.get("message") or "task cancelled")

        if not call_method_if_exists(task, "mark_cancelled", reason):
            set_record_status_directly(task, status)
            setattr(task, "error", reason)

    elif status == TaskStatus.ABORTED.value:
        reason = str(args.get("reason") or args.get("message") or "task aborted")

        if not call_method_if_exists(task, "mark_aborted", reason):
            set_record_status_directly(task, status)
            setattr(task, "error", reason)

    elif status == TaskStatus.PENDING.value:
        set_record_status_directly(task, status)

    else:
        set_record_status_directly(task, status)

    merge_metadata(
        task,
        metadata,
        replace=optional_bool_arg(args, "replace_metadata", default=False),
    )
    touch_task(task)


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
    }

    return mapping.get(status, status)


def update_team_task_status(
    task: TeamSharedTask,
    status: str | None,
    args: Mapping[str, Any],
    *,
    metadata: dict[str, Any],
) -> None:
    if status is None:
        merge_metadata(
            task,
            metadata,
            replace=optional_bool_arg(args, "replace_metadata", default=False),
        )
        task.touch()
        return

    team_status = normalize_team_status_from_common(status)
    result_payload = extract_result_payload(args)
    error_text = extract_error_text(args)

    if team_status == TeamTaskStatus.RUNNING.value:
        task.mark_running()

    elif team_status == TeamTaskStatus.SUCCEEDED.value:
        result = (
            dict(result_payload)
            if isinstance(result_payload, Mapping)
            else {
                "value": result_payload,
            }
        )

        task.mark_succeeded(result)

    elif team_status == TeamTaskStatus.FAILED.value:
        result = (
            dict(result_payload)
            if isinstance(result_payload, Mapping)
            else {
                "value": result_payload,
            }
        )

        task.mark_failed(
            error_text,
            result=result,
        )

    elif team_status == TeamTaskStatus.CANCELLED.value:
        task.mark_cancelled(
            str(args.get("reason") or args.get("message") or "team task cancelled")
        )

    elif team_status == TeamTaskStatus.PENDING.value:
        set_team_task_status_directly(task, team_status)

    elif team_status == TeamTaskStatus.ASSIGNED.value:
        set_team_task_status_directly(task, team_status)

    elif team_status == TeamTaskStatus.DISPATCHED.value:
        set_team_task_status_directly(task, team_status)

    else:
        set_team_task_status_directly(task, team_status)

    merge_metadata(
        task,
        metadata,
        replace=optional_bool_arg(args, "replace_metadata", default=False),
    )
    task.touch()


def task_record_to_data(task: Any) -> dict[str, Any]:
    to_dict = getattr(task, "to_dict", None)

    if callable(to_dict):
        return safe_jsonable(to_dict())

    return {
        "id": getattr(task, "id", None),
        "name": getattr(task, "name", None),
        "status": safe_jsonable(getattr(task, "status", None)),
        "result": safe_jsonable(getattr(task, "result", None)),
        "error": getattr(task, "error", None),
        "metadata": safe_jsonable(getattr(task, "metadata", None)),
    }


class TaskUpdateTool(BaseTool):
    name = "task_update"
    description = "Update TaskRecord or TeamSharedTask status, result, error, and metadata."
    risk = getattr(ToolRiskLevel, "MEDIUM", ToolRiskLevel.LOW)

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "update | status | metadata | result | fail | cancel",
                "default": "update",
            },
            "task_id": {
                "type": "string",
                "description": "TaskRecord id or TeamSharedTask id.",
            },
            "status": {
                "type": "string",
                "description": "pending | queued | running | retrying | succeeded | failed | cancelled | aborted",
            },
            "result": {
                "description": "Structured success result.",
            },
            "output": {
                "description": "Alias for result.",
            },
            "value": {
                "description": "Alias for result.",
            },
            "content": {
                "type": "string",
                "description": "Optional textual result content.",
            },
            "error": {
                "type": "string",
                "description": "Failure error message.",
            },
            "error_type": {
                "type": "string",
            },
            "traceback": {
                "type": "string",
            },
            "reason": {
                "type": "string",
                "description": "Cancel/abort reason.",
            },
            "message": {
                "type": "string",
                "description": "Status message or reason.",
            },
            "metadata": {
                "type": "object",
                "description": "Metadata to merge into the task.",
            },
            "replace_metadata": {
                "type": "boolean",
                "default": False,
            },
            "target": {
                "type": "string",
                "description": "auto | task_manager | team",
                "default": "auto",
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
        action = normalize_action(args.get("action"))
        task_id = ""

        try:
            if not has_task_runtime(context):
                raise TaskUpdateRuntimeMissingError(
                    "task_update requires context.metadata['task_manager'] "
                    "or a team/swarm object with task state."
                )

            task_id = require_task_id(args)
            raw_status = args.get("status")
            status = normalize_common_status(
                str(raw_status)
                if raw_status is not None
                else None,
                action=action,
            )

            update_metadata = optional_dict_arg(args, "metadata")
            target = str(args.get("target") or "auto").strip().lower()

            if target not in {"auto", "task_manager", "team"}:
                raise ToolValidationError("target must be auto, task_manager, or team")

            updated_targets: list[str] = []
            task_record_data: dict[str, Any] | None = None
            team_task_data: dict[str, Any] | None = None

            task_manager = resolve_task_manager(context)

            if target in {"auto", "task_manager"} and task_manager is not None:
                task_record = await get_task_record(
                    task_manager,
                    task_id,
                )

                if task_record is not None:
                    update_task_record_status(
                        task_record,
                        status,
                        args,
                        metadata=update_metadata,
                    )

                    await persist_task_record(
                        task_manager,
                        task_record,
                    )

                    updated_targets.append("task_manager")
                    task_record_data = task_record_to_data(task_record)

            team = resolve_team(context)

            if target in {"auto", "team"} and team is not None:
                team_task = team.get_shared_task(task_id)

                if team_task is not None:
                    update_team_task_status(
                        team_task,
                        status,
                        args,
                        metadata=update_metadata,
                    )

                    updated_targets.append("team")
                    team_task_data = team_task.to_dict()

            if not updated_targets:
                raise TaskUpdateNotFoundError(f"task not found: {task_id}")

            final_status = status or "metadata_updated"

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Task updated: {task_id} -> {final_status}",
                data={
                    "action": action.value,
                    "task_id": task_id,
                    "status": final_status,
                    "updated_targets": updated_targets,
                    "task_record": task_record_data,
                    "team_task": team_task_data,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"task_update failed: {exc}",
                data={
                    "action": action.value,
                    "task_id": task_id,
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


__all__ = [
    "TaskUpdateAction",
    "TaskUpdateNotFoundError",
    "TaskUpdateRuntimeMissingError",
    "TaskUpdateTool",
    "TaskUpdateToolError",
    "has_task_runtime",
    "normalize_action",
    "normalize_common_status",
    "resolve_task_manager",
    "resolve_team",
]