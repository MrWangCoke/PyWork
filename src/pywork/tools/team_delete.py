from __future__ import annotations

import inspect
from collections.abc import Mapping, MutableMapping
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.teams.team import Team
from pywork.tools.team_create import (
    TeamRegistryError,
    context_metadata,
    object_has_attr,
    registry_get_team,
    resolve_team_registry,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


TEAM_DELETE_RISK = getattr(ToolRiskLevel, "HIGH", ToolRiskLevel.LOW)


class TeamDeleteToolError(Exception):
    """team_delete tool 基础异常。"""


class TeamDeleteNotFoundError(TeamDeleteToolError):
    """找不到要删除的 Team。"""


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


def require_team_id(args: Mapping[str, Any]) -> str:
    value = args.get("team_id") or args.get("id")

    text = str(value or "").strip()

    if not text:
        raise ToolValidationError("team_id is required")

    return text


def registry_delete_team(
    registry: Any,
    team_id: str,
) -> tuple[bool, Team | None]:
    if isinstance(registry, MutableMapping):
        team = registry.pop(team_id, None)
        return team is not None, team if isinstance(team, Team) else None

    existing_before = registry_get_team(registry, team_id)

    for method_name in ("delete_team", "remove_team", "unregister_team"):
        method = getattr(registry, method_name, None)

        if callable(method):
            result = method(team_id)

            if isinstance(result, Team):
                return True, result

            if result is True:
                return True, existing_before

            existing_after = registry_get_team(registry, team_id)

            if existing_before is not None and existing_after is None:
                return True, existing_before

    teams = getattr(registry, "teams", None)

    if isinstance(teams, MutableMapping):
        team = teams.pop(team_id, None)
        return team is not None, team if isinstance(team, Team) else None

    return False, None


def resolve_current_team(context: ToolExecutionContext) -> Team | None:
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


async def shutdown_team(
    team: Team,
    *,
    stop_members: bool,
    cancel_current: bool,
    reason: str,
) -> dict[str, Any]:
    data = {
        "cancelled_current_count": 0,
        "stopped_member_count": 0,
    }

    if cancel_current:
        data["cancelled_current_count"] = await team.cancel_all_current(
            reason=reason,
        )

    if stop_members:
        data["stopped_member_count"] = await team.stop_all_teammates(
            reason=reason,
        )

    return data


class TeamDeleteTool(BaseTool):
    name = "team_delete"
    description = "Delete a Team from the runtime team registry and optionally stop its teammates."
    risk = TEAM_DELETE_RISK

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "team_id": {
                "type": "string",
            },
            "stop_members": {
                "type": "boolean",
                "default": True,
            },
            "cancel_current": {
                "type": "boolean",
                "default": True,
            },
            "clear_current": {
                "type": "boolean",
                "default": True,
            },
            "reason": {
                "type": "string",
            },
        },
        "required": [
            "team_id",
        ],
        "additionalProperties": True,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = get_call_args(call)

        try:
            metadata = context_metadata(context)
            team_id = require_team_id(args)
            reason = str(args.get("reason") or f"team deleted: {team_id}")

            registry = None

            try:
                registry = resolve_team_registry(
                    context,
                    create_if_missing=False,
                )
            except TeamRegistryError:
                registry = None

            team = None

            if registry is not None:
                team = registry_get_team(
                    registry,
                    team_id,
                )

            current_team = resolve_current_team(context)

            if team is None and current_team is not None and current_team.team_id == team_id:
                team = current_team

            if team is None:
                raise TeamDeleteNotFoundError(f"team not found: {team_id}")

            shutdown_data = await shutdown_team(
                team,
                stop_members=optional_bool_arg(args, "stop_members", default=True),
                cancel_current=optional_bool_arg(args, "cancel_current", default=True),
                reason=reason,
            )

            removed_from_registry = False

            if registry is not None:
                removed_from_registry, deleted_team = registry_delete_team(
                    registry,
                    team_id,
                )

            if (
                optional_bool_arg(args, "clear_current", default=True)
                and isinstance(metadata, dict)
                and metadata.get("team") is team
            ):
                metadata.pop("team", None)

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Team deleted: {team_id}",
                data={
                    "team_id": team_id,
                    "deleted_team": team.to_dict(),
                    "removed_from_registry": removed_from_registry,
                    **shutdown_data,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"team_delete failed: {exc}",
                data={
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


__all__ = [
    "TeamDeleteNotFoundError",
    "TeamDeleteTool",
    "TeamDeleteToolError",
    "registry_delete_team",
    "resolve_current_team",
    "shutdown_team",
]