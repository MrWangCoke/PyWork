from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.subagents.base import SubAgentLLMCallable, SubAgentToolScope
from pywork.subagents.manager import SubAgentManager
from pywork.teams.mailbox import AgentMailbox, create_agent_mailbox, safe_jsonable
from pywork.teams.team import Team, TeamConfig, create_team
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


TEAM_CREATE_RISK = getattr(ToolRiskLevel, "MEDIUM", ToolRiskLevel.LOW)


class TeamCreateToolError(Exception):
    """team_create tool 基础异常。"""


class TeamRegistryError(TeamCreateToolError):
    """Team registry 异常。"""


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

    if isinstance(metadata, dict):
        return metadata

    if isinstance(metadata, Mapping):
        return dict(metadata)

    return {}


def object_has_attr(value: Any, attr: str) -> bool:
    return hasattr(value, attr) and getattr(value, attr) is not None


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


def resolve_team_registry(
    context: ToolExecutionContext,
    *,
    create_if_missing: bool = False,
) -> Any:
    metadata = context_metadata(context)

    registry = metadata.get("team_registry")

    if registry is None:
        registry = metadata.get("teams")

    if registry is not None:
        return registry

    if create_if_missing and isinstance(metadata, dict):
        registry = {}
        metadata["team_registry"] = registry
        return registry

    raise TeamRegistryError(
        "team registry is required in context.metadata['team_registry']"
    )


def registry_get_team(
    registry: Any,
    team_id: str,
) -> Team | None:
    if isinstance(registry, Mapping):
        team = registry.get(team_id)

        if isinstance(team, Team):
            return team

        return None

    for method_name in ("get_team", "require_team"):
        method = getattr(registry, method_name, None)

        if callable(method):
            try:
                team = method(team_id)
            except Exception:
                continue

            if isinstance(team, Team):
                return team

    teams = getattr(registry, "teams", None)

    if isinstance(teams, Mapping):
        team = teams.get(team_id)

        if isinstance(team, Team):
            return team

    return None


def registry_register_team(
    registry: Any,
    team: Team,
    *,
    replace: bool = False,
) -> None:
    team_id = team.team_id

    if isinstance(registry, MutableMapping):
        if team_id in registry and not replace:
            raise TeamRegistryError(f"team already exists: {team_id}")

        registry[team_id] = team
        return

    existing = registry_get_team(registry, team_id)

    if existing is not None and not replace:
        raise TeamRegistryError(f"team already exists: {team_id}")

    for method_name in ("register_team", "add_team"):
        method = getattr(registry, method_name, None)

        if callable(method):
            method(team)
            return

    teams = getattr(registry, "teams", None)

    if isinstance(teams, MutableMapping):
        teams[team_id] = team
        return

    raise TeamRegistryError(
        "team registry does not support registration"
    )


def resolve_manager(context: ToolExecutionContext) -> SubAgentManager | None:
    metadata = context_metadata(context)

    manager = (
        metadata.get("subagent_manager")
        or metadata.get("manager")
    )

    if isinstance(manager, SubAgentManager):
        return manager

    return manager


def resolve_llm(context: ToolExecutionContext) -> SubAgentLLMCallable | None:
    metadata = context_metadata(context)

    llm = metadata.get("llm") or metadata.get("agent_llm")

    if callable(llm):
        return llm

    return None


def resolve_tool_definitions(context: ToolExecutionContext) -> Sequence[dict[str, Any]] | None:
    metadata = context_metadata(context)

    tool_definitions = metadata.get("tool_definitions")

    if isinstance(tool_definitions, Sequence) and not isinstance(tool_definitions, str):
        return list(tool_definitions)

    return None


def resolve_mailbox(context: ToolExecutionContext) -> AgentMailbox | None:
    metadata = context_metadata(context)

    mailbox = metadata.get("mailbox")

    if isinstance(mailbox, AgentMailbox):
        return mailbox

    return None


def build_team_config(args: Mapping[str, Any]) -> TeamConfig:
    config = args.get("config")

    if not isinstance(config, Mapping):
        config = {}

    return TeamConfig(
        default_assignment_strategy=str(
            config.get("default_assignment_strategy")
            or args.get("default_assignment_strategy")
            or "round_robin"
        ),
        auto_mark_result_messages_read=bool(
            config.get("auto_mark_result_messages_read", True)
        ),
        auto_ack_result_messages=bool(
            config.get("auto_ack_result_messages", True)
        ),
    )


def build_tool_scope(value: Any) -> SubAgentToolScope | None:
    if value is None:
        return None

    if isinstance(value, SubAgentToolScope):
        return value

    if not isinstance(value, Mapping):
        raise ToolValidationError("tool_scope must be an object")

    allowed_tools = value.get("allowed_tools")
    denied_tools = value.get("denied_tools")

    return SubAgentToolScope(
        allowed_tools=(
            frozenset(str(item) for item in allowed_tools)
            if isinstance(allowed_tools, Sequence) and not isinstance(allowed_tools, str)
            else None
        ),
        denied_tools=(
            frozenset(str(item) for item in denied_tools)
            if isinstance(denied_tools, Sequence) and not isinstance(denied_tools, str)
            else frozenset()
        ),
        permission_mode=(
            str(value["permission_mode"])
            if value.get("permission_mode") is not None
            else None
        ),
    )


def normalize_member_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {
            "role": value,
        }

    if isinstance(value, Mapping):
        return dict(value)

    raise ToolValidationError("members must contain strings or objects")


def default_member_specs() -> list[dict[str, Any]]:
    return [
        {
            "teammate_id": "planner_1",
            "role": "planner",
            "name": "Planner",
        },
        {
            "teammate_id": "reviewer_1",
            "role": "reviewer",
            "name": "Reviewer",
        },
        {
            "teammate_id": "verifier_1",
            "role": "verifier",
            "name": "Verifier",
        },
        {
            "teammate_id": "general_1",
            "role": "general",
            "name": "General",
        },
    ]


def create_members_for_team(
    team: Team,
    args: Mapping[str, Any],
) -> list[dict[str, Any]]:
    members_arg = args.get("members")
    member_specs: list[dict[str, Any]] = []

    if isinstance(members_arg, Sequence) and not isinstance(members_arg, str):
        member_specs.extend(
            normalize_member_spec(item)
            for item in members_arg
        )

    if optional_bool_arg(args, "create_default_members", default=False):
        existing_ids = {
            str(spec.get("teammate_id") or "")
            for spec in member_specs
        }

        for spec in default_member_specs():
            if spec["teammate_id"] not in existing_ids:
                member_specs.append(spec)

    created: list[dict[str, Any]] = []

    for spec in member_specs:
        role = str(spec.get("role") or "general")
        teammate_id = (
            str(spec["teammate_id"])
            if spec.get("teammate_id") is not None
            else None
        )

        member = team.create_teammate(
            teammate_id=teammate_id,
            name=str(spec.get("name") or teammate_id or role),
            role=role,
            agent_name=(
                str(spec["agent_name"])
                if spec.get("agent_name") is not None
                else None
            ),
            description=str(spec.get("description") or ""),
            tool_scope=build_tool_scope(spec.get("tool_scope")),
            max_steps=(
                int(spec["max_steps"])
                if spec.get("max_steps") is not None
                else None
            ),
            metadata=dict(spec.get("metadata") or {}),
            replace=bool(spec.get("replace", False)),
        )

        created.append(member.to_dict())

    return created


class TeamCreateTool(BaseTool):
    name = "team_create"
    description = "Create a Team and register it into the runtime team registry."
    risk = TEAM_CREATE_RISK

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "team_id": {
                "type": "string",
            },
            "name": {
                "type": "string",
            },
            "description": {
                "type": "string",
            },
            "workspace_path": {
                "type": "string",
            },
            "replace": {
                "type": "boolean",
                "default": False,
            },
            "create_default_members": {
                "type": "boolean",
                "default": False,
            },
            "members": {
                "type": "array",
                "description": "List of teammate specs. String item means role name.",
            },
            "metadata": {
                "type": "object",
            },
            "config": {
                "type": "object",
            },
            "use_shared_mailbox": {
                "type": "boolean",
                "default": True,
            },
            "set_current": {
                "type": "boolean",
                "default": False,
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
            metadata = context_metadata(context)
            registry = resolve_team_registry(
                context,
                create_if_missing=True,
            )

            team_id = optional_text_arg(args, "team_id", aliases=("id",)).strip() or None
            name = optional_text_arg(args, "name", default="")
            description = optional_text_arg(args, "description", default="")
            workspace_path = (
                args.get("workspace_path")
                or getattr(context, "workspace_path", ".")
                or "."
            )

            replace = optional_bool_arg(args, "replace", default=False)

            if team_id is not None and registry_get_team(registry, team_id) is not None and not replace:
                raise TeamRegistryError(f"team already exists: {team_id}")

            mailbox = None

            if optional_bool_arg(args, "use_shared_mailbox", default=True):
                mailbox = resolve_mailbox(context)

            if mailbox is None:
                mailbox = create_agent_mailbox(
                    metadata={
                        "owner": "TeamCreateTool",
                        "team_id": team_id,
                    }
                )

            team = create_team(
                team_id=team_id,
                name=name,
                description=description,
                mailbox=mailbox,
                manager=resolve_manager(context),
                llm=resolve_llm(context),
                tool_definitions=resolve_tool_definitions(context),
                workspace_path=Path(workspace_path),
                config=build_team_config(args),
                metadata={
                    "created_by": "team_create",
                    **optional_dict_arg(args, "metadata"),
                },
            )

            created_members = create_members_for_team(
                team,
                args,
            )

            registry_register_team(
                registry,
                team,
                replace=replace,
            )

            if optional_bool_arg(args, "set_current", default=False) and isinstance(metadata, dict):
                metadata["team"] = team

            return make_result(
                call,
                tool_name=self.name,
                success=True,
                content=f"Team created: {team.team_id}",
                data={
                    "team": team.to_dict(),
                    "created_members": created_members,
                    "registered": True,
                    "registry_key": team.team_id,
                },
            )

        except Exception as exc:
            return make_result(
                call,
                tool_name=self.name,
                success=False,
                content=f"team_create failed: {exc}",
                data={
                    "error_type": type(exc).__name__,
                },
                error=str(exc),
            )


__all__ = [
    "TeamCreateTool",
    "TeamCreateToolError",
    "TeamRegistryError",
    "build_team_config",
    "create_members_for_team",
    "default_member_specs",
    "registry_get_team",
    "registry_register_team",
    "resolve_team_registry",
]