from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pywork.llm.router import create_llm_router
from pywork.schemas.message_schema import AnyMessage, message_from_dict
from pywork.subagents.manager import (
    SubAgentManager,
    create_default_subagent_manager,
)
from pywork.teams.mailbox import AgentMailbox, create_agent_mailbox
from pywork.teams.team import Team, create_team
from pywork.tools.registry import ToolRegistry

from pywork.runtime.events import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeEventSource,
)
from pywork.tasks.task import TaskEvent

def registry_get_tool(
    registry: Any,
    name: str,
) -> Any | None:
    get_tool = getattr(registry, "get", None)

    if not callable(get_tool):
        return None

    try:
        return get_tool(name)
    except Exception:
        return None


def registry_tool_definitions(registry: Any) -> list[dict[str, Any]]:
    list_definitions = getattr(registry, "list_definitions", None)

    if not callable(list_definitions):
        return []

    try:
        definitions = list_definitions()
    except Exception:
        return []

    if not isinstance(definitions, list):
        return []

    return [
        dict(item)
        for item in definitions
        if isinstance(item, Mapping)
    ]


def resolve_agent_tool_manager(registry: Any) -> SubAgentManager | None:
    agent_tool = registry_get_tool(registry, "agent")

    if agent_tool is None:
        return None

    manager = getattr(agent_tool, "manager", None)

    if isinstance(manager, SubAgentManager):
        return manager

    fallback_runtime = getattr(agent_tool, "_fallback_runtime", None)

    if fallback_runtime is not None:
        fallback_manager = getattr(fallback_runtime, "manager", None)

        if isinstance(fallback_manager, SubAgentManager):
            return fallback_manager

    return None


def bind_agent_tool_manager(
    registry: Any,
    manager: SubAgentManager,
) -> None:
    """
    让 AgentTool 使用共享 SubAgentManager。

    这样 AgentTool 就不会再临时创建 fallback manager，
    TUI / Runtime / AgentTool 看到的是同一个 task_manager。
    """
    agent_tool = registry_get_tool(registry, "agent")

    if agent_tool is None:
        return

    if hasattr(agent_tool, "manager"):
        agent_tool.manager = manager

    if hasattr(agent_tool, "_fallback_runtime"):
        agent_tool._fallback_runtime = None


def resolve_mailbox_from_metadata(
    metadata: dict[str, Any],
) -> AgentMailbox | None:
    mailbox = metadata.get("mailbox")

    if isinstance(mailbox, AgentMailbox):
        return mailbox

    team = metadata.get("team")

    if team is not None:
        team_mailbox = getattr(team, "mailbox", None)

        if isinstance(team_mailbox, AgentMailbox):
            return team_mailbox

    swarm = metadata.get("swarm")

    if swarm is not None:
        swarm_team = getattr(swarm, "team", None)
        swarm_mailbox = getattr(swarm_team, "mailbox", None)

        if isinstance(swarm_mailbox, AgentMailbox):
            return swarm_mailbox

    teammate = metadata.get("teammate")

    if teammate is not None:
        teammate_mailbox = getattr(teammate, "mailbox", None)

        if isinstance(teammate_mailbox, AgentMailbox):
            return teammate_mailbox

    return None


def ensure_team_registry(
    metadata: dict[str, Any],
) -> Any:
    registry = metadata.get("team_registry")

    if registry is not None:
        return registry

    teams = metadata.get("teams")

    if teams is not None:
        metadata["team_registry"] = teams
        return teams

    registry = {}
    metadata["team_registry"] = registry

    return registry


def resolve_team_from_metadata(
    metadata: dict[str, Any],
) -> Team | None:
    team = metadata.get("team")

    if isinstance(team, Team):
        return team

    swarm = metadata.get("swarm")

    if swarm is not None:
        swarm_team = getattr(swarm, "team", None)

        if isinstance(swarm_team, Team):
            return swarm_team

    team_registry = metadata.get("team_registry")

    if isinstance(team_registry, Mapping):
        for value in team_registry.values():
            if isinstance(value, Team):
                return value

    teams = metadata.get("teams")

    if isinstance(teams, Mapping):
        for value in teams.values():
            if isinstance(value, Team):
                return value

    return None


def register_runtime_team(
    metadata: dict[str, Any],
    team: Team,
) -> None:
    team_registry = ensure_team_registry(metadata)

    if isinstance(team_registry, dict):
        team_registry[team.team_id] = team
        return

    register = getattr(team_registry, "register", None)

    if callable(register):
        try:
            register(team)
            return
        except Exception:
            pass

    add_team = getattr(team_registry, "add_team", None)

    if callable(add_team):
        try:
            add_team(team)
            return
        except Exception:
            pass


def ensure_runtime_team(
    metadata: dict[str, Any],
    *,
    mailbox: AgentMailbox,
    manager: SubAgentManager,
    registry: ToolRegistry | None = None,
    workspace_path: str | Path = ".",
) -> Team:
    """
    确保 Runtime 有默认 Team，并且 Team 使用共享 mailbox。

    这样：
    - send_message 可以从 metadata["team"].mailbox 拿邮箱
    - TeammateAgent 由 Team 创建时会复用同一个 mailbox
    - TUI TeamViewPanel 可以展示 mailbox 统计
    """
    team = resolve_team_from_metadata(metadata)

    if team is None:
        team = create_team(
            team_id="runtime_team",
            name="Runtime Team",
            description="Default runtime team for agent messaging.",
            mailbox=mailbox,
            manager=manager,
            tool_definitions=registry_tool_definitions(registry),
            workspace_path=workspace_path,
            metadata={
                "source": "runtime.shared_objects",
            },
        )

    metadata["team"] = team
    metadata["mailbox"] = team.mailbox

    register_runtime_team(
        metadata,
        team,
    )

    return team

def task_event_to_runtime_status(event: TaskEvent) -> str:
    event_type = str(getattr(event.event_type, "value", event.event_type))

    mapping = {
        "created": "task_created",
        "queued": "task_queued",
        "started": "task_started",
        "retrying": "task_retrying",
        "succeeded": "task_finished",
        "failed": "task_failed",
        "cancelled": "task_cancelled",
        "aborted": "task_aborted",
        "updated": "task_updated",
    }

    return mapping.get(event_type, f"task_{event_type}")


def task_event_to_runtime_metadata(event: TaskEvent) -> dict[str, Any]:
    event_type = str(getattr(event.event_type, "value", event.event_type))
    status = str(getattr(event.status, "value", event.status))

    to_dict = getattr(event, "to_dict", None)

    if callable(to_dict):
        try:
            event_data = to_dict()
        except Exception:
            event_data = {}
    else:
        event_data = {}

    return {
        "category": "task",
        "task_event": True,
        "task_id": event.task_id,
        "task_event_type": event_type,
        "task_status": status,
        "task_event_data": event_data,
    }


def bridge_task_manager_events_to_runtime_event_bus(
    metadata: dict[str, Any],
    *,
    task_manager: Any,
    event_bus: RuntimeEventBus | None,
) -> None:
    """
    把 TaskManager 自己的 TaskEvent 桥接到 RuntimeEventBus。

    这样 TUI 不必一直轮询 TaskManager。
    后续 TaskPanel 可以收到 task_started / task_finished / task_failed
    之类的 RuntimeEvent 后局部刷新。
    """
    if task_manager is None or event_bus is None:
        return

    add_event_handler = getattr(task_manager, "add_event_handler", None)

    if not callable(add_event_handler):
        return

    bridge_handlers = getattr(
        task_manager,
        "_runtime_event_bus_bridge_handlers",
        None,
    )

    if not isinstance(bridge_handlers, dict):
        bridge_handlers = {}
        setattr(
            task_manager,
            "_runtime_event_bus_bridge_handlers",
            bridge_handlers,
        )

    bus_key = id(event_bus)

    if bus_key in bridge_handlers:
        return

    async def handle_task_event(event: TaskEvent) -> None:
        runtime_event = RuntimeEvent.status_event(
            status=task_event_to_runtime_status(event),
            content=getattr(event, "message", "") or "",
            source=RuntimeEventSource.SYSTEM,
            metadata={
                **task_event_to_runtime_metadata(event),
                "source": "task_manager_event_bridge",
            },
        )

        await event_bus.emit_async(runtime_event)

    add_event_handler(handle_task_event)
    bridge_handlers[bus_key] = handle_task_event

    metadata["task_event_bridge_ready"] = True


def has_runtime_llm_config(config: dict[str, Any] | None) -> bool:
    if not isinstance(config, dict):
        return False

    llm_config = config.get("llm")

    if isinstance(llm_config, dict) and llm_config:
        return True

    return bool(
        config.get("providers")
        or config.get("provider")
        or config.get("model")
    )


def normalize_subagent_llm_messages(
    messages: list[dict[str, Any]],
) -> list[AnyMessage]:
    normalized: list[AnyMessage] = []

    for message in messages:
        if isinstance(message, dict):
            normalized.append(
                message_from_dict(
                    normalize_subagent_llm_message_dict(message)
                )
            )
            continue

        normalized.append(
            message_from_dict(
                {
                    "role": "user",
                    "content": str(message),
                }
            )
        )

    return normalized


def normalize_subagent_llm_message_dict(
    message: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert lightweight runtime messages to strict schema messages.

    AgentMessage.to_dict() includes optional fields like tool_call_id even when
    they are None. System/User/Assistant schemas forbid that extra field, so the
    SubAgent LLM adapter must strip it before validation.
    """
    role = str(message.get("role", "user"))
    payload = {
        key: value
        for key, value in message.items()
        if value is not None
    }

    if role == "tool":
        tool_name = payload.get("tool_name") or payload.get("name")
        tool_call_id = payload.get("tool_call_id")

        if tool_name and tool_call_id:
            payload["tool_name"] = str(tool_name)
            payload["tool_call_id"] = str(tool_call_id)
            return payload

        content = str(payload.get("content", "") or "")

        if tool_name:
            content = f"Tool observation from {tool_name}:\n\n{content}"
        else:
            content = f"Tool observation:\n\n{content}"

        metadata = payload.get("metadata", {})

        return {
            "role": "user",
            "content": content,
            "metadata": metadata if isinstance(metadata, dict) else {},
        }

    if role != "tool":
        payload.pop("tool_call_id", None)
        payload.pop("tool_name", None)

    return payload


def create_runtime_subagent_llm(
    config: dict[str, Any] | None,
) -> Any | None:
    if not has_runtime_llm_config(config):
        return None

    router = create_llm_router(config)

    async def runtime_subagent_llm(
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        return await router.chat(
            normalize_subagent_llm_messages(messages),
            tools=tools,
            metadata=metadata or {},
        )

    return runtime_subagent_llm


def ensure_subagent_manager(
    metadata: dict[str, Any],
    *,
    registry: ToolRegistry | None = None,
    workspace_path: str | Path = ".",
    config: dict[str, Any] | None = None,
) -> SubAgentManager:
    existing = metadata.get("subagent_manager")
    llm = create_runtime_subagent_llm(config)

    if isinstance(existing, SubAgentManager):
        manager = existing
    else:
        manager = resolve_agent_tool_manager(registry) if registry is not None else None

    if manager is None:
        task_manager = metadata.get("task_manager")

        manager = create_default_subagent_manager(
            llm=llm,
            workspace_path=workspace_path,
            task_manager=task_manager,
            tool_definitions=registry_tool_definitions(registry),
            metadata={
                "source": "runtime.shared_objects",
            },
        )

    if getattr(manager, "llm", None) is None and llm is not None:
        manager.llm = llm

    metadata["subagent_manager"] = manager
    metadata["manager"] = manager
    metadata["task_manager"] = manager.task_manager

    if registry is not None:
        bind_agent_tool_manager(
            registry,
            manager,
        )

    return manager


def ensure_runtime_shared_objects(
    metadata: dict[str, Any] | None = None,
    *,
    registry: ToolRegistry | None = None,
    workspace_path: str | Path = ".",
    config: dict[str, Any] | None = None,
    event_bus: RuntimeEventBus | None = None,
) -> dict[str, Any]:
    """
    确保 Runtime / ToolExecutionContext 使用同一批共享对象。

    会注入：
    - task_manager
    - subagent_manager
    - mailbox
    - team
    - team_registry
    - tool_registry / registry
    - config
    """
    shared = metadata if isinstance(metadata, dict) else {}

    if registry is not None:
        shared["tool_registry"] = registry
        shared["registry"] = registry

    if config is not None:
        shared["config"] = config

    shared["workspace_path"] = str(Path(workspace_path).expanduser().resolve())

    ensure_team_registry(shared)

    mailbox = resolve_mailbox_from_metadata(shared)

    if mailbox is None:
        mailbox = create_agent_mailbox(
            metadata={
                "owner": "RuntimeSharedObjects",
            }
        )

    shared["mailbox"] = mailbox

    manager = ensure_subagent_manager(
        shared,
        registry=registry,
        workspace_path=workspace_path,
        config=config,
    )

    team = ensure_runtime_team(
        shared,
        mailbox=mailbox,
        manager=manager,
        registry=registry,
        workspace_path=workspace_path,
    )

    shared["team"] = team
    shared["mailbox"] = team.mailbox
    shared["subagent_manager"] = manager
    shared["task_manager"] = manager.task_manager

    if event_bus is not None:
        shared["event_bus"] = event_bus

    bridge_task_manager_events_to_runtime_event_bus(
        shared,
        task_manager=manager.task_manager,
        event_bus=event_bus,
    )

    shared["runtime_shared_objects_ready"] = True
    return shared
