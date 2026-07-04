from __future__ import annotations

from pywork.llm.router import create_llm_router
from pywork.schemas.message_schema import (
    AnyMessage,
    AssistantMessage,
    MessageRole,
    create_assistant_message,
    create_system_message,
    create_user_message,
)

from pywork.runtime.tool_result_payload import (
    append_tool_result_to_agent_state,
    build_tool_result_agent_content,
)

from pywork.llm.providers import LLMResponse
from pywork.schemas.message_schema import AssistantMessage, create_assistant_message
from pywork.schemas.tool_schema import ToolCall, create_tool_call

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

from pywork.permission.audit import (
    PermissionAuditLog,
    PermissionAuditUserAction,
)
from pywork.permission.session_overrides import (
    PermissionGateState,
    user_action_is_allow,
    user_action_is_always_allow,
)
from pywork.runtime.permission_gate import (
    PermissionGate,
    PermissionGateResult,
    render_permission_gate_result,
)
from pywork.runtime.state import AgentState, AgentStatus, create_agent_state
from pywork.runtime.events import (
    RuntimeEvent,
    RuntimeEventBus,
    RuntimeEventSource,
    RuntimeLifecycleEvent,
    get_default_event_bus,
    new_run_id,
)
from pywork.schemas.tool_schema import ToolCall, create_tool_call
from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.registry import ToolRegistry, create_default_registry
from pywork.tools.tool import ToolExecutionContext

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover
    END = "__end__"
    START = "__start__"
    StateGraph = None  # type: ignore[assignment]


GraphRoute = Literal["continue", "stop"]


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    reason: str = ""
    requires_confirmation: bool = False


PERMISSION_MODE_DEFAULT = "default"
PERMISSION_MODE_ACCEPT_EDITS = "accept_edits"
PERMISSION_MODE_PLAN = "plan"
PERMISSION_MODE_READONLY = "readonly"
PERMISSION_MODE_BYPASS = "bypass_permissions"

PERMISSION_MODE_ALIASES: dict[str, str] = {
    "": PERMISSION_MODE_DEFAULT,
    "normal": PERMISSION_MODE_DEFAULT,
    "default": PERMISSION_MODE_DEFAULT,
    "accept": PERMISSION_MODE_ACCEPT_EDITS,
    "accept-edits": PERMISSION_MODE_ACCEPT_EDITS,
    "accept_edits": PERMISSION_MODE_ACCEPT_EDITS,
    "plan": PERMISSION_MODE_PLAN,
    "planning": PERMISSION_MODE_PLAN,
    "readonly": PERMISSION_MODE_READONLY,
    "read_only": PERMISSION_MODE_READONLY,
    "read-only": PERMISSION_MODE_READONLY,
    "safe": PERMISSION_MODE_READONLY,
    "bypass": PERMISSION_MODE_BYPASS,
    "bypass_permissions": PERMISSION_MODE_BYPASS,
    "dangerous": PERMISSION_MODE_BYPASS,
}

VALID_PERMISSION_MODES: set[str] = {
    PERMISSION_MODE_DEFAULT,
    PERMISSION_MODE_ACCEPT_EDITS,
    PERMISSION_MODE_PLAN,
    PERMISSION_MODE_READONLY,
    PERMISSION_MODE_BYPASS,
}


def normalize_permission_mode(mode: str | None) -> str:
    text = str(mode or PERMISSION_MODE_DEFAULT).strip().lower()

    return PERMISSION_MODE_ALIASES.get(
        text,
        PERMISSION_MODE_DEFAULT,
    )


class AgentGraphData(TypedDict, total=False):

    registry: ToolRegistry
    llm_router: Any
    llm_response: Any
    assistant_message: AssistantMessage | None
    tool_definitions: list[dict[str, Any]]
    llm_error: str | None
    llm_output: str

    parsed_tool_calls: list[ToolCall]
    remaining_tool_calls: list[ToolCall]
    has_tool_call: bool
    """
    Whether the LLM output contains at least one tool call.

    Set by parse_tool_call_node after examining the LLM response.
    When True, the graph routes to permission_check -> execute_tool.
    When False and no assistant message is present, the graph stops.
    The graph also inspects agent_state tool_calls to avoid re-emitting
    tool call events that were already recorded in a previous iteration.
    """

    agent_state: AgentState
    user_input: str

    context: dict[str, Any]
    llm_output: str

    parsed_tool_call: ToolCall | None
    permission_decision: PermissionDecision | None
    permission_gate_result: PermissionGateResult | None
    permission_gate_error: str | None
    approval_handler: Any | None
    approval_result: Any | None
    permission_gate_state: PermissionGateState | None

    tool_result: ToolResult | None
    observation: str

    should_continue: bool
    stop_reason: str
    graph_route: GraphRoute
    route_reason: str
    awaiting_final_response: bool
    final_response_requested: bool
    pending_file_read_paths: list[str]
    completed_file_read_paths: list[str]
    file_read_batch_active: bool

    tool_registry: ToolRegistry
    config: dict[str, Any]
    metadata: dict[str, Any]
    run_id: str
    session_id: str | None
    event_bus: RuntimeEventBus
    emit_events: bool
    runtime_events: list[RuntimeEvent]
    emitted_tool_call_ids: set[str]


def get_nested_config_value(
    config: dict[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = config

    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def get_registry(data: AgentGraphData) -> ToolRegistry:
    registry = data.get("tool_registry")

    if isinstance(registry, ToolRegistry):
        return registry

    return create_default_registry()


def get_config(data: AgentGraphData) -> dict[str, Any]:
    config = data.get("config", {})

    if isinstance(config, dict):
        return config

    return {}


def get_permission_mode(data: AgentGraphData) -> str:
    config = get_config(data)

    raw_mode = get_nested_config_value(
        config,
        "permissions.mode",
        get_nested_config_value(
            config,
            "app.permission_mode",
            PERMISSION_MODE_DEFAULT,
        ),
    )

    return normalize_permission_mode(str(raw_mode))


def get_workspace_path(data: AgentGraphData) -> str:
    config = get_config(data)

    return str(
        get_nested_config_value(
            config,
            "workspace.path",
            ".",
        )
    )


def get_project_root(data: AgentGraphData) -> str:
    config = get_config(data)

    return str(
        get_nested_config_value(
            config,
            "workspace.project_root",
            get_workspace_path(data),
        )
    )


def create_default_agent_graph_state(
    *,
    user_input: str = "",
    registry: ToolRegistry | None = None,
    config: dict[str, Any] | None = None,
    agent_state: AgentState | None = None,
    metadata: dict[str, Any] | None = None,
) -> AgentGraphData:
    return {
        "agent_state": agent_state or create_agent_state(
            system_prompt=None,
            max_iterations=int(
                get_nested_config_value(
                    config or {},
                    "agent.max_iterations",
                    20,
                )
            ),
        ),
        "user_input": user_input,
        "tool_registry": registry or create_default_registry(),
        "config": config or {},
        "metadata": metadata or {},
        "context": {},
        "llm_output": "",
        "parsed_tool_call": None,
        "permission_decision": None,
        "permission_gate_result": None,
        "permission_gate_error": None,
        "approval_handler": None,
        "approval_result": None,
        "permission_gate_state": None,
        "tool_result": None,
        "observation": "",
        "should_continue": False,
        "stop_reason": "",
    }


def reset_agent_turn_state(state: AgentState) -> None:
    """
    Reset per-turn agent state before starting a new graph iteration.

    If the agent is in a terminal state (FINISHED, ERROR, CANCELLED),
    reset it to idle. Clears current_tool_call_id, last_error, resets
    the iteration counter, and updates the timestamp.

    Does NOT clear messages or tool_calls history.
    """
    if state.status in {
        AgentStatus.FINISHED,
        AgentStatus.ERROR,
        AgentStatus.CANCELLED,
    }:
        state.set_idle()

    state.current_tool_call_id = None
    state.last_error = None
    state.reset_iteration()
    state.touch()


def user_input_node(data: AgentGraphData) -> dict[str, Any]:
    """
    UserInput node — the entry point of the agent graph.

    Responsibilities:
    1. Reset per-turn agent state via reset_agent_turn_state().
    2. Append the user's input to AgentState.messages.
    3. Set the agent status to idle so the LLM node can proceed.
    4. Emit lifecycle (STARTED) and message events for the TUI.
    """
    state = data["agent_state"]
    user_input = data.get("user_input", "").strip()

    reset_agent_turn_state(state)

    emit_lifecycle_event(
        data,
        RuntimeLifecycleEvent.STARTED,
        content="runtime graph started",
    )

    if user_input:
        state.add_user_message(user_input)

        emit_message_event(
            data,
            "user",
            user_input,
            metadata={
                "node": "user_input",
            },
        )

    state.set_idle()

    emit_status_event(
        data,
        "thinking",
        content="building context",
        metadata={
            "node": "user_input",
        },
    )

    return data


def build_context_node(data: AgentGraphData) -> dict[str, Any]:
    """
    BuildContext node — assembles the context payload for the LLM call.

    Collects all information the LLM router needs:
    - messages (conversation history from AgentState)
    - tool definitions (from ToolRegistry)
    - workspace path and project root
    - current permission_mode and iteration counter
    """
    state = data["agent_state"]
    registry = get_registry(data)

    tool_definitions = get_graph_tool_definitions(data)

    context = {
        "messages": state.to_messages_payload(),
        "tool_definitions": tool_definitions,
        "workspace_path": get_workspace_path(data),
        "project_root": get_project_root(data),
        "permission_mode": get_permission_mode(data),
        "iteration": state.iteration,
        "checkpoint_id": state.checkpoint_id,
    }

    return {
        "context": context,
        "agent_state": state,
        "registry": registry,
        "tool_definitions": tool_definitions,
    }


def parse_tool_shortcut(user_input: str) -> dict[str, Any] | None:
    """
    Parse a /tool shortcut from user input, bypassing the LLM.

    Supports two formats:
        /tool echo hello
        /tool echo {"text": "hello"}

    When the user starts input with "/tool <name>", this function
    extracts the tool name and arguments directly, skipping the LLM
    call. Returns a dict with "tool_name" and "arguments", or None
    if the input is not a tool shortcut.
    """
    text = user_input.strip()

    if not text.startswith("/tool "):
        return None

    rest = text[len("/tool ") :].strip()

    if not rest:
        return None

    parts = rest.split(maxsplit=1)
    tool_name = parts[0]
    raw_args = parts[1] if len(parts) > 1 else ""

    arguments: dict[str, Any]

    if raw_args.startswith("{"):
        try:
            loaded = json.loads(raw_args)
            arguments = loaded if isinstance(loaded, dict) else {"input": loaded}
        except json.JSONDecodeError:
            arguments = {"input": raw_args}
    else:
        if tool_name == "echo":
            arguments = {"text": raw_args}
        else:
            arguments = {"input": raw_args}

    return {
        "tool_name": tool_name,
        "arguments": arguments,
    }


def mock_call_llm_output(data: AgentGraphData) -> str:
    """Temporary mock LLM output used when real LLM is unavailable."""
    user_input = data.get("user_input", "").strip()

    if data.get("awaiting_final_response"):
        result = data.get("tool_result")

        if isinstance(result, ToolResult):
            return (
                f"Tool `{result.tool_name}` result:\n\n"
                f"{result.content}"
            )

    shortcut_tool_call = parse_tool_shortcut(user_input)

    if shortcut_tool_call is not None:
        return json.dumps(
            shortcut_tool_call,
            ensure_ascii=False,
        )

    return (
        "Received your input:\n\n"
        f"> {user_input}\n\n"
        "The Runtime Graph is running in mock mode."
    )


READ_FILE_INTENT_PATTERN = re.compile(
    r"(read|summari[sz]e|inspect|analy[sz]e|look\s+at|open|查看|读取|读一下|读|总结|分析|看看)",
    re.IGNORECASE,
)

FILE_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.\\/:-]+\.(?:md|py|toml|txt|json|yaml|yml|ini|cfg|rst))",
    re.IGNORECASE,
)

REVIEWER_INTENT_PATTERN = re.compile(
    (
        r"(subagent|sub-agent|\u5b50\s*agent|\u5b50\u4ee3\u7406|"
        r"reviewer|review|code\s*review|\u5ba1\u67e5|\u5ba1\u6838|"
        r"\u68c0\u67e5\u4ee3\u7801|\u4ee3\u7801\u5ba1\u67e5)"
    ),
    re.IGNORECASE,
)

COORDINATOR_PARALLEL_INTENT_PATTERN = re.compile(
    r"(\u5e76\u884c|\u5e76\u53d1|\u540c\u65f6|\u4e00\u8d77\u8dd1|\u4e00\u8d77\u6267\u884c|parallel|concurrent|run\s+in\s+parallel)",
    re.IGNORECASE,
)

COORDINATOR_TASK_PREFIX_PATTERN = re.compile(
    r"^\s*(?:[-*\u2022]|\d+[).\u3001]|[\u4e00\u4e8c\u4e09\u56db\u4e94\u516d\u4e03\u516b\u4e5d\u5341]+[\u3001.\uff09)])\s*"
)


def split_coordinator_task_lines(text: str) -> list[str]:
    normalized = text.strip()

    # \u4f18\u5148\u53d6\u5192\u53f7\u540e\u9762\u7684\u4efb\u52a1\u5217\u8868\u3002
    for marker in ("\uff1a", ":"):
        if marker in normalized:
            normalized = normalized.split(marker, 1)[1].strip()
            break

    pieces = re.split(r"(?:\r?\n|[\uff1b;])+", normalized)

    tasks: list[str] = []

    for piece in pieces:
        item = COORDINATOR_TASK_PREFIX_PATTERN.sub("", piece).strip()

        if not item:
            continue

        # \u53bb\u6389\u5e38\u89c1\u5f15\u5bfc\u8bcd\u3002
        item = re.sub(
            r"^(\u628a)?(\u8fd9)?(\u4e09\u4e2a|3\u4e2a|\u591a\u4e2a)?\u4efb\u52a1(\u5e76\u884c|\u5e76\u53d1|\u540c\u65f6|\u4e00\u8d77)?(\u8dd1|\u6267\u884c)?",
            "",
            item,
            flags=re.IGNORECASE,
        ).strip(" \uff1a:\uff0c,\u3002")

        if item:
            tasks.append(item)

    # \u517c\u5bb9\u5355\u884c\uff1a1.xxx 2.xxx 3.xxx
    if len(tasks) <= 1:
        inline_parts = re.split(
            r"\s+(?=\d+[).\u3001]\s*)",
            normalized,
        )

        inline_tasks: list[str] = []

        for part in inline_parts:
            item = COORDINATOR_TASK_PREFIX_PATTERN.sub("", part).strip()

            if item:
                inline_tasks.append(item)

        if len(inline_tasks) > len(tasks):
            tasks = inline_tasks

    return tasks


def infer_coordinator_agent_name(task: str) -> str:
    lowered = task.lower()

    if any(word in lowered for word in ["review", "code review", "\u5ba1\u67e5", "\u5ba1\u6838", "\u4ee3\u7801\u5ba1\u67e5"]):
        return "reviewer"

    if any(word in lowered for word in ["test", "pytest", "verify", "\u9a8c\u8bc1", "\u6d4b\u8bd5", "\u8fd0\u884c\u6d4b\u8bd5"]):
        return "verifier"

    if any(word in lowered for word in ["debug", "diagnose", "\u8c03\u8bd5", "\u6392\u67e5", "\u4fee bug", "\u4feebug"]):
        return "debugger"

    if any(word in lowered for word in ["plan", "planning", "\u89c4\u5212", "\u8ba1\u5212", "\u62c6\u89e3", "\u65b9\u6848"]):
        return "planner"

    return "general"


def make_coordinator_parallel_tool_call(
    *,
    user_input: str,
    tasks: list[str],
) -> ToolCall:
    steps = [
        {
            "worker_id": f"worker_{index}",
            "agent_name": infer_coordinator_agent_name(task),
            "task": task,
            "metadata": {
                "source": "deterministic_parallel_route",
                "worker_index": index,
            },
        }
        for index, task in enumerate(tasks, start=1)
    ]

    return create_tool_call(
        tool_name="coordinator",
        arguments={
            "action": "run",
            "strategy": "parallel",
            "execution_mode": "task",
            "wait": True,
            "max_concurrency": len(steps),
            "steps": steps,
            "metadata": {
                "source": "runtime_graph.parallel_intent",
                "original_user_input": user_input,
            },
        },
        metadata={
            "source": "deterministic_coordinator_route",
            "strategy": "parallel",
            "execution_mode": "task",
            "worker_count": len(steps),
        },
    )


def detect_coordinator_parallel_tool_call(
    data: AgentGraphData,
) -> ToolCall | None:
    user_input = str(data.get("user_input", "") or "").strip()

    if not user_input or user_input.startswith("/"):
        return None

    if not COORDINATOR_PARALLEL_INTENT_PATTERN.search(user_input):
        return None

    tasks = split_coordinator_task_lines(user_input)

    if len(tasks) < 2:
        return None

    return make_coordinator_parallel_tool_call(
        user_input=user_input,
        tasks=tasks,
    )


DIRECT_FINISH_TOOL_NAMES: set[str] = {
    "file_write",
    "file_edit",
}


def should_finish_after_tool_result(result: ToolResult) -> bool:
    return result.tool_name in DIRECT_FINISH_TOOL_NAMES


def build_direct_tool_finish_message(result: ToolResult) -> str:
    return (
        "文件操作已完成。\n\n"
        f"Tool `{result.tool_name}` finished successfully.\n\n"
        f"{result.content}"
    )


def is_permission_blocked_tool_result(result: ToolResult) -> bool:
    metadata = result.metadata or {}
    data = result.data or {}

    return bool(
        metadata.get("permission_blocked")
        or data.get("permission_blocked")
    )


def build_permission_blocked_finish_message(result: ToolResult) -> str:
    metadata = result.metadata or {}
    decision = str(metadata.get("permission_decision") or "").strip()
    reason = str(metadata.get("permission_reason") or "").strip()
    user_denied = bool(metadata.get("user_denied"))

    if user_denied:
        title = "已取消执行。"
        detail = f"你拒绝了 `{result.tool_name}` 的授权，所以工具没有运行，文件也没有被修改。"
    elif decision == "deny":
        title = "操作已被安全规则阻止。"
        detail = f"`{result.tool_name}` 没有运行，文件没有被修改。"
    else:
        title = "操作需要授权，当前没有执行。"
        detail = f"`{result.tool_name}` 没有运行，文件没有被修改。"

    if reason:
        return f"{title}\n\n{detail}\n\n原因：{reason}"

    return f"{title}\n\n{detail}"


DIRECTORY_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.\\/-]+)(?:\s*(?:目录|文件夹|folder|directory))?",
    re.IGNORECASE,
)

TEXT_FILE_EXTENSIONS = {
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".rst",
}

GLOB_FILE_READ_MAX_RESULTS = 10000
FILE_READ_MAX_LINES = 5000
FILE_READ_MAX_CHARS = 500000


def normalize_workspace_relative_path(path: str) -> str:
    return path.strip().strip("`'\".,，。；;：:").replace("\\", "/")


def canonicalize_review_target_path(
    path: str,
    *,
    workspace_path: Path,
) -> str:
    """
    Support the common shorthand src/utils/foo.py for src/pywork/utils/foo.py.
    """
    normalized = normalize_workspace_relative_path(path)
    direct_candidate = (workspace_path / normalized).resolve()

    if (
        path_inside_workspace(direct_candidate, workspace_path=workspace_path)
        and direct_candidate.is_file()
    ):
        return normalized

    if normalized.startswith("src/utils/"):
        rewritten = "src/pywork/utils/" + normalized.removeprefix("src/utils/")
        rewritten_candidate = (workspace_path / rewritten).resolve()

        if (
            path_inside_workspace(rewritten_candidate, workspace_path=workspace_path)
            and rewritten_candidate.is_file()
        ):
            return rewritten

    return normalized


def build_reviewer_agent_task(
    *,
    target_path: str,
    user_input: str,
) -> str:
    return f"""
Review the code file `{target_path}`.

Original user request:
{user_input}

Instructions:
- Use the reviewer role.
- Focus on correctness, maintainability, safety, edge cases, and test coverage.
- Do not modify files.
- Do not run shell commands.
- Read and reason about the target file content before commenting.
- Mention concrete functions, classes, branches, or edge cases when possible.

Required output format:
1. Summary
2. Issues found
3. Safety and permission concerns
4. Test coverage gaps
5. Suggested fixes
6. Recommended next action
""".strip()


def make_reviewer_agent_tool_call(
    *,
    target_path: str,
    user_input: str,
) -> ToolCall:
    return create_tool_call(
        tool_name="agent",
        arguments={
            "action": "run",
            "agent_name": "reviewer",
            "task": build_reviewer_agent_task(
                target_path=target_path,
                user_input=user_input,
            ),
            "metadata": {
                "review_target_path": target_path,
                "review_original_request": user_input,
                "deterministic_route": "reviewer_file_review",
            },
        },
        metadata={
            "source": "deterministic_reviewer_route",
            "review_target_path": target_path,
            "user_input": user_input,
        },
    )


def detect_reviewer_subagent_tool_call(data: AgentGraphData) -> ToolCall | None:
    user_input = str(data.get("user_input", "") or "").strip()

    if not user_input or user_input.startswith("/"):
        return None

    if not REVIEWER_INTENT_PATTERN.search(user_input):
        return None

    path_match = FILE_PATH_PATTERN.search(user_input)

    if path_match is None:
        return None

    workspace_path = Path(get_workspace_path(data)).expanduser().resolve()
    raw_path = path_match.group("path")
    target_path = canonicalize_review_target_path(
        raw_path,
        workspace_path=workspace_path,
    )

    if Path(target_path).suffix.lower() != ".py":
        return None

    return make_reviewer_agent_tool_call(
        target_path=target_path,
        user_input=user_input,
    )


def is_text_file_path(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_FILE_EXTENSIONS


def make_file_read_tool_call(
    path: str,
    *,
    source: str,
    user_input: str | None = None,
) -> ToolCall:
    metadata: dict[str, Any] = {
        "source": source,
    }

    if user_input is not None:
        metadata["user_input"] = user_input

    return create_tool_call(
        tool_name="file_read",
        arguments={
            "path": normalize_workspace_relative_path(path),
            "max_lines": FILE_READ_MAX_LINES,
            "max_chars": FILE_READ_MAX_CHARS,
        },
        metadata=metadata,
    )


def make_glob_tool_call(
    pattern: str,
    *,
    source: str,
    user_input: str | None = None,
    max_results: int = GLOB_FILE_READ_MAX_RESULTS,
) -> ToolCall:
    metadata: dict[str, Any] = {
        "source": source,
    }

    if user_input is not None:
        metadata["user_input"] = user_input

    return create_tool_call(
        tool_name="glob",
        arguments={
            "pattern": pattern.replace("\\", "/"),
            "max_results": max_results,
            "include_files": True,
            "include_dirs": False,
        },
        metadata=metadata,
    )


def path_inside_workspace(path: Path, *, workspace_path: Path) -> bool:
    try:
        path.resolve().relative_to(workspace_path)
    except ValueError:
        return False

    return True


def detect_directory_path(user_input: str, *, workspace_path: Path) -> str | None:
    normalized_input = user_input.replace("\\", "/")

    for match in DIRECTORY_PATH_PATTERN.finditer(normalized_input):
        raw_path = normalize_workspace_relative_path(match.group("path"))

        if not raw_path or raw_path in {".", "-"}:
            continue

        if "." in Path(raw_path).name:
            continue

        candidate = (workspace_path / raw_path).resolve()

        if path_inside_workspace(candidate, workspace_path=workspace_path) and candidate.is_dir():
            return raw_path

    return None


def extract_glob_file_read_paths(result: ToolResult) -> list[str]:
    data = result.data or {}
    matches = data.get("matches", [])
    paths: list[str] = []

    if isinstance(matches, list):
        for item in matches:
            if not isinstance(item, dict):
                continue

            if item.get("kind") != "file":
                continue

            path = str(item.get("relative_path") or item.get("path") or "").strip()

            if path and is_text_file_path(path):
                paths.append(normalize_workspace_relative_path(path))

    if not paths:
        raw_paths = data.get("paths", [])

        if isinstance(raw_paths, list):
            for item in raw_paths:
                path = normalize_workspace_relative_path(str(item))

                if path and is_text_file_path(path):
                    paths.append(path)

    deduped: list[str] = []
    seen: set[str] = set()

    for path in paths:
        if path in seen:
            continue

        seen.add(path)
        deduped.append(path)

    return deduped


def detect_initial_file_tool_call(data: AgentGraphData) -> ToolCall | None:
    user_input = str(data.get("user_input", "") or "").strip()

    if not user_input or user_input.startswith("/"):
        return None

    if not READ_FILE_INTENT_PATTERN.search(user_input):
        return None

    workspace_path = Path(get_workspace_path(data)).expanduser().resolve()
    path_match = FILE_PATH_PATTERN.search(user_input)

    if path_match is not None:
        path = normalize_workspace_relative_path(path_match.group("path"))
        candidate = (workspace_path / path).resolve()

        if path_inside_workspace(candidate, workspace_path=workspace_path) and candidate.is_file():
            return make_file_read_tool_call(
                path,
                source="direct_file_read_intent",
                user_input=user_input,
            )

        return make_glob_tool_call(
            f"**/{Path(path).name}",
            source="filename_glob_read_intent",
            user_input=user_input,
        )

    directory_path = detect_directory_path(user_input, workspace_path=workspace_path)

    if directory_path is not None:
        return make_glob_tool_call(
            f"{directory_path}/**/*",
            source="directory_glob_read_intent",
            user_input=user_input,
        )

    return None


def detect_next_file_tool_call(data: AgentGraphData) -> ToolCall | None:
    pending_paths = list(data.get("pending_file_read_paths") or [])

    if pending_paths:
        next_path = pending_paths.pop(0)
        completed_paths = list(data.get("completed_file_read_paths") or [])
        completed_paths.append(next_path)

        data["pending_file_read_paths"] = pending_paths
        data["completed_file_read_paths"] = completed_paths
        data["file_read_batch_active"] = True

        return make_file_read_tool_call(
            next_path,
            source="batch_file_read",
        )

    if data.get("file_read_batch_active"):
        return None

    return detect_initial_file_tool_call(data)

DEFAULT_LLM_SYSTEM_PROMPT = """
You are PyWork, a local coding assistant operating inside the current workspace.

You have access to tools for reading and searching files:
- file_read: read the exact contents of a known file path.
- grep: search inside file contents for functions, classes, text, or regex patterns.
- glob: find files when the user asks to list/find files or when the path is unknown.

Critical tool-selection rules:
1. If the user asks to read, inspect, analyze, or summarize a specific known file, call file_read directly.
   Examples:
   - "read README.md" -> file_read {"path": "README.md"}
   - "summarize README.md" -> file_read {"path": "README.md"}
   - "look at src/pywork/runtime/graph.py" -> file_read {"path": "src/pywork/runtime/graph.py"}

2. Do not use glob for a known file such as README.md. Use glob only when the user asks to find/list files or the exact path is unknown.

3. If the user asks to search for code/text such as "async def", class names, or keywords, call grep.

4. After a tool result is provided, do not call another tool unless it is truly necessary. Answer the user directly using the tool result.

5. Never summarize a file you have not read. For file summaries, read the file first, then summarize the tool result clearly and briefly.

6. If the user asks to create a background task, mentions the Tasks panel, or says not to wait for completion,
   use task_create with target="subagent" and wait=false. Do not use agent.run for background tasks,
   because direct agent runs are not TaskRecords.
""".strip()


def get_nested_value(
    data: dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def graph_has_llm_config(config: dict[str, Any]) -> bool:
    """
    Check whether the config contains a usable LLM provider configuration.

    Looks for LLM-specific keys (provider, model, api_format, base_url,
    api_key, api_key_env, providers, default_provider) in either a nested
    "llm" key or at the top level. Returns False if no LLM config is found,
    which causes the graph to fall back to mock mode.
    """
    if not config:
        return False

    llm_keys = {
        "provider",
        "model",
        "api_format",
        "base_url",
        "api_key",
        "api_key_env",
        "providers",
        "default_provider",
    }

    llm_config = config.get("llm")

    if isinstance(llm_config, dict):
        return any(key in llm_config for key in llm_keys)

    return any(key in config for key in llm_keys)


def message_role_value(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))

    role = getattr(message, "role", "")

    if isinstance(role, MessageRole):
        return role.value

    return str(role)


def agent_message_to_llm_message(message: Any) -> AnyMessage | None:
    """
    Convert an AgentState message to the schemas/message_schema.py format.

    Maps role strings ("system", "user", "assistant", "tool", "error")
    to the corresponding Pydantic message model. Tool messages are converted
    to UserMessage (as observations) since the LLM router expects tool
    results wrapped as user messages.
    """
    role = message_role_value(message)

    if isinstance(message, dict):
        content = str(message.get("content", "") or "")
        metadata = dict(message.get("metadata", {}) or {})
    else:
        content = str(getattr(message, "content", "") or "")
        metadata = dict(getattr(message, "metadata", {}) or {})

    if role == "system":
        return create_system_message(
            content,
            metadata={
                **metadata,
                "source": "agent_state",
            },
        )

    if role == "user":
        return create_user_message(
            content,
            metadata={
                **metadata,
                "source": "agent_state",
            },
        )

    if role == "assistant":
        return create_assistant_message(
            content,
            metadata={
                **metadata,
                "source": "agent_state",
            },
        )

    if role == "tool":
        # AgentState stores tool results as "tool" messages. Since the LLM
        # expects tool results as user messages (observations), wrap the
        # tool result content with a descriptive prefix.
        return create_user_message(
            "Tool execution result:\n" + content,
            metadata={
                **metadata,
                "source": "agent_state_tool_observation",
            },
        )

    if role == "error":
        return create_user_message(
            "Runtime error:\n" + content,
            metadata={
                **metadata,
                "source": "agent_state_error",
            },
        )

    return None


def build_llm_messages(data: AgentGraphData) -> list[AnyMessage]:
    """
    Build the message list for the LLMRouter call.

    Constructs messages from the system prompt (from config or DEFAULT_LLM_SYSTEM_PROMPT)
    followed by converted AgentState messages (excluding system messages).
    Applies max_context_messages truncation, keeping the system prompt and
    the most recent history messages.
    """
    agent_state = data["agent_state"]
    config = data.get("config") or {}

    max_context_messages = int(
        get_nested_config_value(
            config,
            "agent.max_context_messages",
            20,
        )
    )

    system_prompt = str(
        get_nested_config_value(
            config,
            "llm.system_prompt",
            DEFAULT_LLM_SYSTEM_PROMPT,
        )
        or DEFAULT_LLM_SYSTEM_PROMPT
    )

    messages: list[AnyMessage] = [
        create_system_message(
            system_prompt,
            metadata={
                "source": "runtime_graph.system_prompt",
            },
        )
    ]

    for item in getattr(agent_state, "messages", []):
        if message_role_value(item) == "system":
            continue

        converted = agent_message_to_llm_message(item)

        if converted is not None:
            if getattr(converted, "content", None) is None:
                converted.content = ""

            messages.append(converted)

    if max_context_messages > 0:
        system_messages = messages[:1]
        history_messages = messages[1:]
        messages = system_messages + history_messages[-max_context_messages:]

    return messages


def get_graph_tool_definitions(data: AgentGraphData) -> list[dict[str, Any]]:
    registry = data.get("registry")

    if registry is None:
        return []

    if hasattr(registry, "list_definitions"):
        return registry.list_definitions()

    return []


def should_use_real_llm(data: AgentGraphData) -> bool:
    config = data.get("config") or {}
    return graph_has_llm_config(config)


def get_graph_tool_definitions(data: AgentGraphData) -> list[dict[str, Any]]:
    """
    Retrieve tool definitions from the ToolRegistry for LLM tool calling.

    Caches the result in data["tool_definitions"]. Each definition includes
    the tool name, description, input_schema (JSON Schema), and risk_level.
    The LLM router uses these definitions to decide which tools to call.
    Format:
        {
            "name": "file_read",
            "description": "...",
            "input_schema": {...},
            "risk_level": "safe"
        }
    """
    cached = data.get("tool_definitions")

    if isinstance(cached, list):
        return cached

    registry = data.get("registry")

    if registry is None:
        registry = create_default_registry()
        data["registry"] = registry

    if not hasattr(registry, "list_definitions"):
        data["tool_definitions"] = []
        return []

    tool_definitions = registry.list_definitions()
    data["tool_definitions"] = tool_definitions

    return tool_definitions

async def call_real_llm(data: AgentGraphData) -> AssistantMessage:
    """
    Call the real LLM through LLMRouter and return the AssistantMessage.

    Creates the LLM router from config if not already cached in data,
    builds the message list and tool definitions, then calls router.chat().
    Stores the raw LLMResponse in data["llm_response"] for later parsing.
    """
    config = data.get("config") or {}

    router = data.get("llm_router")

    if router is None:
        router = create_llm_router(config)
        data["llm_router"] = router

    messages = build_llm_messages(data)

    tool_definitions = get_graph_tool_definitions(data)

    response = await router.chat(
        messages,
        tools=tool_definitions,
        metadata={
            "source": "runtime_graph.call_llm_node",
            "tool_count": len(tool_definitions),
        },
    )

    data["llm_response"] = response

    return response.message


def add_assistant_message_to_agent_state(
    agent_state: Any,
    message: AssistantMessage,
) -> None:
    """
    Append an AssistantMessage to the AgentState message history.
    """
    metadata = {
        "source": "llm_router",
        "tool_call_count": len(message.tool_calls),
    }

    if hasattr(agent_state, "add_assistant_message"):
        agent_state.add_assistant_message(
            message.content,
            metadata=metadata,
        )
        return

    if hasattr(agent_state, "add_message"):
        agent_state.add_message(
            "assistant",
            message.content,
            metadata=metadata,
        )


def add_tool_call_to_agent_state(
    agent_state: Any,
    tool_call: ToolCall,
) -> None:
    if hasattr(agent_state, "add_tool_call"):
        agent_state.add_tool_call(tool_call)


async def call_llm_node(data: AgentGraphData) -> AgentGraphData:
    """
    CallLLM node — the "thinking" step of the agent loop.

    When a real LLM is configured:
        Calls LLMRouter via call_real_llm().

    When no LLM is configured or the call fails with fallback_to_mock:
        Falls back to mock_call_llm_output() which generates deterministic
        responses for /tool shortcuts and echoes user input.

    Before calling the LLM, checks for deterministic reviewer and file-read
    intents to skip the LLM call when the user's intent is unambiguous.
    """
    agent_state = data["agent_state"]

    if hasattr(agent_state, "set_thinking"):
        agent_state.set_thinking()

    emit_status_event(
        data,
        "thinking",
        content="calling llm",
        metadata={
            "node": "call_llm",
        },
    )

    data["assistant_message"] = None
    data["llm_response"] = None
    data["llm_error"] = None

    if not data.get("awaiting_final_response"):
        coordinator_tool_call = detect_coordinator_parallel_tool_call(data)

        if coordinator_tool_call is not None:
            data["llm_output"] = json.dumps(
                {
                    "tool_name": coordinator_tool_call.tool_name,
                    "arguments": coordinator_tool_call.arguments,
                },
                ensure_ascii=False,
            )

            emit_status_event(
                data,
                "tool_route",
                content="deterministic coordinator parallel route selected",
                metadata={
                    "node": "call_llm",
                    "tool_name": coordinator_tool_call.tool_name,
                    "arguments": coordinator_tool_call.arguments,
                },
            )

            return data

        reviewer_tool_call = detect_reviewer_subagent_tool_call(data)

        if reviewer_tool_call is not None:
            data["llm_output"] = json.dumps(
                {
                    "tool_name": reviewer_tool_call.tool_name,
                    "arguments": reviewer_tool_call.arguments,
                },
                ensure_ascii=False,
            )

            emit_status_event(
                data,
                "tool_route",
                content="deterministic reviewer route selected",
                metadata={
                    "node": "call_llm",
                    "tool_name": reviewer_tool_call.tool_name,
                    "arguments": reviewer_tool_call.arguments,
                },
            )

            return data

        file_tool_call = detect_next_file_tool_call(data)

        if file_tool_call is not None:
            data["llm_output"] = json.dumps(
                {
                    "tool_name": file_tool_call.tool_name,
                    "arguments": file_tool_call.arguments,
                },
                ensure_ascii=False,
            )

            emit_status_event(
                data,
                "tool_route",
                content=f"deterministic {file_tool_call.tool_name} route selected",
                metadata={
                    "node": "call_llm",
                    "tool_name": file_tool_call.tool_name,
                    "arguments": file_tool_call.arguments,
                },
            )

            return data

    if hasattr(agent_state, "next_iteration"):
        agent_state.next_iteration()

    if should_use_real_llm(data):
        try:
            assistant_message = await call_real_llm(data)

            data["assistant_message"] = assistant_message
            data["llm_output"] = assistant_message.content

            emit_status_event(
                data,
                "llm_response",
                content="llm response received",
                metadata={
                    "node": "call_llm",
                    "has_tool_calls": bool(assistant_message.tool_calls),
                    "tool_call_count": len(assistant_message.tool_calls),
                },
            )

            return data

        except Exception as exc:
            data["llm_error"] = str(exc)

            emit_error_event(
                data,
                str(exc),
                error_type=type(exc).__name__,
                metadata={
                    "node": "call_llm",
                },
            )

            fallback_to_mock = bool(
                get_nested_value(
                    data.get("config") or {},
                    "llm.fallback_to_mock",
                    True,
                )
            )

            if not fallback_to_mock:
                if hasattr(agent_state, "set_error"):
                    agent_state.set_error(str(exc))

                data["error"] = str(exc)
                data["llm_output"] = ""
                return data

    llm_output = mock_call_llm_output(data)

    data["llm_output"] = llm_output
    data["assistant_message"] = None

    emit_status_event(
        data,
        "mock_llm_response",
        content="using mock llm output",
        metadata={
            "node": "call_llm",
        },
    )

    return data

def try_load_tool_call_json(text: str) -> dict[str, Any] | None:
    stripped = text.strip()

    if not stripped:
        return None

    if stripped.startswith("```"):
        stripped = stripped.strip("`").strip()

        if stripped.startswith("json"):
            stripped = stripped[len("json") :].strip()

    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        return None

    if not isinstance(loaded, dict):
        return None

    if "tool_name" not in loaded:
        return None

    return loaded


def build_tool_call_from_parsed(
    parsed: dict[str, Any],
    registry: ToolRegistry,
) -> ToolCall:
    tool_name = str(parsed["tool_name"]).strip()
    arguments = parsed.get("arguments", {})

    if not isinstance(arguments, dict):
        arguments = {
            "input": arguments,
        }

    tool = registry.get(tool_name)

    if tool is not None:
        return tool.create_call(
            arguments,
            metadata={
                "source": "llm_output",
            },
        )

    return ToolCall(
        tool_name=tool_name,
        arguments=arguments,
        risk_level=ToolRiskLevel.MEDIUM,
        metadata={
            "source": "llm_output",
            "unknown_tool": True,
        },
    )

def mark_graph_continue(
    data: AgentGraphData,
    *,
    reason: str = "",
) -> AgentGraphData:
    data["graph_route"] = "continue"
    data["should_continue"] = True

    if reason:
        data["route_reason"] = reason

    return data


def mark_graph_stop(
    data: AgentGraphData,
    *,
    reason: str = "",
) -> AgentGraphData:
    data["graph_route"] = "stop"
    data["should_continue"] = False

    if reason:
        data["route_reason"] = reason

    return data


def normalize_tool_call_list(value: Any) -> list[ToolCall]:
    """
    Normalize various tool call representations into a list[ToolCall].

    Handles:
    - list[ToolCall] / tuple[ToolCall]
    - single ToolCall instance
    - None
    """
    if value is None:
        return []

    if isinstance(value, ToolCall):
        return [value]

    if isinstance(value, list | tuple):
        result: list[ToolCall] = []

        for item in value:
            if isinstance(item, ToolCall):
                result.append(item)

        return result

    return []


def extract_tool_calls_from_assistant_message(
    assistant_message: AssistantMessage | None,
) -> list[ToolCall]:
    """
    Extract tool_calls from an AssistantMessage.
    """
    if assistant_message is None:
        return []

    return normalize_tool_call_list(
        getattr(assistant_message, "tool_calls", None)
    )


def extract_tool_calls_from_llm_response(
    llm_response: Any,
) -> list[ToolCall]:
    """
    Extract tool_calls from an LLMResponse, handling multiple SDK formats.

    The providers.py LLMResponse may carry tool_calls from different SDKs
    (OpenAI / Anthropic). This function normalizes them into a list[ToolCall].
    Checks LLMResponse.tool_calls first, then falls back to
    LLMResponse.message.tool_calls if the message is an AssistantMessage.
    """
    if llm_response is None:
        return []

    direct_tool_calls = normalize_tool_call_list(
        getattr(llm_response, "tool_calls", None)
    )

    if direct_tool_calls:
        return direct_tool_calls

    message = getattr(llm_response, "message", None)

    if isinstance(message, AssistantMessage):
        return extract_tool_calls_from_assistant_message(message)

    return []


def parse_mock_tool_call_from_text(text: str) -> ToolCall | None:
    """
    Parse a tool call from mock LLM output (JSON text).

    The mock LLM outputs JSON with a "tool_name" field. Also supports
    the alternate format with "name" instead of "tool_name":
        {"tool_name": "grep", "arguments": {...}}
        {"name": "grep", "arguments": {...}}
    """
    raw = text.strip()

    if not raw:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None

    tool_name = payload.get("tool_name") or payload.get("name")

    if not tool_name:
        return None

    arguments = payload.get("arguments", {}) or {}

    if not isinstance(arguments, dict):
        arguments = {
            "input": arguments,
        }

    return create_tool_call(
        tool_name=str(tool_name),
        arguments=arguments,
    )


def set_parsed_tool_calls(
    data: AgentGraphData,
    tool_calls: list[ToolCall],
    *,
    source: str,
) -> AgentGraphData:
    """
    Store parsed tool calls into GraphData and emit events.

    Sets data["parsed_tool_calls"], data["tool_call"] (first call),
    data["remaining_tool_calls"] (calls after the first), and
    data["has_tool_call"]. Each tool call is added to AgentState and
    emitted via emit_tool_call_event_once (deduplicated). If tool_calls
    is empty, marks the graph as "stop" since there is nothing to execute.
    """
    agent_state = data["agent_state"]

    data["parsed_tool_calls"] = tool_calls
    data["remaining_tool_calls"] = tool_calls[1:] if len(tool_calls) > 1 else []

    if not tool_calls:
        data["tool_call"] = None
        data["parsed_tool_call"] = None
        data["has_tool_call"] = False

        return mark_graph_stop(
            data,
            reason=f"{source}_no_tool_call",
        )

    first_tool_call = tool_calls[0]

    data["tool_call"] = first_tool_call
    data["parsed_tool_call"] = first_tool_call
    data["has_tool_call"] = True

    for call in tool_calls:
        add_tool_call_to_agent_state(
            agent_state,
            call,
        )

        emit_tool_call_event_once(
            data,
            call,
            metadata={
                "node": "parse_tool_call",
                "source": source,
            },
        )

    emit_status_event(
        data,
        "tool_call_parsed",
        content=f"parsed {len(tool_calls)} tool call(s)",
        metadata={
            "node": "parse_tool_call",
            "source": source,
            "tool_count": len(tool_calls),
        },
    )

    return mark_graph_continue(
        data,
        reason=f"{source}_has_tool_call",
    )

def parse_tool_call_node(data: AgentGraphData) -> AgentGraphData:
    """
    ParseToolCall node -- extract tool calls from the LLM response.

    Examines three sources in order:
    1. LLMResponse.tool_calls (native provider SDK tool calls)
    2. AssistantMessage.tool_calls (message-level tool calls)
    3. Mock LLM output JSON (parsed from llm_output text)

    Routing outcomes:
    - Has tool calls -> routes to PermissionCheck then ExecuteTool
    - No tool calls, has assistant message -> marks graph finished, stops
    - No tool calls, mock text -> renders as plain assistant response, stops
    """
    agent_state = data["agent_state"]

    llm_response = data.get("llm_response")
    assistant_message = data.get("assistant_message")

    # 1. Try extracting tool_calls from the native LLMResponse
    response_tool_calls = extract_tool_calls_from_llm_response(llm_response)

    if response_tool_calls:
        if isinstance(assistant_message, AssistantMessage):
            add_assistant_message_to_agent_state(
                agent_state,
                assistant_message,
            )

        return set_parsed_tool_calls(
            data,
            response_tool_calls,
            source="llm_response",
        )

    # 2. Try extracting tool_calls from the AssistantMessage
    if isinstance(assistant_message, AssistantMessage):
        add_assistant_message_to_agent_state(
            agent_state,
            assistant_message,
        )

        assistant_tool_calls = extract_tool_calls_from_assistant_message(
            assistant_message
        )

        if assistant_tool_calls:
            return set_parsed_tool_calls(
                data,
                assistant_tool_calls,
                source="assistant_message",
            )

        data["tool_call"] = None
        data["parsed_tool_call"] = None
        data["parsed_tool_calls"] = []
        data["remaining_tool_calls"] = []
        data["has_tool_call"] = False
        data["awaiting_final_response"] = False

        if hasattr(agent_state, "set_finished"):
            agent_state.set_finished()

        emit_message_event(
            data,
            "assistant",
            assistant_message.content,
            metadata={
                "node": "parse_tool_call",
                "source": "no_tool_call",
            },
        )

        emit_checkpoint_event(
            data,
            metadata={
                "node": "parse_tool_call",
            },
        )

        emit_lifecycle_event(
            data,
            RuntimeLifecycleEvent.FINISHED,
            content="runtime graph finished",
            metadata={
                "node": "parse_tool_call",
            },
        )

        return mark_graph_stop(
            data,
            reason="assistant_message_no_tool_call",
        )

    # 3. Try parsing mock JSON tool call from llm_output text
    llm_output = str(data.get("llm_output", "") or "").strip()

    if not llm_output:
        data["tool_call"] = None
        data["parsed_tool_call"] = None
        data["parsed_tool_calls"] = []
        data["remaining_tool_calls"] = []
        data["has_tool_call"] = False

        if hasattr(agent_state, "set_finished"):
            agent_state.set_finished()

        return mark_graph_stop(
            data,
            reason="empty_llm_output",
        )

    mock_tool_call = parse_mock_tool_call_from_text(llm_output)

    if mock_tool_call is not None:
        return set_parsed_tool_calls(
            data,
            [mock_tool_call],
            source="mock_json",
        )

    # 4. No tool calls found -- treat as plain text assistant response
    message = create_assistant_message(
        llm_output,
        metadata={
            "source": "mock_llm_output",
        },
    )

    add_assistant_message_to_agent_state(
        agent_state,
        message,
    )

    data["tool_call"] = None
    data["parsed_tool_call"] = None
    data["parsed_tool_calls"] = []
    data["remaining_tool_calls"] = []
    data["has_tool_call"] = False
    data["awaiting_final_response"] = False

    if hasattr(agent_state, "set_finished"):
        agent_state.set_finished()

    emit_message_event(
        data,
        "assistant",
        message.content,
        metadata={
            "node": "parse_tool_call",
            "source": "no_tool_call",
        },
    )

    emit_checkpoint_event(
        data,
        metadata={
            "node": "parse_tool_call",
        },
    )

    emit_lifecycle_event(
        data,
        RuntimeLifecycleEvent.FINISHED,
        content="runtime graph finished",
        metadata={
            "node": "parse_tool_call",
        },
    )

    return mark_graph_stop(
        data,
        reason="mock_text_response_no_tool_call",
    )
def risk_value(risk_level: ToolRiskLevel | str) -> int:
    risk = ToolRiskLevel(risk_level)

    order = {
        ToolRiskLevel.SAFE: 0,
        ToolRiskLevel.LOW: 1,
        ToolRiskLevel.MEDIUM: 2,
        ToolRiskLevel.HIGH: 3,
        ToolRiskLevel.DANGEROUS: 4,
    }

    return order[risk]


def max_allowed_risk_for_permission_mode(mode: str) -> ToolRiskLevel:
    normalized_mode = normalize_permission_mode(mode)

    if normalized_mode == PERMISSION_MODE_BYPASS:
        return ToolRiskLevel.DANGEROUS

    if normalized_mode == PERMISSION_MODE_ACCEPT_EDITS:
        return ToolRiskLevel.MEDIUM

    if normalized_mode == PERMISSION_MODE_PLAN:
        return ToolRiskLevel.SAFE

    if normalized_mode == PERMISSION_MODE_READONLY:
        return ToolRiskLevel.SAFE

    return ToolRiskLevel.LOW


def evaluate_permission(
    call: ToolCall | None,
    *,
    registry: ToolRegistry,
    permission_mode: str,
) -> PermissionDecision:
    """
    PermissionCheck -- evaluate whether a tool call is allowed.

    Permission mode determines max allowed risk level:
    - plan mode: no tools execute at all
    - readonly: only safe tools execute
    - default: safe/low risk tools allowed
    - accept_edits: safe/low/medium risk tools allowed
    - bypass_permissions: all tools allowed
    """
    permission_mode = normalize_permission_mode(permission_mode)

    if call is None:
        return PermissionDecision(
            allowed=True,
            reason="no tool call",
            requires_confirmation=False,
        )

    if permission_mode == PERMISSION_MODE_PLAN:
        return PermissionDecision(
            allowed=False,
            reason=f"plan mode does not execute tools: {call.tool_name}",
            requires_confirmation=True,
        )

    tool = registry.get(call.tool_name)

    if tool is None:
        return PermissionDecision(
            allowed=False,
            reason=f"tool not registered: {call.tool_name}",
            requires_confirmation=False,
        )

    call_risk = tool.get_risk_level()
    max_risk = max_allowed_risk_for_permission_mode(permission_mode)

    if risk_value(call_risk) <= risk_value(max_risk):
        return PermissionDecision(
            allowed=True,
            reason=f"allowed by permission mode: {permission_mode}",
            requires_confirmation=False,
        )

    return PermissionDecision(
        allowed=False,
        reason=(
            f"tool {call.tool_name!r} risk {call_risk.value!r} exceeds "
            f"permission mode {permission_mode!r}"
        ),
        requires_confirmation=True,
    )


def get_permission_gate_enabled(data: AgentGraphData) -> bool:
    config = get_config(data)

    return bool(
        get_nested_config_value(
            config,
            "permissions.enabled",
            True,
        )
    )


def get_permission_audit_enabled(data: AgentGraphData) -> bool:
    config = get_config(data)

    return bool(
        get_nested_config_value(
            config,
            "permissions.audit_enabled",
            True,
        )
    )


def create_graph_permission_gate(data: AgentGraphData) -> PermissionGate:
    return PermissionGate(
        workspace_path=get_workspace_path(data),
        audit_enabled=get_permission_audit_enabled(data),
        session_id=data.get("session_id"),
        session_state=get_permission_gate_state(data),
    )


def get_permission_gate_state(data: AgentGraphData) -> PermissionGateState:
    state = data.get("permission_gate_state")

    if isinstance(state, PermissionGateState):
        return state

    state = PermissionGateState()
    data["permission_gate_state"] = state

    return state


def get_registered_tool_risk(data: AgentGraphData, tool_name: str) -> Any | None:
    tool = get_registry(data).get(tool_name)

    if tool is None or not hasattr(tool, "get_risk_level"):
        return None

    return tool.get_risk_level()


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


async def request_graph_approval(
    data: AgentGraphData,
    gate_result: PermissionGateResult,
) -> Any | None:
    """
    Call the external approval handler supplied by the TUI/controller.
    """
    handler = data.get("approval_handler")

    if handler is None:
        return None

    result = handler(gate_result)

    return await maybe_await(result)


def get_approval_user_action(approval_result: Any) -> PermissionAuditUserAction:
    if approval_result is None:
        return PermissionAuditUserAction.DENY

    user_action = getattr(
        approval_result,
        "user_action",
        None,
    )

    if user_action is not None:
        return PermissionAuditUserAction(str(getattr(user_action, "value", user_action)))

    choice = getattr(
        approval_result,
        "choice",
        None,
    )

    if choice is not None:
        value = str(getattr(choice, "value", choice))

        if value == "allow":
            return PermissionAuditUserAction.ALLOW

        if value == "always_allow":
            return PermissionAuditUserAction.ALWAYS_ALLOW

        return PermissionAuditUserAction.DENY

    allowed = bool(
        getattr(
            approval_result,
            "allowed",
            False,
        )
    )

    return (
        PermissionAuditUserAction.ALLOW
        if allowed
        else PermissionAuditUserAction.DENY
    )


def approval_result_allows(approval_result: Any) -> bool:
    return user_action_is_allow(
        get_approval_user_action(approval_result),
    )


def approval_result_is_always_allow(approval_result: Any) -> bool:
    return user_action_is_always_allow(
        get_approval_user_action(approval_result),
    )


def record_graph_permission_user_decision(
    data: AgentGraphData,
    gate_result: PermissionGateResult,
    approval_result: Any,
    *,
    reason: str | None = None,
) -> None:
    """
    Record the user's approval decision.

    When approval_result is None, record it as DENY. This covers a missing
    approval handler, a dismissed dialog, or no explicit user confirmation.
    """
    if not get_permission_audit_enabled(data):
        return

    audit_log = PermissionAuditLog(
        get_workspace_path(data),
    )

    audit_log.record_user_decision(
        gate_result.decision,
        user_action=get_approval_user_action(approval_result),
        session_id=data.get("session_id"),
        reason=reason,
        metadata={
            "node": "approval",
            "run_id": get_graph_run_id(data),
            "checkpoint_id": data["agent_state"].checkpoint_id,
            "approval_result_present": approval_result is not None,
        },
    )


def apply_approval_override_if_needed(
    data: AgentGraphData,
    gate_result: PermissionGateResult,
    approval_result: Any,
) -> None:
    if not approval_result_is_always_allow(approval_result):
        return

    state = get_permission_gate_state(data)

    state.add_always_allow(
        gate_result.decision,
        rule_result=gate_result.rule_result,
        reason="user selected Always Allow",
        metadata={
            "run_id": get_graph_run_id(data),
            "call_id": gate_result.decision.request.call_id,
        },
    )


def make_permission_blocked_tool_result(
    tool_call: ToolCall,
    gate_result: PermissionGateResult,
    *,
    user_denied: bool = False,
    blocked_reason: str = "permission_blocked",
) -> ToolResult:
    decision = gate_result.decision

    if decision.denied:
        title = "Permission denied"
    elif decision.is_elevated:
        title = "Elevated approval required"
    else:
        title = "Approval required"

    content = (
        f"{title}: tool `{tool_call.tool_name}` was not executed.\n\n"
        f"Decision: {decision.decision.value}\n"
        f"Mode: {decision.mode.value}\n"
        f"Risk: {decision.risk.value}\n"
        f"Reason: {decision.reason}\n"
    )

    if gate_result.rule_result is not None:
        content += (
            "\nRule check:\n"
            f"- Source: {gate_result.rule_result.source}\n"
            f"- Rule decision: {gate_result.rule_result.decision.value}\n"
            f"- Matched rules: {', '.join(gate_result.rule_result.matched_rules) or '(none)'}\n"
        )

    rule_result = gate_result.rule_result
    metadata = {
        "permission_blocked": True,
        "blocked_reason": blocked_reason,
        "user_denied": user_denied,
        "permission_decision": decision.decision.value,
        "permission_mode": decision.mode.value,
        "permission_risk": decision.risk.value,
        "permission_reason": decision.reason,
    }

    data = {
        "permission_blocked": True,
        "blocked_reason": blocked_reason,
        "user_denied": user_denied,
        "decision": decision.decision.value,
        "mode": decision.mode.value,
        "risk": decision.risk.value,
        "reason": decision.reason,
    }

    if rule_result is not None:
        metadata["rule_source"] = rule_result.source
        metadata["matched_rules"] = list(rule_result.matched_rules)
        data["rule_check"] = {
            "source": rule_result.source,
            "decision": rule_result.decision.value,
            "matched_rules": list(rule_result.matched_rules),
        }

    return ToolResult.error_result(
        call=tool_call,
        error=content,
        data=data,
        metadata=metadata,
    )


def extract_tool_exit_code(result: ToolResult | None) -> int | None:
    if result is None:
        return None

    data = getattr(result, "data", None)

    if not isinstance(data, dict):
        return None

    exit_code = data.get("exit_code")

    if exit_code is None:
        shell_result = data.get("shell_result")

        if isinstance(shell_result, dict):
            exit_code = shell_result.get("exit_code")

    if exit_code is None:
        return None

    try:
        return int(exit_code)
    except (TypeError, ValueError):
        return None


def record_graph_permission_execution_result(
    data: AgentGraphData,
    gate_result: PermissionGateResult | None,
    *,
    executed: bool,
    result: ToolResult | None = None,
    reason: str | None = None,
) -> None:
    """
    Record whether the tool ultimately executed.
    """
    if gate_result is None:
        return

    if not get_permission_audit_enabled(data):
        return

    audit_log = PermissionAuditLog(
        get_workspace_path(data),
    )

    audit_log.record_execution_result(
        gate_result.decision,
        executed=executed,
        success=(result.success if result is not None else False),
        exit_code=extract_tool_exit_code(result),
        session_id=data.get("session_id"),
        reason=reason,
        metadata={
            "node": "execute_tool",
            "run_id": get_graph_run_id(data),
            "checkpoint_id": data["agent_state"].checkpoint_id,
            "tool_result_present": result is not None,
        },
    )


def permission_check_node(data: AgentGraphData) -> AgentGraphData:
    """
    PermissionCheck node -- evaluate tool call against permission policy.

    Uses the PermissionGate to check the current tool call against
    the configured permission_mode and risk_level. Emits status events
    with the permission decision for TUI display.
    """
    state = data["agent_state"]
    call = data.get("parsed_tool_call")

    data["permission_gate_result"] = None
    data["permission_gate_error"] = None
    data["permission_decision"] = None

    if call is None:
        return data

    if not get_permission_gate_enabled(data):
        return data

    try:
        gate = create_graph_permission_gate(data)

        gate_result = gate.check(
            call,
            mode=get_permission_mode(data),
            risk=get_registered_tool_risk(data, call.tool_name),
            session_id=data.get("session_id"),
            metadata={
                "node": "permission_check",
                "run_id": get_graph_run_id(data),
                "checkpoint_id": state.checkpoint_id,
            },
        )

        data["permission_gate_result"] = gate_result
        data["permission_decision"] = gate_result.decision

        emit_status_event(
            data,
            "permission_checked",
            content=render_permission_gate_result(gate_result),
            metadata={
                "node": "permission_check",
                "tool_name": call.tool_name,
                "call_id": call.call_id,
                "decision": gate_result.decision.decision.value,
                "risk": gate_result.decision.risk.value,
                "allowed": gate_result.allowed,
            },
        )

        if not gate_result.allowed:
            if hasattr(state, "set_waiting_permission"):
                state.set_waiting_permission(call.call_id)

            data["agent_state"] = state

        return data

    except Exception as exc:
        error_text = str(exc)

        data["permission_gate_error"] = error_text
        data["error"] = error_text

        emit_error_event(
            data,
            error_text,
            error_type=type(exc).__name__,
            metadata={
                "node": "permission_check",
                "tool_name": call.tool_name,
                "call_id": call.call_id,
            },
        )

        if hasattr(state, "set_error"):
            state.set_error(error_text)

        data["agent_state"] = state

        return data


def create_graph_tool_context(data: AgentGraphData) -> ToolExecutionContext:
    """
    Create the ToolExecutionContext used by registry-backed graph tools.

    这里会把 Runtime 共享对象注入给工具：
    - task_manager
    - subagent_manager
    - mailbox
    - team / team_registry
    - tool_registry / registry
    - agent_state
    - run_id / session_id
    """
    signature = inspect.signature(ToolExecutionContext)
    parameters = signature.parameters

    workspace_path = Path(get_workspace_path(data)).expanduser().resolve()
    project_root = Path(get_project_root(data)).expanduser().resolve()

    registry = get_registry(data)
    config = get_config(data)

    runtime_metadata = dict(data.get("metadata") or {})

    runtime_metadata.update(
        {
            "source": "runtime_graph.execute_tool_node",
            "permission_mode": get_permission_mode(data),
            "checkpoint_id": data["agent_state"].checkpoint_id,
            "run_id": get_graph_run_id(data),
            "session_id": data.get("session_id"),
            "agent_state": data["agent_state"],
            "tool_registry": registry,
            "registry": registry,
            "tool_definitions": get_graph_tool_definitions(data),
            "event_bus": get_graph_event_bus(data),
            "config": config,
        }
    )

    kwargs: dict[str, Any] = {}

    if "workspace_path" in parameters:
        kwargs["workspace_path"] = str(workspace_path)

    if "project_root" in parameters:
        kwargs["project_root"] = str(project_root)

    if "permission_mode" in parameters:
        kwargs["permission_mode"] = get_permission_mode(data)

    if "session_id" in parameters:
        session_id = data.get("session_id")

        if session_id is not None:
            kwargs["session_id"] = str(session_id)

    if "metadata" in parameters:
        kwargs["metadata"] = runtime_metadata

    return ToolExecutionContext(**kwargs)


def get_current_tool_call(data: AgentGraphData) -> ToolCall | None:
    tool_call = data.get("tool_call") or data.get("parsed_tool_call")

    if isinstance(tool_call, ToolCall):
        return tool_call

    return None


async def run_tool_from_registry(
    data: AgentGraphData,
    tool_call: ToolCall,
) -> ToolResult:
    """
    Execute a ToolCall through the configured ToolRegistry.
    """
    registry = get_registry(data)
    context = create_graph_tool_context(data)

    if hasattr(registry, "execute_call"):
        return await registry.execute_call(
            tool_call,
            context=context,
        )

    if hasattr(registry, "run_tool"):
        try:
            return await registry.run_tool(
                tool_call.tool_name,
                tool_call.arguments,
                context=context,
                metadata=tool_call.metadata,
            )
        except TypeError:
            return await registry.run_tool(
                tool_call.tool_name,
                tool_call.arguments,
                context,
            )

    tool = registry.require(tool_call.tool_name)

    return await tool.run(
        tool_call,
        context,
    )


def add_tool_result_to_agent_state(
    agent_state,
    result: ToolResult,
) -> Any:
    """
    把 ToolResult 写回 AgentState.messages。

    这一步很关键：
    如果不把 stdout / stderr / exit_code 等内容作为 tool message
    塞回 AgentState，LLM 下一轮就不知道工具执行结果。
    """
    return append_tool_result_to_agent_state(
        agent_state,
        result,
    )


def make_tool_error_result(
    tool_call: ToolCall,
    error: str,
) -> ToolResult:
    """
    Create a ToolResult for graph-level tool execution errors.
    """
    try:
        return ToolResult.error_result(
            call=tool_call,
            error=error,
        )
    except TypeError:
        return ToolResult.error_result(
            call=tool_call,
            content=error,
        )


def get_graph_run_id(data: AgentGraphData) -> str:
    run_id = data.get("run_id")

    if not run_id:
        run_id = new_run_id()
        data["run_id"] = run_id

    return run_id


def get_graph_event_bus(data: AgentGraphData) -> RuntimeEventBus:
    bus = data.get("event_bus")

    if bus is None:
        bus = get_default_event_bus()
        data["event_bus"] = bus

    return bus


def should_emit_runtime_events(data: AgentGraphData) -> bool:
    return bool(data.get("emit_events", True))


def emit_graph_event(
    data: AgentGraphData,
    event: RuntimeEvent,
) -> RuntimeEvent:
    """
    Runtime Graph unified event outlet.
    """
    if not should_emit_runtime_events(data):
        return event

    bus = get_graph_event_bus(data)
    bus.emit(event)

    events = data.setdefault("runtime_events", [])
    events.append(event)

    return event


def emit_lifecycle_event(
    data: AgentGraphData,
    lifecycle: RuntimeLifecycleEvent | str,
    *,
    content: str = "",
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_graph_event(
        data,
        RuntimeEvent.lifecycle_event(
            lifecycle=lifecycle,
            content=content,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_status_event(
    data: AgentGraphData,
    status: str,
    *,
    content: str = "",
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_graph_event(
        data,
        RuntimeEvent.status_event(
            status=status,
            content=content,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_message_event(
    data: AgentGraphData,
    role: str,
    content: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_graph_event(
        data,
        RuntimeEvent.message(
            role=role,  # type: ignore[arg-type]
            content=content,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_tool_call_event_once(
    data: AgentGraphData,
    tool_call: ToolCall,
    *,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent | None:
    emitted = data.setdefault("emitted_tool_call_ids", set())

    if tool_call.call_id in emitted:
        return None

    emitted.add(tool_call.call_id)

    return emit_graph_event(
        data,
        RuntimeEvent.tool_call_event(
            tool_call=tool_call,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_tool_result_event(
    data: AgentGraphData,
    tool_result: ToolResult,
    *,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_graph_event(
        data,
        RuntimeEvent.tool_result_event(
            tool_result=tool_result,
            source=RuntimeEventSource.TOOL,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_error_event(
    data: AgentGraphData,
    error: str,
    *,
    error_type: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    return emit_graph_event(
        data,
        RuntimeEvent.error_event(
            error=error,
            error_type=error_type,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            checkpoint_id=data["agent_state"].checkpoint_id,
            metadata=metadata or {},
        ),
    )


def emit_checkpoint_event(
    data: AgentGraphData,
    *,
    metadata: dict[str, Any] | None = None,
) -> RuntimeEvent:
    state = data["agent_state"]

    return emit_graph_event(
        data,
        RuntimeEvent.checkpoint_event(
            checkpoint_id=state.checkpoint_id,
            source=RuntimeEventSource.GRAPH,
            run_id=get_graph_run_id(data),
            session_id=data.get("session_id"),
            data={
                "iteration": state.iteration,
                "status": state.status.value
                if hasattr(state.status, "value")
                else str(state.status),
                "message_count": len(getattr(state, "messages", [])),
                "tool_call_count": len(getattr(state, "tool_calls", [])),
                "tool_result_count": len(getattr(state, "tool_results", [])),
            },
            metadata=metadata or {},
        ),
    )


async def execute_tool_node(data: AgentGraphData) -> dict[str, Any]:
    """
    ExecuteTool node.
    Execute the current parsed ToolCall.
    """
    agent_state = data["agent_state"]
    tool_call = get_current_tool_call(data)

    if tool_call is None:
        if not data.get("has_tool_call") and data.get("graph_route") == "stop":
            return data

        data["tool_result"] = None
        data["has_tool_result"] = False
        data["error"] = "no tool_call to execute"

        emit_error_event(
            data,
            "no tool_call to execute",
            error_type="MissingToolCall",
            metadata={
                "node": "execute_tool",
            },
        )

        if hasattr(agent_state, "set_error"):
            agent_state.set_error("no tool_call to execute")

        return data

    data["current_tool_call_id"] = tool_call.call_id

    emit_tool_call_event_once(
        data,
        tool_call,
        metadata={
            "node": "execute_tool",
        },
    )

    gate_result = data.get("permission_gate_result")

    if isinstance(gate_result, PermissionGateResult) and not gate_result.allowed:
        if gate_result.denied:
            result = make_permission_blocked_tool_result(
                tool_call,
                gate_result,
                blocked_reason="permission_blocked",
            )

            data["tool_result"] = result
            data["has_tool_result"] = True
            data["error"] = gate_result.decision.reason

            add_tool_result_to_agent_state(
                agent_state,
                result,
            )

            record_graph_permission_execution_result(
                data,
                gate_result,
                executed=False,
                result=result,
                reason=gate_result.decision.reason,
            )

            emit_tool_result_event(
                data,
                result,
                metadata={
                    "node": "execute_tool",
                    "tool_name": tool_call.tool_name,
                    "success": False,
                    "permission_blocked": True,
                    "permission_decision": gate_result.decision.decision.value,
                    "risk": gate_result.decision.risk.value,
                },
            )

            emit_status_event(
                data,
                "permission_blocked",
                content=render_permission_gate_result(gate_result),
                metadata={
                    "node": "execute_tool",
                    "tool_name": tool_call.tool_name,
                    "call_id": tool_call.call_id,
                    "decision": gate_result.decision.decision.value,
                    "risk": gate_result.decision.risk.value,
                },
            )

            if hasattr(agent_state, "current_tool_call_id"):
                agent_state.current_tool_call_id = None

            data["agent_state"] = agent_state

            return data

        approval_result = await request_graph_approval(
            data,
            gate_result,
        )

        data["approval_result"] = approval_result

        record_graph_permission_user_decision(
            data,
            gate_result,
            approval_result,
            reason=(
                "user responded to approval request"
                if approval_result is not None
                else "approval unavailable or dismissed"
            ),
        )

        if approval_result is None or not approval_result_allows(approval_result):
            result = make_permission_blocked_tool_result(
                tool_call,
                gate_result,
                user_denied=approval_result is not None,
                blocked_reason=(
                    "user_denied"
                    if approval_result is not None
                    else "approval_unavailable"
                ),
            )

            data["tool_result"] = result
            data["has_tool_result"] = True
            data["error"] = "user denied permission"

            add_tool_result_to_agent_state(
                agent_state,
                result,
            )

            record_graph_permission_execution_result(
                data,
                gate_result,
                executed=False,
                result=result,
                reason="user denied permission",
            )

            emit_tool_result_event(
                data,
                result,
                metadata={
                    "node": "execute_tool",
                    "tool_name": tool_call.tool_name,
                    "success": False,
                    "permission_blocked": True,
                    "permission_decision": gate_result.decision.decision.value,
                    "risk": gate_result.decision.risk.value,
                    "user_denied": True,
                },
            )

            emit_status_event(
                data,
                "permission_denied_by_user",
                content=f"user denied tool `{tool_call.tool_name}`",
                metadata={
                    "node": "execute_tool",
                    "tool_name": tool_call.tool_name,
                    "call_id": tool_call.call_id,
                    "decision": gate_result.decision.decision.value,
                    "risk": gate_result.decision.risk.value,
                },
            )

            if hasattr(agent_state, "current_tool_call_id"):
                agent_state.current_tool_call_id = None

            data["agent_state"] = agent_state

            return data

        apply_approval_override_if_needed(
            data,
            gate_result,
            approval_result,
        )

        emit_status_event(
            data,
            "permission_approved_by_user",
            content=f"user approved tool `{tool_call.tool_name}`",
            metadata={
                "node": "execute_tool",
                "tool_name": tool_call.tool_name,
                "call_id": tool_call.call_id,
                "decision": gate_result.decision.decision.value,
                "risk": gate_result.decision.risk.value,
                "always_allow": approval_result_is_always_allow(approval_result),
            },
        )

    emit_status_event(
        data,
        "running_tool",
        content=f"running tool {tool_call.tool_name}",
        metadata={
            "node": "execute_tool",
            "tool_name": tool_call.tool_name,
            "call_id": tool_call.call_id,
        },
    )

    if hasattr(agent_state, "set_running_tool"):
        agent_state.set_running_tool(tool_call.call_id)

    try:
        result = await run_tool_from_registry(
            data,
            tool_call,
        )

        data["tool_result"] = result
        data["has_tool_result"] = True
        data["error"] = None

        add_tool_result_to_agent_state(
            agent_state,
            result,
        )

        emit_tool_result_event(
            data,
            result,
            metadata={
                "node": "execute_tool",
                "tool_name": result.tool_name,
                "success": result.success,
            },
        )

        emit_status_event(
            data,
            "tool_finished",
            content=f"tool {result.tool_name} finished",
            metadata={
                "node": "execute_tool",
                "tool_name": result.tool_name,
                "success": result.success,
            },
        )

        record_graph_permission_execution_result(
            data,
            data.get("permission_gate_result"),
            executed=True,
            result=result,
            reason="tool executed",
        )

        if hasattr(agent_state, "current_tool_call_id"):
            agent_state.current_tool_call_id = None

        return data

    except Exception as exc:
        error_text = str(exc)

        result = make_tool_error_result(
            tool_call,
            error_text,
        )

        data["tool_result"] = result
        data["has_tool_result"] = True
        data["error"] = error_text

        add_tool_result_to_agent_state(
            agent_state,
            result,
        )

        emit_tool_result_event(
            data,
            result,
            metadata={
                "node": "execute_tool",
                "tool_name": tool_call.tool_name,
                "success": False,
            },
        )

        record_graph_permission_execution_result(
            data,
            data.get("permission_gate_result"),
            executed=True,
            result=result,
            reason=error_text,
        )

        emit_error_event(
            data,
            error_text,
            error_type=type(exc).__name__,
            metadata={
                "node": "execute_tool",
                "tool_name": tool_call.tool_name,
                "call_id": tool_call.call_id,
            },
        )

        if hasattr(agent_state, "set_error"):
            agent_state.set_error(error_text)

        if hasattr(agent_state, "current_tool_call_id"):
            agent_state.current_tool_call_id = None

        return data


def append_observation_node(data: AgentGraphData) -> AgentGraphData:
    """Append a tool observation and route back to the LLM for final answer."""
    agent_state = data["agent_state"]
    result = data.get("tool_result")

    if not data.get("has_tool_call") and data.get("graph_route") == "stop":
        return data

    if not isinstance(result, ToolResult):
        if hasattr(agent_state, "set_error"):
            agent_state.set_error("missing tool_result")

        data["error"] = "missing tool_result"

        emit_error_event(
            data,
            "missing tool_result",
            error_type="MissingToolResult",
            metadata={
                "node": "append_observation",
            },
        )

        return mark_graph_stop(
            data,
            reason="missing_tool_result",
        )

    data["tool_call"] = None
    data["parsed_tool_call"] = None
    data["parsed_tool_calls"] = []
    data["remaining_tool_calls"] = []
    data["has_tool_call"] = False
    data["current_tool_call_id"] = None
    data["tool_result"] = result
    data["llm_output"] = ""

    emit_checkpoint_event(
        data,
        metadata={
            "node": "append_observation",
            "tool_name": result.tool_name,
            "success": result.success,
        },
    )

    if result.success:
        if (
            result.tool_name == "glob"
            and READ_FILE_INTENT_PATTERN.search(str(data.get("user_input", "") or ""))
        ):
            paths = extract_glob_file_read_paths(result)

            if paths:
                data["pending_file_read_paths"] = paths
                data["completed_file_read_paths"] = []
                data["file_read_batch_active"] = True

                observation = (
                    "Glob matched files to read:\n\n"
                    + "\n".join(f"- {path}" for path in paths)
                    + "\n\nI will read these files next before answering."
                )

                if hasattr(agent_state, "add_user_message"):
                    agent_state.add_user_message(
                        observation,
                        metadata={
                            "source": "glob_file_read_queue",
                            "tool_name": result.tool_name,
                            "paths": paths,
                        },
                    )
                elif hasattr(agent_state, "add_message"):
                    agent_state.add_message(
                        "user",
                        observation,
                        metadata={
                            "source": "glob_file_read_queue",
                            "tool_name": result.tool_name,
                            "paths": paths,
                        },
                    )

                emit_status_event(
                    data,
                    "file_read_batch_queued",
                    content=f"queued {len(paths)} file(s) for reading",
                    metadata={
                        "node": "append_observation",
                        "tool_name": result.tool_name,
                        "paths": paths,
                    },
                )

                if hasattr(agent_state, "set_idle"):
                    agent_state.set_idle()

                return mark_graph_continue(
                    data,
                    reason="glob_result_queued_file_reads",
                )

        if should_finish_after_tool_result(result):
            message = build_direct_tool_finish_message(result)

            if hasattr(agent_state, "add_assistant_message"):
                agent_state.add_assistant_message(
                    message,
                    metadata={
                        "source": "tool_result_direct_finish",
                        "tool_name": result.tool_name,
                        "call_id": result.call_id,
                    },
                )
            elif hasattr(agent_state, "add_message"):
                agent_state.add_message(
                    "assistant",
                    message,
                    metadata={
                        "source": "tool_result_direct_finish",
                        "tool_name": result.tool_name,
                        "call_id": result.call_id,
                    },
                )

            emit_status_event(
                data,
                "tool_result_finished",
                content="tool result completed without final llm",
                metadata={
                    "node": "append_observation",
                    "tool_name": result.tool_name,
                },
            )

            data["awaiting_final_response"] = False
            data["file_read_batch_active"] = False

            if hasattr(agent_state, "set_finished"):
                agent_state.set_finished()

            return mark_graph_stop(
                data,
                reason="tool_result_direct_finish",
            )

        observation = (
            f"Tool `{result.tool_name}` finished successfully.\n\n"
            f"Tool result:\n\n{result.content}\n\n"
            "Now answer the original user request using this tool result. "
            "Do not call another tool unless the result is insufficient."
        )

        if hasattr(agent_state, "add_user_message"):
            agent_state.add_user_message(
                observation,
                metadata={
                    "source": "tool_observation",
                    "tool_name": result.tool_name,
                    "call_id": result.call_id,
                },
            )
        elif hasattr(agent_state, "add_message"):
            agent_state.add_message(
                "user",
                observation,
                metadata={
                    "source": "tool_observation",
                    "tool_name": result.tool_name,
                    "call_id": result.call_id,
                },
            )

        emit_status_event(
            data,
            "tool_result_observed",
            content="tool result appended; requesting final answer",
            metadata={
                "node": "append_observation",
                "tool_name": result.tool_name,
            },
        )

        if hasattr(agent_state, "set_idle"):
            agent_state.set_idle()

        pending_paths = list(data.get("pending_file_read_paths") or [])

        if pending_paths:
            data["awaiting_final_response"] = False

            return mark_graph_continue(
                data,
                reason="file_read_batch_continue",
            )

        data["awaiting_final_response"] = True
        data["file_read_batch_active"] = False

        return mark_graph_continue(
            data,
            reason="tool_result_observed_continue_to_llm",
        )

    if is_permission_blocked_tool_result(result):
        message = build_permission_blocked_finish_message(result)

        if hasattr(agent_state, "add_assistant_message"):
            agent_state.add_assistant_message(
                message,
                metadata={
                    "source": "permission_blocked_direct_finish",
                    "tool_name": result.tool_name,
                    "call_id": result.call_id,
                },
            )
        elif hasattr(agent_state, "add_message"):
            agent_state.add_message(
                "assistant",
                message,
                metadata={
                    "source": "permission_blocked_direct_finish",
                    "tool_name": result.tool_name,
                    "call_id": result.call_id,
                },
            )

        emit_status_event(
            data,
            "tool_result_finished",
            content="permission blocked tool result completed",
            metadata={
                "node": "append_observation",
                "tool_name": result.tool_name,
                "permission_blocked": True,
                "blocked_reason": (result.metadata or {}).get("blocked_reason"),
            },
        )

        data["awaiting_final_response"] = False
        data["file_read_batch_active"] = False
        data["error"] = None

        if hasattr(agent_state, "last_error"):
            agent_state.last_error = None

        if hasattr(agent_state, "set_finished"):
            agent_state.set_finished()

        return mark_graph_stop(
            data,
            reason="permission_blocked_direct_finish",
        )

    error_text = result.error or result.content

    if hasattr(agent_state, "set_error"):
        agent_state.set_error(error_text)

    emit_error_event(
        data,
        error_text,
        error_type="ToolExecutionError",
        metadata={
            "node": "append_observation",
            "tool_name": result.tool_name,
        },
    )

    emit_lifecycle_event(
        data,
        RuntimeLifecycleEvent.ERROR,
        content="runtime graph failed",
        metadata={
            "node": "append_observation",
        },
    )

    return mark_graph_stop(
        data,
        reason="tool_result_failed",
    )

def compact_messages_if_needed(
    state: AgentState,
    *,
    max_messages: int = 40,
) -> bool:
    """
    Compact message history when it exceeds max_messages.

    Keeps the first system message and the most recent (max_messages-1)
    messages. Drops messages in the middle. Sets metadata["compacted"]=True
    to record that compaction occurred.
    """
    if len(state.messages) <= max_messages:
        return False

    system_messages = [
        message
        for message in state.messages
        if message.role == "system"
    ]

    first_system = system_messages[:1]
    keep_tail_count = max(1, max_messages - len(first_system))

    state.messages = first_system + state.messages[-keep_tail_count:]

    state.metadata["compacted"] = True
    state.metadata["compact_max_messages"] = max_messages
    state.touch()

    return True


def compact_if_needed_node(data: AgentGraphData) -> dict[str, Any]:
    """
    CompactIfNeeded node -- truncate message history before next LLM call.

    Skips compaction during active file_read batches to avoid losing
    context mid-operation. Otherwise calls compact_messages_if_needed().
    """
    state = data["agent_state"]
    config = get_config(data)

    max_messages = int(
        get_nested_config_value(
            config,
            "agent.max_context_messages",
            40,
        )
    )

    if (
        data.get("file_read_batch_active")
        or data.get("pending_file_read_paths")
        or data.get("awaiting_final_response")
    ):
        data["agent_state"] = state
        data["metadata"] = {
            **data.get("metadata", {}),
            "compacted": False,
            "compact_skipped": "file_read_batch",
        }

        return data

    compacted = compact_messages_if_needed(
        state,
        max_messages=max_messages,
    )

    data["agent_state"] = state
    data["metadata"] = {
        **data.get("metadata", {}),
        "compacted": compacted,
    }

    return data


def continue_or_stop_node(data: AgentGraphData) -> AgentGraphData:
    """
    ContinueOrStop node -- decide whether the graph loops or terminates.

    Routes based on:
    - error present -> stop
    - pending tool_call -> continue
    - no tool_call -> stop
    - max_iterations reached -> stop
    """
    agent_state = data["agent_state"]

    if data.get("awaiting_final_response") and not data.get("final_response_requested"):
        data["final_response_requested"] = True

        return mark_graph_continue(
            data,
            reason="awaiting_final_response",
        )

    if data.get("pending_file_read_paths"):
        return mark_graph_continue(
            data,
            reason="pending_file_read_paths",
        )

    if data.get("error"):
        mark_graph_stop(
            data,
            reason="graph_error",
        )

        if hasattr(agent_state, "set_error"):
            agent_state.set_error(str(data["error"]))

        return data

    if data.get("has_tool_call"):
        return mark_graph_continue(
            data,
            reason="has_tool_call",
        )

    if hasattr(agent_state, "is_max_iterations_reached"):
        if agent_state.is_max_iterations_reached():
            if hasattr(agent_state, "set_finished"):
                agent_state.set_finished()

            return mark_graph_stop(
                data,
                reason="max_iterations_reached",
            )

    if hasattr(agent_state, "set_finished"):
        agent_state.set_finished()

    return mark_graph_stop(
        data,
        reason="no_tool_call",
    )

def route_continue_or_stop(data: AgentGraphData) -> GraphRoute:
    route = str(data.get("graph_route", "stop"))

    if route == "continue":
        return "continue"

    return "stop"


def build_agent_graph():
    """
    Build and compile the LangGraph agent execution graph.
    """
    if StateGraph is None:
        raise RuntimeError(
            "langgraph is not installed or cannot be imported. "
            "Please install langgraph first."
        )

    graph = StateGraph(AgentGraphData)

    graph.add_node("user_input", user_input_node)
    graph.add_node("build_context", build_context_node)
    graph.add_node("call_llm", call_llm_node)
    graph.add_node("parse_tool_call", parse_tool_call_node)
    graph.add_node("permission_check", permission_check_node)
    graph.add_node("execute_tool", execute_tool_node)
    graph.add_node("append_observation", append_observation_node)
    graph.add_node("compact_if_needed", compact_if_needed_node)
    graph.add_node("continue_or_stop", continue_or_stop_node)

    graph.add_edge(START, "user_input")
    graph.add_edge("user_input", "build_context")
    graph.add_edge("build_context", "call_llm")
    graph.add_edge("call_llm", "parse_tool_call")
    graph.add_edge("parse_tool_call", "permission_check")
    graph.add_edge("permission_check", "execute_tool")
    graph.add_edge("execute_tool", "append_observation")
    graph.add_edge("append_observation", "compact_if_needed")
    graph.add_edge("compact_if_needed", "continue_or_stop")

    graph.add_conditional_edges(
        "continue_or_stop",
        route_continue_or_stop,
        {
            "continue": "build_context",
            "stop": END,
        },
    )

    return graph.compile()


class AgentGraphRunner:
    """
    AgentGraphRunner -- wraps the compiled LangGraph for easy invocation.

    Usage:
        runner = AgentGraphRunner(registry=..., config=...)
        state = await runner.arun("user input")
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        config: dict[str, Any] | None = None,
        llm_router: Any | None = None,
        event_bus: RuntimeEventBus | None = None,
        emit_events: bool = True,
        approval_handler: Any | None = None,
        permission_gate_state: PermissionGateState | None = None,
        runtime_objects: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.config = config or {}
        self.llm_router = llm_router
        self.event_bus = event_bus or get_default_event_bus()
        self.emit_events = emit_events
        self.approval_handler = approval_handler
        self.permission_gate_state = permission_gate_state or PermissionGateState()
        self.runtime_objects = dict(runtime_objects or {})
        self.graph = build_agent_graph()

    async def arun(
        self,
        user_input: str,
        *,
        agent_state: AgentState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        run_metadata = {
            **self.runtime_objects,
            **(metadata or {}),
        }

        initial_state = create_default_agent_graph_state(
            user_input=user_input,
            registry=self.registry,
            config=self.config,
            agent_state=agent_state,
            metadata=run_metadata,
        )

        run_id = str(run_metadata.get("run_id") or new_run_id())
        session_id = run_metadata.get("session_id")
        
        initial_state["registry"] = self.registry
        initial_state["config"] = self.config
        initial_state["llm_router"] = self.llm_router
        initial_state["tool_definitions"] = self.registry.list_definitions()
        initial_state["run_id"] = run_id
        initial_state["session_id"] = session_id
        initial_state["event_bus"] = self.event_bus
        initial_state["emit_events"] = self.emit_events
        initial_state["runtime_events"] = []
        initial_state["emitted_tool_call_ids"] = set()
        initial_state["approval_handler"] = self.approval_handler
        initial_state["permission_gate_state"] = self.permission_gate_state

        result = await self.graph.ainvoke(initial_state)

        return result["agent_state"]

    def run(
        self,
        user_input: str,
        *,
        agent_state: AgentState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        return asyncio.run(
            self.arun(
                user_input,
                agent_state=agent_state,
                metadata=metadata,
            )
        )


async def demo() -> None:
    event_bus = RuntimeEventBus()

    def print_runtime_event(event: RuntimeEvent) -> None:
        print(f"[event] {event.compact_text()}")

    event_bus.subscribe(print_runtime_event)

    runner = AgentGraphRunner(
        event_bus=event_bus,
        emit_events=True,
        config={
            "permissions": {
                "mode": "default",
            },
            "agent": {
                "max_iterations": 5,
                "max_context_messages": 20,
            },
            "llm": {
                "default_provider": "qwen",
                "fallback_to_mock": False,
                "providers": {
                    "qwen": {
                        "provider": "qwen",
                        "api_format": "openai_compatible",
                        "model": "qwen3.6-flash",
                        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                        "api_key_env": "DASHSCOPE_API_KEY",
                        "temperature": 0.2,
                        "max_tokens": 2048,
                    }
                },
            },
        }
    )

    print("Run real LLM file_read message:")
    state = await runner.arun("Read README.md and briefly summarize its contents.")

    print(
        json.dumps(
            state.summary(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    last_message = state.get_last_message()
    if last_message:
        print(last_message.content)

    print("\nRun real LLM grep message:")
    state = await runner.arun("Search the project for all async def occurrences and return the matches.")

    print(
        json.dumps(
            state.summary(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    last_message = state.get_last_message()
    if last_message:
        print(last_message.content)

    print("\nRun real LLM glob message:")
    state = await runner.arun("Find all Python files under src/pywork/tools.")

    print(
        json.dumps(
            state.summary(),
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )

    last_message = state.get_last_message()
    if last_message:
        print(last_message.content)

    print("\nFull AgentState:")
    print(state.to_json(indent=2))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
