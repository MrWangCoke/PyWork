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
    LangGraph 闂備礁鎲￠崝鏇㈠箠濮椻偓瀹曟洟骞橀钘変汗闂佺厧鎽滈。浠嬪磻閹惧瓨濯寸痪鐗埫禍?

    婵犵數鍋涢ˇ顓㈠礉瀹ュ绀堝ù鐓庣摠閺?
    闂備焦妞挎禍鐐哄窗閹伴偊鏁嗘繝濠傜墛閸?Agent 闂備胶绮…鍫ュ春閺嶎厼鐒垫い鎴ｆ硶椤︼附銇勯弴銊ユ灓闁瑰憡甯￠崺鍕礃閳哄倹绶梻?agent_state 闂傚倷鐒﹁ぐ鍐偖椤愶箑鐒?
    闂備胶顭堢换鎴濓耿閸︻厼鍨濇い鎺戝€甸崑鎾斥槈濞嗘ɑ鐣峰銈嗘煥閻倸顕ｆ导鎼晬婵浜瓏闂備礁鎲￠幐鍝ョ矓閺夋嚚鐟邦潨閳ь剟鐛鍫▉濡炪們鍨洪崹璺侯焽婵犳艾鐐婇柍鍦亾閻撶姵绻涢幋鐐存儎闁告ɑ鍎抽埢鎾诲箣閻愮鏋栭柣搴㈢⊕钃辨い蟻鍥ㄧ厸濞达絽鎼。鑲┾偓瑙勬尫缁舵碍淇?
    """

    agent_state: AgentState
    user_input: str

    context: dict[str, Any]
    llm_output: str

    parsed_tool_call: ToolCall | None
    permission_decision: PermissionDecision | None

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

    return str(
        get_nested_config_value(
            config,
            "permissions.mode",
            get_nested_config_value(
                config,
                "app.permission_mode",
                "default",
            ),
        )
    )


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
        "tool_result": None,
        "observation": "",
        "should_continue": False,
        "stop_reason": "",
    }


def reset_agent_turn_state(state: AgentState) -> None:
    """
    闁诲孩顔栭崰鎺楀磻閹炬枼鏀芥い鏃傗拡閸庢劙鏌＄仦鏂ゆ敾缂佸顦甸崺鈧い鎺嶇劍婵粍銇勯弮鍌氫壕闁哄棗绻橀弻鐔衡偓闈涙啞閻掕法绱掓０婵嗗籍鐎规洘顨婃俊鐑藉Ψ閵夘喗鐎梻浣瑰缁嬫垿寮甸鍌滃崥濠电姵纰嶉崑鐘绘煕閳╁啯绀堥柣鐔稿姇閳藉骞橀姘闂佸搫顦遍崕鎰板窗濮橆兘鏋旈柟瀵稿仧閳绘棃鎮楀☉娅虫垿藝瑜旈弻锝呂熼崹顔惧帿闂侀€炲苯鍘搁梻鍕Ч閸┾偓?
    濠电偞鍨堕幐鍝ョ矓閻㈢數鍗氶柤濮愬€楅惌?messages闂備焦瀵х粙鎴炵附閺冨倻绠斿璺侯焾閳ь剚甯″畷銊╊敍濡や焦娅堥梻浣告啞閿氭俊顐ｎ殙閵?tool_calls闂?
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
    UserInput 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸杩?

    闂佽崵濮甸崝妤呭窗閺囥垺鍎楁俊銈呮噺閺?
    1. 闂備浇顫夋禍浠嬪磿閺屻儱鏋佺憸鐗堝笚閸嬨劑鏌曟繝蹇曠暠闁绘挻娲熷鍫曞醇閻旂纰嶉梺?
    2. 闂備礁鎲￠崝鏍偡閵夆晛鐭?AgentState.messages
    3. 闂?Agent 闂佸搫顦弲婊呯矙閹达箑鐭?idle 闂備胶绮…鍫ュ春閺嶎厼鐒垫い鎴ｆ硶缁涘繒绱掓潏銊х疄鐎规洘鐟╁畷鍗炍旀繝鍐冿絾绻涢幋鐐村碍妞ゆ垵瀚划鈺呭级閹搭厽妗ㄩ梺闈涱檧婵″洭鍩€椤掑鐏犳い鏇熺懄濞碱亪骞嶉鐐暟濠电偞鍨堕幐鎼侇敄閸℃稑鍑?
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
    BuildContext 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸杩?

    闂佽崵濮甸崝妤呭窗閺囥垺鍎楁俊銈呮噹閸戠娀鏌涢弴銊ヤ簽缂佹唻濡囩槐?LLM 濠电偠鎻紞鈧繛澶嬫礋瀵偊濡舵径瀣壄闂佸憡娲︽禍鐐烘偡閹邦兘妲堥柟鐐墯閸庢劙鏌＄€ｎ亜鏆ｉ柡?
    - messages
    - tool definitions
    - workspace 濠电儑绲藉ú鐘诲礈濠靛洤顕?
    - permission_mode
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
    濠电偞鍨堕幐鎼佹偤閵娿儺娓婚柛灞惧嚬閸熷懘鏌曟径鍫濆姎鐎电増妫冮幃鍦偓锝庝簻閺嗙喖鏌℃担闈涒偓妤呭箯閻樻椿鏁囬柣鏃€浜介埀顒€锕弻?

    闁荤喐绮庢晶妤呭箰閸涘﹥娅犻柣妯挎珪娴溿倖淇婇婵嗗惞婵﹪绠栭弻鐔煎箒閹烘垵濮㈠┑鐘亾闁挎稑瀚ч崑?LLM闂備焦瀵х粙鎴﹀嫉椤掍焦娅犵€广儱妫涢々鐑芥煏婢跺牆鍔氶悽顖樺劦閺岋繝鍩€椤掑嫷鏁嶆繛鎴烆焽濡茬兘姊?

        /tool echo hello
        /tool echo {"text": "hello"}

    闂佸搫顦弲婊堟偡閿曞倹鍋嬮梺顒€绉撮惌妤併亜閺嶃劎鎳佺紒銊у缁绘盯寮堕幋顓炲壍闂佷紮瀵岄崹鎶藉焵椤掑倹鍤€闁哄牜鍓熷?Tool 闂備礁婀遍悷鎶藉幢閳哄倹鏉搁梻鍌欒兌閸嬫挸鐣峰鈧幆灞俱偅閸愩剮?
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
    闂備礁鎲＄敮鍥磹閺嶎厼钃熼柛銉簵娴滃綊鏌熼幆褍鏆辨い銈呮嚇濮婃椽寮剁捄銊愩倝鏌ｉ妶鍛棦闁哄苯鐬兼禒锕傛倷椤戭偓绠撻弻娑橆潩妤ｅ啯顎嶅┑鐘亾?LLM 闂傚倷鐒﹀妯肩矓閸洘鍋柛鈩冪☉杩?

    婵犵數鍋涙径鍥礈濠靛棴鑰?LLM 闂傚倷鐒﹀妯肩矓閸洘鍋柛鈩冪☉缁秹鎮规担鍛婅础缂佲偓婢跺瞼纾藉ù锝呯墕閹虫劙寮ィ鍐╁仯?mock闂?
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
    role = getattr(message, "role", "")

    if isinstance(role, MessageRole):
        return role.value

    return str(role)


def agent_message_to_llm_message(message: Any) -> AnyMessage | None:
    """
    AgentState 闂備礁鎲￠崝鏇㈠箠濮椻偓瀹?message -> schemas/message_schema.py 闂備焦鐪归崝宀€鈧凹鍘剧划鏃堟倻閽樺鐣冲銈嗙墬娑撹绗熼埀顒€鐣烽崷顓涘亾閿濆啫濡烽柛?
    """
    role = message_role_value(message)
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
        # 闂備胶鍎甸弲鈺呭窗閺嶃劍娅?AgentState 闂?tool message 濠电偞鍨堕幐鍝ョ矓瀹曞洨鍗氶悗鐢电《閸嬫挾鎲撮崟顔碱棟闂?tool_call_id闂?
        # 闂備胶顭堢换鎰版偋閹邦優褰掑幢濞戞顓?user 闂備礁鎲￠悷顖炲垂閻㈠壊鏁嬮柟娈垮枟閸犲棝鏌涚仦鐐殤婵＄虎浜炵槐鎾存媴鐟欏嫬闉嶉梺璇茬箰椤︾敻寮鍥︽勃闁兼亽鍎遍埀顒傛暬濮婂宕煎☉妯间患闂佸摜鍟块崑鎾绘⒑閸涘娈旀繛灞傚妽椤ㄣ儵宕堕鈧粻锝夋煙闁箑骞橀柛銊ャ偢閺?ToolMessage闂?
        return create_user_message(
            "闁诲氦顫夐幃鍫曞磿闁秴鐭楅柛褎顨呯粻銉ф喐閹达负鈧線骞嬮悩鐢碉紲闂佽鍎抽顓熺椤栫偞鐓ユ繛鎴炵懅閹芥\n" + content,
            metadata={
                **metadata,
                "source": "agent_state_tool_observation",
            },
        )

    if role == "error":
        return create_user_message(
            "Runtime 闂傚倷鐒︾€笛囨偡閵娾晩鏁嬮柕鍫濐槹閺咁剚鎱ㄥ┑鍥ㄢ枎\n" + content,
            metadata={
                **metadata,
                "source": "agent_state_error",
            },
        )

    return None


def build_llm_messages(data: AgentGraphData) -> list[AnyMessage]:
    """
    闂備礁鎼鍛偓姘嵆閸┾偓妞ゆ帒鍊稿瓭閻熸粍婢橀崐鑽ゆ?LLMRouter 闂?messages闂?
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
    濠?ToolRegistry 闂傚倷鐒﹁ぐ鍐嚐椤栨縿浜归柛銉戝苯鏅犻梺闈涱槶閸庡崬顕ラ弮鍫熷€甸悷娆忓閻擃垳绱掗悩闈涙灈闁轰礁绉舵禒锕傛嚃閳哄啫鈧偟绱?LLM tool calling闂?

    闂佸搫顦弲婊堝蓟閵娿儍娲冀椤撶偛鍞ㄩ梺缁樻尭濞撮娑甸埀顒勬⒑閸濆嫷妲堕柛搴㈡尦瀹曢潧顭ㄩ崼鐔告珫?
        registry.list_definitions()

    婵犳鍣徊鐣屾崲鐎ｎ喗鐓?definition 濠电姰鍨归悥銏ゅ礋閸偅鎮欓梻浣告惈鐎氱兘宕归悾灞绢潟?
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
    闂佽崵濮撮鍛村疮娴兼潙鏋侀柕鍫濐槹閸庢垿鏌ｉ弮鍫闁?LLMRouter闂?
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
    闂?AssistantMessage 闂備礁鎲￠崝鏍偡閵夈儍?AgentState闂?
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
    闂佽崵濮撮鍛村疮娴兼潙鏋?LLM闂?

    闂備礁鎼悧鍡浰囬崡鐑囪€块柨娑樺閸?LLM 闂傚倷鐒﹀妯肩矓閸洘鍋柛鈩冪☉缁秹鎮规担鍛婅础缂?
        闂佽崵濮撮鍛村疮娴兼潙鏋?LLMRouter

    婵犵數鍋涙径鍥礈濠靛棴鑰垮〒姘ｅ亾闁哄苯锕ら濂稿炊閳哄倻鈧?/ 闂傚倷鐒﹀妯肩矓閸洘鍋柛鈩兠欢鐐哄级閸偄浜悮婵嬫⒑閸濆嫬鏆㈡い鏃€鍔楀Σ?
        fallback 闂?mock_call_llm_output
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
    闂備胶顢婂▔娑㈡晝閿濆洨绠斿鑸靛姇鐟欙箓鎮橀悙闈涗壕闂傚嫬绉电换婵婎槼闁告梹顨呴埢?tool_calls 缂傚倸鍊烽懗鍫曞窗瀹ュ洨鍗氶悗娑櫳戞慨婊勩亜閹哄秶顦﹂柣?list[ToolCall]闂?

    闂備浇銆€閸嬫挻銇勯弽銊р槈闁伙富鍣ｉ弻?
    - list[ToolCall]
    - tuple[ToolCall]
    - 闂備礁鎲￠〃鍡椕洪幋锔界厒?ToolCall
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
    濠?AssistantMessage 闂傚倷鐒﹁ぐ鍐嚐椤栨縿浜?tool_calls闂?
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
    濠?LLMResponse 闂傚倷鐒﹁ぐ鍐嚐椤栨縿浜?tool_calls闂?

    providers.py 闂傚倷鐒﹁ぐ鍐儔閻撳簶鏋?LLMResponse 闁诲氦顫夐悺鏇犱焊濞嗘垵鍨濋柨鐔哄Т缁?OpenAI / Anthropic 闂備焦鐪归崝宀€鈧凹鍨抽幑銏ゅ箣閿曗偓閻?
    缂傚倸鍊烽懗鍫曞窗瀹ュ洨鍗氶悗闈涙啞閸犲棝鏌ㄥ┑鍡橆棤闁逞屽厸缁瑩鐛€ｎ喖绠涙い蹇撴閺?ToolCall闂備焦瀵х粙鎴﹀嫉椤掍焦娅犵€广儱妫涢々鐑芥煏婢跺牆濡跨紒鐘茬秺濮婃椽骞撻幒鎴缂備焦顨呴ˇ闈浳涙笟鈧崺鈧い鎺戝€归崯鍝劽归敐鍥剁劸闁哄懏鐟╅幃宄扳枎濞嗘垹蓱闂佽绨肩划娆忕暦閵忋倖鍋勫┑鍌氼槹缂?SDK 闂佽娴烽弫鎼併€佹繝鍥ㄥ瘶闁告洦鍨拌繚?
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
    mock 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀顒佺箾閹寸偞灏い鎴炲灩濡叉劕鈹戦崶鈺婃祫?JSON 闂佽瀛╃粙鎺椼€冮崼銉晞濞达絽婀遍埢鏃堟煛鐏炶鍔氭繛鍜冪節閹嘲鈻庡▎鎴犐戦梺璇叉唉濡嫰顢欒箛娑樜ㄩ柕澶堝劚缁额噣鏌ｉ悩宸剰闁哥喍鍗冲顐﹀Χ婢跺á?

    闂備浇銆€閸嬫挻銇勯弽銊р槈闁伙富鍣ｉ弻鈥愁吋閸パ冧粯缂備焦鍞荤换婵嬪极?
        {"tool_name": "grep", "arguments": {...}}

    闂備胶鎳撻悺銊╂偋椤撶姵顫?
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
    闂備胶顢婂▔娑㈡晝閵婎煈鏆伴梻浣告惈椤戝棝宕濋弴銏犲惞妞ゆ挶鍨洪崕?tool_calls 闂備礁鎲￠崝鏍偡閵夆晛鐭?GraphData闂?

    闁荤喐绮庢晶妤呭箰閸涘﹥娅?Runtime Graph 闂備胶顭堢换鎰版偋婵犲啯娅犻柣锝呮湰閸嬫鈧厜鍋撻柍褜鍓熼、鏍炊閵娧€鏋栭梺閫炲苯澧紒瀣樀椤㈡棃宕熼鍡欑厬闂備胶顭堢换鎺楀蓟婢舵劖鍎嶉柣鏂垮悑閸嬨劑鏌曟繝蹇曞矝闁?
    濠电姰鍨奸崺鏍儗椤曗偓閺屽苯顭ㄩ崘鎯ф櫊闂侀潧顦崕鍗烆嚗閺冨牊鍋ｉ悗锝庝簻閺嗙喖鏌℃担闈涒偓婵嗙暦濡ゅ懎閱囬柣鏂挎啞濠㈡帡姊?remaining_tool_calls闂備焦瀵х粙鎴︽嚐椤栫偛绠栨俊銈呮噺椤ュ牓鏌曡箛鏇炐㈤悹浣瑰絻闇夐柣姗嗗枛閸旀氨鈧娲栭惉濂告儉椤忓浂妲奸梺鎼炲妽瀹€鎼佺嵁瀹ュ牄浜归柟鐑樺灦椤?婵°倗濮烽崑鐐哄磿婵傛悶鈧線骞嬮敃鈧粻銉ф喐閹达负鈧線骞嬮敃鈧繚?
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
    ParseToolCall 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸杩?

    闂備浇銆€閸嬫挻銇勯弽銊р槈闁伙富鍠栭埥澶愬箻閹颁焦楔濡炪値鍋呯敮鈥愁嚕閸洖唯闁挎柨澧介崺宥夋⒑?
    1. 闂備焦妞挎禍鐐哄窗鎼淬劍鍋?LLMResponse.tool_calls
    2. 闂備焦妞挎禍鐐哄窗鎼淬劍鍋?AssistantMessage.tool_calls
    3. mock 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀?JSON 闂佽瀛╃粙鎺椼€冮崼銉晞濞达絽婀遍埢?

    闂佽崵鍠愰悷銉р偓姘煎墴瀹曞綊顢涢悙瀛樻珫?
    - 闂佽崵鍠愰悷杈╁緤妤ｅ啯鍊靛ù鐘差儏缁€?tool_call闂備焦瀵х粙鎺撶┍閾忚宕叉慨妞诲亾鐎?PermissionCheck / ExecuteTool
    - 婵犵數鍋涙径鍥礈濠靛棴鑰?tool_call闂備焦瀵х粙鎺戭潩閵娧冨灊闁靛ň鏅涚痪?Graph
    """
    agent_state = data["agent_state"]

    llm_response = data.get("llm_response")
    assistant_message = data.get("assistant_message")

    # 1. 濠电偞娼欓崥瀣晪闂佸憡蓱缁嬫帞绮?LLMResponse 闂佽崵鍠愰悷杈╁緤妤ｅ啯鍊?tool_calls
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

    # 2. 闂備礁鎲￠崝鏇犵矓瀹曞洤鍨?AssistantMessage 闂佽崵鍠愰悷杈╁緤妤ｅ啯鍊?tool_calls
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

    # 3. mock 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀顒勬⒑閹稿海鈯曢柣顓у枤閸?llm_output 闂佽瀛╃粙鎺椼€冮崼銉晞濞达絽婀遍埢鏃堟煠閼测晝绀嬮柟鐑橆殔閸?JSON 闁诲氦顫夐幃鍫曞磿闁秴鐭楅悹鎭掑妽鐎氼剟鏌涢幇鍏哥凹闁?
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

    # 4. 闂備礁鎼幏瀣闯閿濆鐒垫い鎺嶈兌椤ｆ煡鏌＄€ｎ亜鏆ｇ€殿喚鏁婚、妤呭焵椤掆偓鐓ら柡宥冨妼缁剁偟鈧箍鍎辩换鎺旂矆婢跺绠鹃悘鐐殿焾婢у弶绻濋埀顒佹媴閸撴彃鏅犻梺闈涱槶閸庡崬顕ラ弮鍫熷仯閻庯綆浜滈弳鐔兼煛?
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
    if mode == "bypass_permissions":
        return ToolRiskLevel.DANGEROUS

    if mode == "accept_edits":
        return ToolRiskLevel.MEDIUM

    if mode == "plan":
        return ToolRiskLevel.SAFE

    return ToolRiskLevel.LOW


def evaluate_permission(
    call: ToolCall | None,
    *,
    registry: ToolRegistry,
    permission_mode: str,
) -> PermissionDecision:
    """
    PermissionCheck 闂備焦鐪归崝宀€鈧凹鍙冮幃褏鈧湱濮烽悿鈧梺鍛婂姂閸斿矂鎮橀弻銉︾厸闁稿被鍊曢獮鎴︽煃?

    闁荤喐绮庢晶妤呭箰閸涘﹥娅犻柣妯虹仛閸犲棝鏌涢弴銊ヤ簻闁诲繒鍠栭弻?
    - 婵犵數鍋涙径鍥礈濠靛棴鑰垮ù锝呭閸熷懘鏌曟径鍫濆姎鐎电増妫冮幃鍦偓锝庝簻閺嗙喖鏌℃担闈涒偓婵嬪极瀹ュ拋娼╂い鎺嗗亾閻㈩垱甯￠幃瑙勬媴閸涘﹥鍠愰梺瑙勬尦椤ユ挾妲?
    - plan 婵犵妲呴崹顏堝焵椤掆偓绾绢厾娑甸埀顒勬⒑閹稿海鈯曢柣顓у枤缁厽寰勯幇顒傤吋閻熸粍鍨块妴渚€骞嬪┑鎰櫊闂侀潧顦崕鍗烆嚗?
    - default闂備焦瀵х粙鎺楁嚌妤ｅ啫鍌ㄩ柟瀵稿У婵?safe/low
    - accept_edits闂備焦瀵х粙鎺楁嚌妤ｅ啫鍌ㄩ柟瀵稿У婵?safe/low/medium
    - bypass_permissions闂備焦瀵х粙鎺楁嚌妤ｅ啫鐭楅煫鍥ㄧ⊕閻掗箖鏌曟繛鍨姎閻㈩垱甯￠幃?
    """
    if call is None:
        return PermissionDecision(
            allowed=True,
            reason="no tool call",
            requires_confirmation=False,
        )

    if permission_mode == "plan":
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


def permission_check_node(data: AgentGraphData) -> AgentGraphData:
    """
    PermissionCheck 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸杩?

    闂佽崵濮甸崝妤呭窗閺囥垺鍎楁俊銈呮噹閸愨偓闁荤喐鐟ョ€氼剛绮?permission_mode 闂備礁鎲＄划宀勬嚐椤栨稑顕遍柟鐗堟緲缁€?risk_level 闂備礁鎲＄敮鍥磹閺嶎厼钃熼柛銉墮閸欏﹥銇勯弽銊ь暡闁稿骸锕弻娑滅疀鎼淬垻銈板銈嗘煥缁绘﹢鐛鍫▉濡炪們鍨洪崹鎸庝繆?
    """
    state = data["agent_state"]
    registry = get_registry(data)
    call = data.get("parsed_tool_call")
    permission_mode = get_permission_mode(data)

    decision = evaluate_permission(
        call,
        registry=registry,
        permission_mode=permission_mode,
    )

    if call is not None and not decision.allowed:
        state.set_waiting_permission(call.call_id)

    data["agent_state"] = state
    data["permission_decision"] = decision

    return data


def create_graph_tool_context(data: AgentGraphData) -> ToolExecutionContext:
    """
    Create the ToolExecutionContext used by registry-backed graph tools.
    """
    signature = inspect.signature(ToolExecutionContext)
    parameters = signature.parameters

    workspace_path = Path(get_workspace_path(data)).expanduser().resolve()
    project_root = Path(get_project_root(data)).expanduser().resolve()

    kwargs: dict[str, Any] = {}

    if "workspace_path" in parameters:
        kwargs["workspace_path"] = str(workspace_path)

    if "project_root" in parameters:
        kwargs["project_root"] = str(project_root)

    if "config" in parameters:
        kwargs["config"] = data.get("config") or {}

    if "metadata" in parameters:
        kwargs["metadata"] = {
            "source": "runtime_graph.execute_tool_node",
            "permission_mode": get_permission_mode(data),
            "checkpoint_id": data["agent_state"].checkpoint_id,
        }

    if "permission_mode" in parameters:
        kwargs["permission_mode"] = get_permission_mode(data)

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
    agent_state: Any,
    result: ToolResult,
) -> None:
    """
    Record a ToolResult on AgentState, with fallbacks for older state APIs.
    """
    if hasattr(agent_state, "add_tool_result"):
        agent_state.add_tool_result(result)
        return

    metadata = {
        "success": result.success,
        "status": result.status,
        "duration_ms": result.duration_ms,
    }

    if hasattr(agent_state, "add_tool_message"):
        try:
            agent_state.add_tool_message(
                result.content,
                name=result.tool_name,
                tool_call_id=result.call_id,
                metadata=metadata,
            )
            return
        except TypeError:
            pass

    if hasattr(agent_state, "add_message"):
        try:
            agent_state.add_message(
                "tool",
                result.content,
                name=result.tool_name,
                tool_call_id=result.call_id,
                metadata=metadata,
            )
        except TypeError:
            agent_state.add_message(
                role="tool",
                content=result.content,
                metadata=metadata,
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
    缂傚倷鑳舵慨顓㈠磻閹剧粯鐓曟俊銈勭劍绾捐崵绱掑Δ鈧崐鍦矙婢跺备鍋撻敐搴′簼闁诲寒鍣ｉ弻娑樜熸笟顖氬壋缂傚倸鍊搁ˇ鍨繆?

    闁荤喐绮庢晶妤呭箰閸涘﹥娅犻柣妯兼暩妞规娊鏌″搴′簼婵炲牞缍侀弻?
    - 婵犵數鍋為崹鐢告偋婵犲啫顕遍柛娑欐綑閺嬩礁霉閸忓吋缍戞繛鍛€曢埥澶愬箻瀹曞泦銉х磼婢跺﹦鎽犳繛?max_messages闂備焦瀵х粙鎺楁儗椤斿墽绠斿鑸靛姇閸屻劑鎮楅敐搴″缂?
    - 闂佺儵鍓濈敮鎺楀箠閹捐埖宕查柛鎰靛枛鐟欙箓骞栫€涙绠樼紒鐘辩矙閺岋綁鏁愰崱娆愬櫘濡炪倖鍨靛ú锔剧矙婢舵劕鐒垫い鎺戝缁?system 婵犵數鍋為崹鐢告偋婵犲啫顕遍柛娑欐綑濡炰粙鎮橀悙闈涗壕濞寸姵锕㈠鍫曞煛閸屾氨浼囬柣搴㈠搸閸ㄤ粙鎮伴鈧畷锝嗗緞鐎ｎ厽瀚繝鐢靛仦閸ㄧ敻鎮ф繝鍐嚤?
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
    CompactIfNeeded 闂備胶鍘ч幖顐﹀磹婵犳艾纾婚柨婵嗩槸杩?

    闂備礁鎲￠懝鎯归悜鑺ュ仺濠电姵鑹鹃惌妤併亜閺嶃劎鎳佺紒銊ゅ嵆閺岋繝宕堕妸銉殝閻庤娲橀悡锟犵嵁鐎ｎ喖绠涙い鏃囨閻掔鈹戦鐣岀缂佹彃鐏濋埢鎾诲箣閿曗偓缁犵儤淇婇娑卞劌婵炶缍侀弻娑樜熸笟顖氬壋缂傚倸鍊搁ˇ鍨繆?
    闁荤喐绮庢晶妤呭箰閸涘﹥娅犻柣妯款嚙缁€鍌炴煕椤愶絿绠戠紒顔炬暩缁辨帡鎮╅顫闂備礁鎲￠〃鍡椕哄鈧畷娲川閺夋垶顥濆銈嗘尰缁诲嫮绮绘禒瀣厽闁挎繂妫涢幊鈧梺?
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
    闂備礁鎲＄敮鍥磹閺嶎厼钃?Graph 闂備礁鎼€氱兘宕规导鏉戠畾濞达綀娅ｇ壕鑲╂喐韫囨稒鍋ㄥ┑鐘宠壘杩?

    闂佽崵鍠愰悷銉р偓姘煎墴瀹曞綊顢涢悙瀛樻珫?
    - 闂?error闂備焦瀵х粙鎺楊敆閻犵郸p
    - 闂?pending tool_call闂備焦瀵х粙鎺楁偤閻犵tinue
    - 闂?tool_call闂備焦瀵х粙鎺楊敆閻犵郸p
    - 闂佺儵鍓濈敮鎺楀箠閹捐埖宕?max_iterations闂備焦瀵х粙鎺楊敆閻犵郸p
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
    闂備礁鎼鍛偓姘煎墰缁?LangGraph 闂備礁婀遍悷鎶藉幢閳哄倹鏉搁梻浣规偠閸庝即宕熷鈧崺鈧?
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
    Agent Graph 闂佸搫顦弲婊堝礉濮椻偓閵嗕線骞嬮敃鈧梻顖炴煏婵犲繒宀涢柛?

    濠电姰鍨奸崺鏍偋閻樿纾块柟缁㈠枛缁犳娊鏌曟繛鍨缂佲偓閸愵煁褰掓晲閸喓銆婇梺杞伴檷閸婃繈寮?
        runner = AgentGraphRunner()
        state = await runner.arun("hello")
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        config: dict[str, Any] | None = None,
        llm_router: Any | None = None,
        event_bus: RuntimeEventBus | None = None,
        emit_events: bool = True,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.config = config or {}
        self.llm_router = llm_router
        self.event_bus = event_bus or get_default_event_bus()
        self.emit_events = emit_events
        self.graph = build_agent_graph()

    async def arun(
        self,
        user_input: str,
        *,
        agent_state: AgentState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        metadata = metadata or {}

        initial_state = create_default_agent_graph_state(
            user_input=user_input,
            registry=self.registry,
            config=self.config,
            agent_state=agent_state,
            metadata=metadata,
        )

        run_id = str(metadata.get("run_id") or new_run_id())
        session_id = metadata.get("session_id")
        
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
