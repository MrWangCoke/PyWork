from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

from pywork.runtime.state import AgentState, AgentStatus, create_agent_state
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
    """
    LangGraph 内部状态。

    注意：
    真正的 Agent 状态对象放在 agent_state 里。
    其他字段是当前图执行过程中的临时数据。
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

    tool_registry: ToolRegistry
    config: dict[str, Any]
    metadata: dict[str, Any]


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
            system_prompt="You are PyWork, a coding agent.",
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
    开始新一轮用户输入前，清理上一轮的临时状态。
    不清空 messages，不清空历史 tool_calls。
    """
    if state.status in {
        AgentStatus.FINISHED,
        AgentStatus.ERROR,
        AgentStatus.CANCELLED,
    }:
        state.set_idle()

    state.current_tool_call_id = None
    state.last_error = None
    state.touch()


def user_input_node(data: AgentGraphData) -> dict[str, Any]:
    """
    UserInput 节点。

    负责：
    1. 接收用户输入
    2. 写入 AgentState.messages
    3. 让 Agent 进入 idle 状态，准备下一步构建上下文
    """
    state = data["agent_state"]
    user_input = data.get("user_input", "").strip()

    reset_agent_turn_state(state)

    if user_input:
        state.add_user_message(user_input)

    state.set_idle()

    return {
        "agent_state": state,
    }


def build_context_node(data: AgentGraphData) -> dict[str, Any]:
    """
    BuildContext 节点。

    负责构建给 LLM 使用的上下文：
    - messages
    - tool definitions
    - workspace 信息
    - permission_mode
    """
    state = data["agent_state"]
    registry = get_registry(data)

    context = {
        "messages": state.to_messages_payload(),
        "tool_definitions": registry.list_definitions(),
        "workspace_path": get_workspace_path(data),
        "project_root": get_project_root(data),
        "permission_mode": get_permission_mode(data),
        "iteration": state.iteration,
        "checkpoint_id": state.checkpoint_id,
    }

    return {
        "context": context,
        "agent_state": state,
    }


def parse_tool_shortcut(user_input: str) -> dict[str, Any] | None:
    """
    临时工具调用语法。

    当前还没接真实 LLM，所以先支持：

        /tool echo hello
        /tool echo {"text": "hello"}

    这样可以测试完整 Tool 执行链路。
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
    """
    临时 LLM 输出。

    后面接真实模型时，可以把这个函数替换成：
    - OpenAI tool calling
    - Anthropic tool use
    - DeepSeek/OpenAI-compatible function calling
    """
    user_input = data.get("user_input", "").strip()

    shortcut_tool_call = parse_tool_shortcut(user_input)

    if shortcut_tool_call is not None:
        return json.dumps(
            shortcut_tool_call,
            ensure_ascii=False,
        )

    return (
        "收到你的输入：\n\n"
        f"> {user_input}\n\n"
        "当前 Runtime Graph 已经跑通。"
    )


def call_llm_node(data: AgentGraphData) -> dict[str, Any]:
    """
    CallLLM 节点。

    当前版本：
    - 增加 iteration
    - 设置 thinking 状态
    - 使用 mock_call_llm_output() 模拟模型输出

    后续替换点：
    - 在这里调用真正的 LLM Provider
    """
    state = data["agent_state"]

    if not state.can_continue():
        state.set_error("max iterations reached")
        return {
            "agent_state": state,
            "llm_output": "",
            "stop_reason": "max iterations reached",
        }

    state.next_iteration()
    state.set_thinking()

    llm_output = mock_call_llm_output(data)

    return {
        "agent_state": state,
        "llm_output": llm_output,
    }


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


def parse_tool_call_node(data: AgentGraphData) -> dict[str, Any]:
    """
    ParseToolCall 节点。

    负责从 LLM 输出里解析工具调用。

    当前支持：
    - 普通文本：当成 assistant message
    - JSON：{"tool_name": "...", "arguments": {...}}
    """
    state = data["agent_state"]
    registry = get_registry(data)
    llm_output = data.get("llm_output", "")

    parsed = try_load_tool_call_json(llm_output)

    if parsed is None:
        state.add_assistant_message(llm_output)
        state.set_finished()

        return {
            "agent_state": state,
            "parsed_tool_call": None,
            "observation": "",
        }

    call = build_tool_call_from_parsed(parsed, registry)
    state.add_tool_call(call)

    return {
        "agent_state": state,
        "parsed_tool_call": call,
    }


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
    PermissionCheck 的核心判断。

    当前规则：
    - 没有工具调用：允许继续
    - plan 模式：不执行工具
    - default：允许 safe/low
    - accept_edits：允许 safe/low/medium
    - bypass_permissions：全部允许
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


def permission_check_node(data: AgentGraphData) -> dict[str, Any]:
    """
    PermissionCheck 节点。

    负责根据 permission_mode 和工具 risk_level 判断是否允许执行。
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

    return {
        "agent_state": state,
        "permission_decision": decision,
    }


async def execute_tool_node(data: AgentGraphData) -> dict[str, Any]:
    """
    ExecuteTool 节点。

    负责执行工具。
    """
    state = data["agent_state"]
    registry = get_registry(data)

    call = data.get("parsed_tool_call")
    decision = data.get("permission_decision")

    if call is None:
        return {
            "agent_state": state,
            "tool_result": None,
            "observation": "",
        }

    if decision is not None and not decision.allowed:
        result = ToolResult.cancelled_result(
            call=call,
            reason=decision.reason,
            metadata={
                "permission_denied": True,
                "requires_confirmation": decision.requires_confirmation,
            },
        )

        state.add_tool_result(result)

        return {
            "agent_state": state,
            "tool_result": result,
            "observation": decision.reason,
        }

    result = await registry.execute_call(
        call,
        context=ToolExecutionContext(
            workspace_path=get_workspace_path(data),
            project_root=get_project_root(data),
            permission_mode=get_permission_mode(data),
            metadata={
                "checkpoint_id": state.checkpoint_id,
                "iteration": state.iteration,
            },
        ),
    )

    state.add_tool_result(result)

    return {
        "agent_state": state,
        "tool_result": result,
        "observation": registry.render_result(result),
    }


def append_observation_node(data: AgentGraphData) -> dict[str, Any]:
    """
    AppendObservation 节点。

    负责把工具观察结果追加回 Agent 消息。
    """
    state = data["agent_state"]
    call = data.get("parsed_tool_call")
    result = data.get("tool_result")
    observation = data.get("observation", "")

    if call is None:
        return {
            "agent_state": state,
        }

    if result is None:
        state.add_assistant_message(
            f"工具 `{call.tool_name}` 没有返回结果。"
        )
        state.set_error("tool result missing")

        return {
            "agent_state": state,
        }

    if result.success:
        state.add_assistant_message(
            f"工具 `{call.tool_name}` 执行完成。\n\n"
            f"观察结果：\n\n{observation}"
        )
        state.set_finished()
    else:
        state.add_assistant_message(
            f"工具 `{call.tool_name}` 未执行成功。\n\n"
            f"原因：\n\n{observation}"
        )

    return {
        "agent_state": state,
    }


def compact_messages_if_needed(
    state: AgentState,
    *,
    max_messages: int = 40,
) -> bool:
    """
    简单上下文压缩。

    当前策略：
    - 消息数量不超过 max_messages：不压缩
    - 超过后保留第一条 system 消息和最近若干条消息
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
    CompactIfNeeded 节点。

    后续可以替换成真正的摘要压缩。
    当前先做简单截断保留。
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

    compacted = compact_messages_if_needed(
        state,
        max_messages=max_messages,
    )

    return {
        "agent_state": state,
        "metadata": {
            **data.get("metadata", {}),
            "compacted": compacted,
        },
    }


def continue_or_stop_node(data: AgentGraphData) -> dict[str, Any]:
    """
    ContinueOrStop 节点。

    决定是否继续下一轮 Agent 循环。
    """
    state = data["agent_state"]

    if state.status in {
        AgentStatus.FINISHED,
        AgentStatus.ERROR,
        AgentStatus.CANCELLED,
        AgentStatus.WAITING_PERMISSION,
    }:
        return {
            "should_continue": False,
            "stop_reason": state.status.value,
            "agent_state": state,
        }

    if state.iteration >= state.max_iterations:
        state.set_error("max iterations reached")

        return {
            "should_continue": False,
            "stop_reason": "max iterations reached",
            "agent_state": state,
        }

    return {
        "should_continue": state.can_continue(),
        "stop_reason": "",
        "agent_state": state,
    }


def route_continue_or_stop(data: AgentGraphData) -> GraphRoute:
    should_continue = bool(data.get("should_continue", False))

    return "continue" if should_continue else "stop"


def build_agent_graph():
    """
    构建 LangGraph 执行图。
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
    Agent Graph 运行器。

    外部推荐使用：
        runner = AgentGraphRunner()
        state = await runner.arun("hello")
    """

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.config = config or {}
        self.graph = build_agent_graph()

    async def arun(
        self,
        user_input: str,
        *,
        agent_state: AgentState | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentState:
        initial_state = create_default_agent_graph_state(
            user_input=user_input,
            registry=self.registry,
            config=self.config,
            agent_state=agent_state,
            metadata=metadata or {},
        )

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
    runner = AgentGraphRunner(
        config={
            "permissions": {
                "mode": "default",
            },
            "agent": {
                "max_iterations": 5,
                "max_context_messages": 20,
            },
        }
    )

    print("Run normal message:")
    state = await runner.arun("hello PyWork")

    print(json.dumps(state.summary(), ensure_ascii=False, indent=2))
    print(state.get_last_message().content if state.get_last_message() else "")

    print("\nRun tool message:")
    state = await runner.arun(
        "/tool echo Hello from LangGraph tool path.",
        agent_state=state,
    )

    print(json.dumps(state.summary(), ensure_ascii=False, indent=2))
    print(state.get_last_message().content if state.get_last_message() else "")

    print("\nFull AgentState:")
    print(state.to_json(indent=2))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())