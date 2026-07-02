from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.tool import BaseTool, EchoTool, ToolExecutionContext


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolRegistryError(Exception):
    """
    工具注册表基础异常。
    """

    pass


class ToolAlreadyRegisteredError(ToolRegistryError):
    """
    工具重复注册。
    """

    pass


class ToolNotFoundError(ToolRegistryError):
    """
    工具不存在。
    """

    pass


@dataclass(frozen=True)
class ToolRegistryEntry:
    """
    注册表中的工具条目。
    """

    name: str
    tool: BaseTool
    registered_at: datetime = field(default_factory=utc_now)
    source: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tool_class": self.tool.__class__.__name__,
            "description": self.tool.description,
            "risk_level": self.tool.get_risk_level().value,
            "registered_at": self.registered_at.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
        }


def normalize_tool_name(name: str) -> str:
    normalized = name.strip()

    if not normalized:
        raise ValueError("tool name cannot be empty")

    return normalized


class ToolRegistry:
    """
    PyWork 工具注册表。

    负责：
    - 注册工具
    - 查找工具
    - 列出工具
    - 执行工具
    """

    def __init__(self) -> None:
        self._entries: dict[str, ToolRegistryEntry] = {}

    def register(
        self,
        tool: BaseTool,
        *,
        overwrite: bool = False,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> BaseTool:
        """
        注册一个工具实例。
        """
        if not isinstance(tool, BaseTool):
            raise TypeError("tool must be an instance of BaseTool")

        name = normalize_tool_name(tool.name)

        if name in self._entries and not overwrite:
            raise ToolAlreadyRegisteredError(f"Tool {name!r} is already registered")

        self._entries[name] = ToolRegistryEntry(
            name=name,
            tool=tool,
            source=source,
            metadata=metadata or {},
        )

        return tool

    def register_many(
        self,
        tools: Iterable[BaseTool],
        *,
        overwrite: bool = False,
        source: str = "manual",
    ) -> list[BaseTool]:
        """
        批量注册工具。
        """
        registered: list[BaseTool] = []

        for tool in tools:
            registered.append(
                self.register(
                    tool,
                    overwrite=overwrite,
                    source=source,
                )
            )

        return registered

    def unregister(self, name: str) -> BaseTool:
        """
        取消注册工具。
        """
        normalized = normalize_tool_name(name)

        if normalized not in self._entries:
            raise ToolNotFoundError(f"Tool {normalized!r} is not registered")

        entry = self._entries.pop(normalized)
        return entry.tool

    def clear(self) -> None:
        """
        清空注册表。
        """
        self._entries.clear()

    def has(self, name: str) -> bool:
        """
        判断工具是否存在。
        """
        try:
            normalized = normalize_tool_name(name)
        except ValueError:
            return False

        return normalized in self._entries

    def get(self, name: str) -> BaseTool | None:
        """
        查找工具，不存在时返回 None。
        """
        normalized = normalize_tool_name(name)
        entry = self._entries.get(normalized)

        if entry is None:
            return None

        return entry.tool

    def require(self, name: str) -> BaseTool:
        """
        查找工具，不存在时抛异常。
        """
        tool = self.get(name)

        if tool is None:
            raise ToolNotFoundError(f"Tool {name!r} is not registered")

        return tool

    def get_entry(self, name: str) -> ToolRegistryEntry | None:
        """
        获取注册条目。
        """
        normalized = normalize_tool_name(name)
        return self._entries.get(normalized)

    def list_names(self) -> list[str]:
        """
        列出所有工具名。
        """
        return sorted(self._entries.keys())

    def list_tools(self) -> list[BaseTool]:
        """
        列出所有工具实例。
        """
        return [self._entries[name].tool for name in self.list_names()]
    
    def list(self) -> list[BaseTool]:
        """
        返回已注册工具列表。

        这是 list_tools() 的别名，用来满足：

            ToolRegistry.register(tool) -> ToolRegistry.list()

        这种更直观的调用方式。
        """
        return self.list_tools()

    def list_entries(self) -> list[ToolRegistryEntry]:
        """
        列出所有注册条目。
        """
        return [self._entries[name] for name in self.list_names()]

    def list_definitions(self) -> list[dict[str, Any]]:
        """
        列出所有工具定义。

        后面可以给 LLM Tool Calling 使用。
        """
        return [tool.get_definition() for tool in self.list_tools()]

    def list_by_risk_level(
        self,
        risk_level: ToolRiskLevel | str,
    ) -> list[BaseTool]:
        """
        按风险等级列出工具。
        """
        target = ToolRiskLevel(risk_level)

        return [
            tool
            for tool in self.list_tools()
            if tool.get_risk_level() == target
        ]

    def find(self, query: str) -> list[BaseTool]:
        """
        模糊查找工具。

        会匹配：
        - 工具名
        - 工具描述
        """
        keyword = query.strip().lower()

        if not keyword:
            return self.list_tools()

        matched: list[BaseTool] = []

        for tool in self.list_tools():
            haystack = f"{tool.name}\n{tool.description}".lower()

            if keyword in haystack:
                matched.append(tool)

        return matched

    def create_call(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCall:
        """
        根据工具名创建 ToolCall。
        """
        tool = self.require(tool_name)

        return tool.create_call(
            arguments or {},
            metadata=metadata or {},
        )

    async def run_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        context: ToolExecutionContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        根据工具名和参数执行工具。

        常用入口：
            await registry.run_tool("echo", {"text": "hello"})
        """
        tool = self.require(tool_name)

        call = tool.create_call(
            arguments or {},
            metadata=metadata or {},
        )

        return await tool.run(
            call,
            context or ToolExecutionContext(),
        )

    async def execute_call(
        self,
        call: ToolCall,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """
        执行已有 ToolCall。

        后面 LLM 生成 ToolCall 后，会走这个方法。
        """
        tool = self.require(call.tool_name)

        return await tool.run(
            call,
            context or ToolExecutionContext(),
        )

    def render_result(self, result: ToolResult) -> str:
        """
        使用对应工具渲染 ToolResult。
        """
        tool = self.get(result.tool_name)

        if tool is not None:
            return tool.render_result(result)

        if result.success:
            return result.content or json.dumps(
                result.data,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        return result.content or result.error or "Tool failed."

    def to_dict(self) -> dict[str, Any]:
        """
        注册表信息转 dict。
        """
        return {
            "count": len(self._entries),
            "tools": [
                entry.to_dict()
                for entry in self.list_entries()
            ],
        }

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False

        return self.has(name)

    def __iter__(self):
        return iter(self.list_tools())

    def __repr__(self) -> str:
        names = ", ".join(self.list_names())
        return f"ToolRegistry(count={len(self)}, tools=[{names}])"


def create_default_registry() -> ToolRegistry:
    """
    创建默认工具注册表。

    当前默认注册：
    - EchoTool
    - FileReadTool
    - GlobTool
    - GrepTool
    - FileWriteTool
    - FileEditTool
    - BashTool
    - PowerShellTool

    注意：
    file_write / file_edit / bash / powershell 都是高风险工具。
    它们注册进来只是为了让 Agent 能看到工具定义。
    真正执行前必须经过 Runtime PermissionGate。
    """
    from pywork.tools.bash import BashTool
    from pywork.tools.file_edit import FileEditTool
    from pywork.tools.file_read import FileReadTool
    from pywork.tools.file_write import FileWriteTool
    from pywork.tools.glob import GlobTool
    from pywork.tools.grep import GrepTool
    from pywork.tools.powershell import PowerShellTool
    from pywork.tools.agent_tool import AgentTool
    from pywork.tools.coordinator_tool import CoordinatorTool
    from pywork.tools.send_message import SendMessageTool
    from pywork.tools.task_update import TaskUpdateTool
    from pywork.tools.task_tools import (
        TaskCreateTool,
        TaskListTool,
        TaskOutputTool,
        TaskStopTool,
    )
    from pywork.tools.team_create import TeamCreateTool
    from pywork.tools.team_delete import TeamDeleteTool


    registry = ToolRegistry()

    registry.register(
        EchoTool(),
        source="builtin",
        metadata={
            "category": "utility",
            "requires_permission_gate": False,
        },
    )

    registry.register(
        AgentTool(),
        source="builtin",
        metadata={
            "category": "agent",
            "operation": "delegate",
            "requires_permission_gate": False,
            "description": "Create, route, run, inspect, and stop SubAgents.",
        },
    )

    registry.register(
        CoordinatorTool(),
        source="builtin",
        metadata={
            "category": "agent",
            "operation": "coordinate",
            "requires_permission_gate": False,
            "requires_runtime_object": "subagent_manager",
            "description": "Coordinate multiple SubAgent workers in parallel or sequence.",
        },
    )

    registry.register(
        SendMessageTool(),
        metadata={
            "category": "agent",
            "capability": "message",
            "requires_runtime_object": "mailbox",
        },
    )
    
    registry.register(
        TaskUpdateTool(),
        metadata={
            "category": "task",
            "capability": "update_status",
            "requires_runtime_object": "task_manager_or_team",
        },
    )

    registry.register(
        TaskCreateTool(),
        metadata={
            "category": "task",
            "capability": "create",
            "requires_runtime_object": "task_manager_or_team_or_subagent_manager",
        },
    )

    registry.register(
        TaskListTool(),
        metadata={
            "category": "task",
            "capability": "list",
            "requires_runtime_object": "task_manager_or_team",
        },
    )

    registry.register(
        TaskOutputTool(),
        metadata={
            "category": "task",
            "capability": "output",
            "requires_runtime_object": "task_manager_or_team",
        },
    )

    registry.register(
        TaskStopTool(),
        metadata={
            "category": "task",
            "capability": "stop",
            "requires_runtime_object": "task_manager_or_team_or_subagent_manager",
        },
    )

    registry.register(
        TeamCreateTool(),
        metadata={
            "category": "team",
            "capability": "create",
            "requires_runtime_object": "team_registry",
        },
    )

    registry.register(
        TeamDeleteTool(),
        metadata={
            "category": "team",
            "capability": "delete",
            "requires_runtime_object": "team_registry",
        },
    )

    registry.register(
        FileReadTool(),
        source="builtin",
        metadata={
            "category": "file",
            "operation": "read",
            "requires_permission_gate": True,
        },
    )

    registry.register(
        GlobTool(),
        source="builtin",
        metadata={
            "category": "file",
            "operation": "list",
            "requires_permission_gate": True,
        },
    )

    registry.register(
        GrepTool(),
        source="builtin",
        metadata={
            "category": "file",
            "operation": "search",
            "requires_permission_gate": True,
        },
    )

    registry.register(
        FileWriteTool(),
        source="builtin",
        metadata={
            "category": "file",
            "operation": "write",
            "requires_permission_gate": True,
            "requires_diff_preview": True,
        },
    )

    registry.register(
        FileEditTool(),
        source="builtin",
        metadata={
            "category": "file",
            "operation": "edit",
            "requires_permission_gate": True,
            "requires_diff_preview": True,
        },
    )

    registry.register(
        BashTool(),
        source="builtin",
        metadata={
            "category": "shell",
            "operation": "execute",
            "requires_permission_gate": True,
            "requires_command_safety_check": True,
        },
    )

    registry.register(
        PowerShellTool(),
        source="builtin",
        metadata={
            "category": "shell",
            "operation": "execute",
            "requires_permission_gate": True,
            "requires_command_safety_check": True,
        },
    )

    return registry

_DEFAULT_REGISTRY: ToolRegistry | None = None


def get_default_registry() -> ToolRegistry:
    """
    获取全局默认注册表。
    """
    global _DEFAULT_REGISTRY

    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = create_default_registry()

    return _DEFAULT_REGISTRY


def reset_default_registry() -> ToolRegistry:
    """
    重置全局默认注册表。

    测试时比较有用。
    """
    global _DEFAULT_REGISTRY

    _DEFAULT_REGISTRY = create_default_registry()
    return _DEFAULT_REGISTRY


async def demo() -> None:
    registry = create_default_registry()

    print("Registry:")
    print(json.dumps(registry.to_dict(), ensure_ascii=False, indent=2))

    print("\nTool names:")
    print(registry.list_names())

    print("\nTool list:")
    for tool in registry.list():
        print(f"- {tool.name}: {tool.description}")

    print("\nTool definitions:")
    print(json.dumps(registry.list_definitions(), ensure_ascii=False, indent=2))

    print("\nFind echo:")
    for tool in registry.find("echo"):
        print(f"- {tool.name}: {tool.description}")

    print("\nRun echo:")
    result = await registry.run_tool(
        "echo",
        {
            "text": "Hello from ToolRegistry.",
        },
        context=ToolExecutionContext(
            workspace_path=".",
            project_root=".",
            permission_mode="default",
        ),
    )

    print(result.model_dump_json(indent=2))

    print("\nRendered result:")
    print(registry.render_result(result))

    print("\nExecute existing ToolCall:")
    call = registry.create_call(
        "echo",
        {
            "text": "Hello from existing ToolCall.",
        },
    )

    result2 = await registry.execute_call(call)
    print(registry.render_result(result2))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())