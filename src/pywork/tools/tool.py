from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToolError(Exception):
    """
    工具体系基础异常。
    """

    pass


class ToolValidationError(ToolError):
    """
    工具参数校验失败。
    """

    pass


class ToolExecutionError(ToolError):
    """
    工具执行失败。
    """

    pass


@dataclass
class ToolExecutionContext:
    """
    工具执行上下文。

    后面会逐步扩展：
    - workspace_path：当前工作区
    - project_root：项目根目录
    - permission_mode：权限模式
    - session_id：会话 ID
    - metadata：额外信息
    """

    workspace_path: str = "."
    project_root: str = "."
    permission_mode: str = "default"
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTool(ABC):
    """
    PyWork 工具抽象基类。

    每个工具都应该提供：

    - name：工具名
    - description：工具描述
    - input_schema：输入参数 JSON Schema
    - risk_level：风险等级
    - execute()：真正执行工具
    - render_result()：把 ToolResult 渲染成适合展示给用户的文本
    """

    name: ClassVar[str] = ""
    description: ClassVar[str] = ""
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    risk_level: ClassVar[ToolRiskLevel | str] = ToolRiskLevel.LOW

    def __init__(self) -> None:
        self.validate_tool_definition()

    def validate_tool_definition(self) -> None:
        """
        校验工具自身定义是否完整。
        """
        if not self.name or not self.name.strip():
            raise ToolValidationError("Tool name cannot be empty")

        if not self.description or not self.description.strip():
            raise ToolValidationError(f"Tool {self.name!r} description cannot be empty")

        if not isinstance(self.input_schema, dict):
            raise ToolValidationError(f"Tool {self.name!r} input_schema must be a dict")

        if self.input_schema.get("type") != "object":
            raise ToolValidationError(
                f"Tool {self.name!r} input_schema.type must be 'object'"
            )

        try:
            ToolRiskLevel(self.risk_level)
        except ValueError as exc:
            raise ToolValidationError(
                f"Tool {self.name!r} has invalid risk_level: {self.risk_level!r}"
            ) from exc

    def get_risk_level(self) -> ToolRiskLevel:
        return ToolRiskLevel(self.risk_level)

    def get_definition(self) -> dict[str, Any]:
        """
        返回工具定义。

        后面可以给 LLM Provider 使用，用于 function calling / tool calling。
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "risk_level": self.get_risk_level().value,
        }

    def create_call(
        self,
        arguments: dict[str, Any] | None = None,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolCall:
        """
        基于当前工具创建 ToolCall。
        """
        return ToolCall(
            tool_name=self.name,
            arguments=arguments or {},
            risk_level=self.get_risk_level(),
            metadata=metadata or {},
        )

    def validate_call(self, call: ToolCall) -> None:
        """
        校验 ToolCall 是否适合当前工具执行。
        """
        if call.tool_name != self.name:
            raise ToolValidationError(
                f"ToolCall target mismatch: expected {self.name!r}, got {call.tool_name!r}"
            )

        self.validate_arguments(call.arguments)

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """
        简单校验 arguments。

        这里先做基础校验：
        - arguments 必须是 dict
        - required 字段必须存在
        - 简单检查 JSON Schema type

        更完整的 JSON Schema 校验后面可以接 jsonschema 库。
        """
        if not isinstance(arguments, dict):
            raise ToolValidationError("Tool arguments must be a dict")

        required = self.input_schema.get("required", [])
        properties = self.input_schema.get("properties", {})

        for key in required:
            if key not in arguments:
                raise ToolValidationError(
                    f"Missing required argument {key!r} for tool {self.name!r}"
                )

        for key, value in arguments.items():
            schema = properties.get(key)

            if not schema:
                continue

            expected_type = schema.get("type")

            if expected_type is None:
                continue

            if not self._matches_json_schema_type(value, expected_type):
                raise ToolValidationError(
                    f"Invalid type for argument {key!r}: "
                    f"expected {expected_type!r}, got {type(value).__name__!r}"
                )

    def _matches_json_schema_type(
        self,
        value: Any,
        expected_type: str | list[str],
    ) -> bool:
        if isinstance(expected_type, list):
            return any(self._matches_json_schema_type(value, item) for item in expected_type)

        if expected_type == "string":
            return isinstance(value, str)

        if expected_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        if expected_type == "number":
            return isinstance(value, int | float) and not isinstance(value, bool)

        if expected_type == "boolean":
            return isinstance(value, bool)

        if expected_type == "object":
            return isinstance(value, dict)

        if expected_type == "array":
            return isinstance(value, list)

        if expected_type == "null":
            return value is None

        return True

    async def run(
        self,
        call: ToolCall,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """
        工具执行入口。

        这个方法负责：
        1. 校验 ToolCall
        2. 调用 execute()
        3. 捕获异常
        4. 统一返回 ToolResult

        外部一般调用 run()，而不是直接调用 execute()。
        """
        started_at = utc_now()

        try:
            self.validate_call(call)
            result = await self.execute(call, context or ToolExecutionContext())

            if result.call_id != call.call_id:
                raise ToolExecutionError(
                    f"ToolResult call_id mismatch: expected {call.call_id!r}, "
                    f"got {result.call_id!r}"
                )

            if result.tool_name != self.name:
                raise ToolExecutionError(
                    f"ToolResult tool_name mismatch: expected {self.name!r}, "
                    f"got {result.tool_name!r}"
                )

            return result

        except Exception as exc:
            finished_at = utc_now()

            return ToolResult.error_result(
                call=call,
                error=str(exc),
                content=f"Tool {self.name!r} failed: {exc}",
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "exception_type": type(exc).__name__,
                },
            )

    @abstractmethod
    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        """
        真正执行工具。

        子类必须实现。
        """
        raise NotImplementedError

    def render_result(self, result: ToolResult) -> str:
        """
        把 ToolResult 渲染成用户能看的文本。

        子类可以重写这个方法。
        """
        if result.success:
            if result.content:
                return result.content

            if result.data:
                return json.dumps(
                    result.data,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )

            return "Tool executed successfully."

        if result.content:
            return result.content

        if result.error:
            return f"Tool failed: {result.error}"

        return "Tool failed."

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(name={self.name!r}, risk_level={self.get_risk_level().value!r})"
        )


class EchoTool(BaseTool):
    """
    测试用工具。

    正式工具后面会单独放到 tools/builtin/ 里面。
    这里暂时用它验证 BaseTool 是否能跑通。
    """

    name = "echo"
    description = "Echo input text back to the user."
    risk_level = ToolRiskLevel.SAFE
    input_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to echo.",
            }
        },
        "required": ["text"],
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        text = call.arguments["text"]

        return ToolResult.success_result(
            call=call,
            content=text,
            data={
                "text": text,
                "workspace_path": context.workspace_path,
            },
        )


async def demo() -> None:
    tool = EchoTool()

    call = tool.create_call(
        {
            "text": "Hello from PyWork tool system.",
        }
    )

    result = await tool.run(
        call,
        ToolExecutionContext(
            workspace_path=".",
            project_root=".",
            permission_mode="default",
        ),
    )

    print("Tool definition:")
    print(json.dumps(tool.get_definition(), ensure_ascii=False, indent=2))

    print("\nTool call:")
    print(call.model_dump_json(indent=2))

    print("\nTool result:")
    print(result.model_dump_json(indent=2))

    print("\nRendered result:")
    print(tool.render_result(result))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())