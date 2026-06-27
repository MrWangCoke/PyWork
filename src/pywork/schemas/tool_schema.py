from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToolRiskLevel(str, Enum):
    """
    工具风险等级。

    后面权限系统会根据 risk_level 判断是否需要用户确认。
    """

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    DANGEROUS = "dangerous"


class ToolResultStatus(str, Enum):
    """
    工具执行状态。
    """

    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_tool_call_id() -> str:
    return f"toolcall_{uuid4().hex}"


def new_tool_result_id() -> str:
    return f"toolresult_{uuid4().hex}"


class ToolCall(BaseModel):
    """
    一次工具调用。

    例子：
        ToolCall(
            tool_name="bash",
            arguments={"command": "git status"},
            risk_level="medium",
        )
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    call_id: str = Field(default_factory=new_tool_call_id)
    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tool_name")
    @classmethod
    def validate_tool_name(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("tool_name cannot be empty")

        return value

    @field_validator("call_id")
    @classmethod
    def validate_call_id(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("call_id cannot be empty")

        return value

    def to_log_dict(self) -> dict[str, Any]:
        """
        用于日志输出的简洁结构。
        """
        return {
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "risk_level": self.risk_level,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


class ToolResult(BaseModel):
    """
    一次工具执行结果。

    success=True 时：
        content/data 通常有值，error 通常为空。

    success=False 时：
        error 通常有值，content 可以是给用户看的错误说明。
    """

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
        validate_assignment=True,
    )

    result_id: str = Field(default_factory=new_tool_result_id)
    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)

    status: ToolResultStatus = ToolResultStatus.SUCCESS
    success: bool = True

    content: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)
    duration_ms: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("result_id", "call_id", "tool_name")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("value cannot be empty")

        return value

    @field_validator("duration_ms")
    @classmethod
    def validate_duration_ms(cls, value: int) -> int:
        if value < 0:
            raise ValueError("duration_ms cannot be negative")

        return value

    @model_validator(mode="after")
    def sync_status_and_success(self) -> ToolResult:
        """
        保证 status 和 success 不互相矛盾。

        注意：
        这里不能直接写 self.success = True。
        因为 model_config 里开启了 validate_assignment=True，
        普通赋值会再次触发 Pydantic 校验，导致递归。
        """
        if self.status == ToolResultStatus.SUCCESS:
            object.__setattr__(self, "success", True)

        if self.status in {
            ToolResultStatus.ERROR,
            ToolResultStatus.CANCELLED,
        }:
            object.__setattr__(self, "success", False)

        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot be earlier than started_at")

        if self.duration_ms == 0:
            delta = self.finished_at - self.started_at
            duration_ms = max(0, int(delta.total_seconds() * 1000))
            object.__setattr__(self, "duration_ms", duration_ms)

        return self

    @classmethod
    def success_result(
        cls,
        *,
        call: ToolCall,
        content: str = "",
        data: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        start = started_at or utc_now()
        finish = finished_at or utc_now()

        return cls(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.SUCCESS,
            success=True,
            content=content,
            data=data or {},
            error=None,
            started_at=start,
            finished_at=finish,
            metadata=metadata or {},
        )

    @classmethod
    def error_result(
        cls,
        *,
        call: ToolCall,
        error: str,
        content: str = "",
        data: dict[str, Any] | None = None,
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        start = started_at or utc_now()
        finish = finished_at or utc_now()

        return cls(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.ERROR,
            success=False,
            content=content or error,
            data=data or {},
            error=error,
            started_at=start,
            finished_at=finish,
            metadata=metadata or {},
        )

    @classmethod
    def cancelled_result(
        cls,
        *,
        call: ToolCall,
        reason: str = "cancelled",
        started_at: datetime | None = None,
        finished_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolResult:
        start = started_at or utc_now()
        finish = finished_at or utc_now()

        return cls(
            call_id=call.call_id,
            tool_name=call.tool_name,
            status=ToolResultStatus.CANCELLED,
            success=False,
            content=reason,
            error=reason,
            started_at=start,
            finished_at=finish,
            metadata=metadata or {},
        )

    def to_log_dict(self) -> dict[str, Any]:
        """
        用于日志输出的简洁结构。
        """
        return {
            "result_id": self.result_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "status": self.status,
            "success": self.success,
            "content": self.content,
            "data": self.data,
            "error": self.error,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


def create_tool_call(
    *,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    risk_level: ToolRiskLevel | str = ToolRiskLevel.LOW,
    metadata: dict[str, Any] | None = None,
) -> ToolCall:
    return ToolCall(
        tool_name=tool_name,
        arguments=arguments or {},
        risk_level=risk_level,
        metadata=metadata or {},
    )


def main() -> int:
    call = create_tool_call(
        tool_name="bash",
        arguments={
            "command": "git status",
        },
        risk_level=ToolRiskLevel.MEDIUM,
    )

    result = ToolResult.success_result(
        call=call,
        content="On branch main\nnothing to commit, working tree clean",
        data={
            "exit_code": 0,
        },
    )

    print("ToolCall:")
    print(call.model_dump_json(indent=2))

    print("\nToolResult:")
    print(result.model_dump_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())