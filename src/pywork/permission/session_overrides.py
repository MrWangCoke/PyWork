from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pywork.permission.audit import PermissionAuditUserAction
from pywork.permission.policy import PermissionDecision
from pywork.permission.risk import RiskLevel


class PermissionSessionOverrideAction(str, Enum):
    """本会话权限覆盖动作。"""

    ALLOW = "allow"
    DENY = "deny"


@dataclass(slots=True, frozen=True)
class PermissionSessionOverrideKey:
    """
    本会话权限覆盖 key。

    注意：
    Always Allow 不应该太宽。
    所以这里按 tool + action + target + risk 生成 key。

    例子：
    - file_write + write + src/utils/helper.py + high
    - bash + execute + uv run pytest tests + low
    - powershell + execute + Remove-Item demo.txt + critical
    """

    tool_name: str
    action: str | None
    target: str
    risk: str

    def to_tuple(self) -> tuple[str, str | None, str, str]:
        return (
            self.tool_name,
            self.action,
            self.target,
            self.risk,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PermissionSessionOverride:
    """本会话权限覆盖记录。"""

    key: PermissionSessionOverrideKey
    action: PermissionSessionOverrideAction
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PermissionGateState:
    """
    PermissionGate 的会话状态。

    目前只放 session 级 Always Allow / Always Deny。
    后面可以继续扩展：
    - 临时批准次数
    - 按目录批准
    - 按工具批准
    """

    def __init__(self) -> None:
        self.overrides: dict[
            tuple[str, str | None, str, str],
            PermissionSessionOverride,
        ] = {}

    def add_override(
        self,
        key: PermissionSessionOverrideKey,
        *,
        action: PermissionSessionOverrideAction,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionSessionOverride:
        override = PermissionSessionOverride(
            key=key,
            action=action,
            reason=reason,
            metadata=metadata or {},
        )

        self.overrides[key.to_tuple()] = override

        return override

    def add_always_allow(
        self,
        decision: PermissionDecision,
        *,
        rule_result: Any | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionSessionOverride:
        return self.add_override(
            create_session_override_key(
                decision,
                rule_result=rule_result,
            ),
            action=PermissionSessionOverrideAction.ALLOW,
            reason=reason,
            metadata=metadata,
        )

    def add_always_deny(
        self,
        decision: PermissionDecision,
        *,
        rule_result: Any | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionSessionOverride:
        return self.add_override(
            create_session_override_key(
                decision,
                rule_result=rule_result,
            ),
            action=PermissionSessionOverrideAction.DENY,
            reason=reason,
            metadata=metadata,
        )

    def get_override(
        self,
        decision: PermissionDecision,
        *,
        rule_result: Any | None = None,
    ) -> PermissionSessionOverride | None:
        key = create_session_override_key(
            decision,
            rule_result=rule_result,
        )

        return self.overrides.get(key.to_tuple())

    def clear(self) -> None:
        self.overrides.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "overrides": [
                {
                    "key": override.key.to_dict(),
                    "action": override.action.value,
                    "reason": override.reason,
                    "metadata": override.metadata,
                }
                for override in self.overrides.values()
            ]
        }


def normalize_tool_name(tool_name: str) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)

    return str(raw)


def normalize_risk_text(value: RiskLevel | str | Any) -> str:
    return enum_value(value).strip().lower()


def stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    except TypeError:
        return str(value)


def get_argument_target(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    """
    根据工具参数生成较保守的 target。

    - 文件工具按 path
    - shell 工具按完整 command
    - 其他工具按 arguments JSON
    """
    normalized_tool = normalize_tool_name(tool_name)

    if normalized_tool in {
        "file_read",
        "file_write",
        "file_edit",
    }:
        return str(arguments.get("path", ""))

    if normalized_tool == "glob":
        return str(
            arguments.get("path")
            or arguments.get("root")
            or arguments.get("cwd")
            or "."
        )

    if normalized_tool == "grep":
        return str(
            arguments.get("path")
            or arguments.get("directory")
            or arguments.get("cwd")
            or "."
        )

    if normalized_tool in {
        "bash",
        "powershell",
    }:
        return str(arguments.get("command", ""))

    return stable_json(arguments)


def create_session_override_key(
    decision: PermissionDecision,
    *,
    rule_result: Any | None = None,
) -> PermissionSessionOverrideKey:
    request = decision.request

    tool_name = normalize_tool_name(request.tool_name)
    target = get_argument_target(
        tool_name,
        request.arguments,
    )

    risk = normalize_risk_text(decision.risk)

    return PermissionSessionOverrideKey(
        tool_name=tool_name,
        action=request.action,
        target=target,
        risk=risk,
    )


def user_action_is_allow(user_action: PermissionAuditUserAction | str | Any) -> bool:
    value = enum_value(user_action)

    return value in {
        PermissionAuditUserAction.ALLOW.value,
        PermissionAuditUserAction.ALWAYS_ALLOW.value,
    }


def user_action_is_always_allow(
    user_action: PermissionAuditUserAction | str | Any,
) -> bool:
    return enum_value(user_action) == PermissionAuditUserAction.ALWAYS_ALLOW.value