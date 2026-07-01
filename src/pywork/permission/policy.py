from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from pywork.permission.mode import (
    DEFAULT_PERMISSION_MODE,
    PermissionMode,
    get_permission_mode_info,
    mode_auto_allows_risk,
    mode_denies_risk,
    normalize_permission_mode,
)
from pywork.permission.risk import (
    DEFAULT_UNKNOWN_TOOL_RISK,
    RiskLevel,
    get_risk_info,
    get_tool_risk,
    normalize_risk_level,
    risk_requires_elevated_confirmation,
)


class PermissionDecisionType(str, Enum):
    """权限决策结果。"""

    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"
    ASK_ELEVATED = "ask_elevated"


@dataclass(slots=True, frozen=True)
class PermissionRequest:
    """
    一次权限判断请求。

    tool_name:
        工具名，例如 file_read / file_write / bash。

    action:
        操作名，例如 read / write / edit / execute。

    mode:
        当前权限模式。

    risk:
        当前操作风险等级。可以不传，不传时根据 tool_name 推断。

    arguments:
        工具参数，用于后续 TUI 展示确认信息。

    call_id:
        ToolCall 的 call_id。

    reason:
        可选原因，例如 “LLM requested this tool call”。

    metadata:
        额外信息。
    """

    tool_name: str
    action: str | None = None
    mode: PermissionMode | str = DEFAULT_PERMISSION_MODE
    risk: RiskLevel | str | Any | None = None

    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def normalized_tool_name(self) -> str:
        return self.tool_name.strip().lower().replace("-", "_")

    def normalized_action(self) -> str | None:
        if self.action is None:
            return None

        return self.action.strip().lower().replace("-", "_")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)

        mode = normalize_permission_mode(self.mode)
        data["mode"] = mode.value

        if self.risk is not None:
            data["risk"] = normalize_risk_level(
                self.risk,
                default=DEFAULT_UNKNOWN_TOOL_RISK,
            ).value

        return data


@dataclass(slots=True, frozen=True)
class PermissionDecision:
    """
    一次权限决策结果。

    decision:
        allow / deny / ask / ask_elevated。

    allowed:
        是否可以直接执行。

    reason:
        给日志 / TUI 展示用的人类可读原因。
    """

    decision: PermissionDecisionType
    request: PermissionRequest

    mode: PermissionMode
    risk: RiskLevel

    reason: str
    requires_confirmation: bool = False
    requires_elevated_confirmation: bool = False

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision == PermissionDecisionType.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision == PermissionDecisionType.DENY

    @property
    def should_ask(self) -> bool:
        return self.decision in {
            PermissionDecisionType.ASK,
            PermissionDecisionType.ASK_ELEVATED,
        }

    @property
    def is_elevated(self) -> bool:
        return self.decision == PermissionDecisionType.ASK_ELEVATED

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "allowed": self.allowed,
            "denied": self.denied,
            "should_ask": self.should_ask,
            "requires_confirmation": self.requires_confirmation,
            "requires_elevated_confirmation": self.requires_elevated_confirmation,
            "mode": self.mode.value,
            "risk": self.risk.value,
            "reason": self.reason,
            "request": self.request.to_dict(),
            "metadata": self.metadata,
        }


@dataclass(slots=True, frozen=True)
class PermissionPolicyConfig:
    """
    权限策略配置。

    tool_risk_overrides:
        覆盖默认工具风险等级。

    always_allow_tools:
        总是允许的工具。

    always_deny_tools:
        总是拒绝的工具。

    always_ask_tools:
        总是普通确认的工具。

    always_ask_elevated_tools:
        总是高风险确认的工具。
    """

    default_mode: PermissionMode = DEFAULT_PERMISSION_MODE

    tool_risk_overrides: dict[str, RiskLevel | str] = field(default_factory=dict)

    always_allow_tools: set[str] = field(default_factory=set)
    always_deny_tools: set[str] = field(default_factory=set)
    always_ask_tools: set[str] = field(default_factory=set)
    always_ask_elevated_tools: set[str] = field(default_factory=set)


def normalize_tool_name(tool_name: str) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def normalize_tool_set(values: set[str] | list[str] | tuple[str, ...]) -> set[str]:
    return {
        normalize_tool_name(value)
        for value in values
    }


class PermissionPolicy:
    """
    权限策略引擎。

    它的核心规则：

    1. bypass 模式：
       直接 allow。

    2. always_deny_tools：
       直接 deny。

    3. always_allow_tools：
       直接 allow。

    4. readonly / plan：
       safe / low 允许，medium 及以上 deny。

    5. default：
       safe / low allow。
       medium / high ask。
       critical ask_elevated。

    6. accept_edits：
       safe / low / medium / high allow。
       critical ask_elevated。

    7. critical 风险：
       默认 ask_elevated。
    """

    def __init__(
        self,
        config: PermissionPolicyConfig | None = None,
    ) -> None:
        self.config = config or PermissionPolicyConfig()

        self.always_allow_tools = normalize_tool_set(
            self.config.always_allow_tools,
        )
        self.always_deny_tools = normalize_tool_set(
            self.config.always_deny_tools,
        )
        self.always_ask_tools = normalize_tool_set(
            self.config.always_ask_tools,
        )
        self.always_ask_elevated_tools = normalize_tool_set(
            self.config.always_ask_elevated_tools,
        )

    def resolve_mode(
        self,
        request: PermissionRequest,
    ) -> PermissionMode:
        return normalize_permission_mode(
            request.mode,
            default=self.config.default_mode,
        )

    def resolve_risk(
        self,
        request: PermissionRequest,
    ) -> RiskLevel:
        if request.risk is not None:
            return normalize_risk_level(
                request.risk,
                default=DEFAULT_UNKNOWN_TOOL_RISK,
            )

        return get_tool_risk(
            request.tool_name,
            default=DEFAULT_UNKNOWN_TOOL_RISK,
            extra_mapping=self.config.tool_risk_overrides,
        )

    def evaluate(
        self,
        request: PermissionRequest,
    ) -> PermissionDecision:
        tool_name = request.normalized_tool_name()
        mode = self.resolve_mode(request)
        risk = self.resolve_risk(request)

        mode_info = get_permission_mode_info(mode)
        risk_info = get_risk_info(risk)

        base_metadata = {
            "tool_name": tool_name,
            "action": request.normalized_action(),
            "mode_label": mode_info.label,
            "risk_label": risk_info.label,
            "risk_score": risk_info.score,
        }

        if mode == PermissionMode.BYPASS:
            return self._allow(
                request,
                mode=mode,
                risk=risk,
                reason="permission mode is bypass",
                metadata=base_metadata,
            )

        if tool_name in self.always_deny_tools:
            return self._deny(
                request,
                mode=mode,
                risk=risk,
                reason=f"tool is explicitly denied: {tool_name}",
                metadata=base_metadata,
            )

        if tool_name in self.always_allow_tools:
            return self._allow(
                request,
                mode=mode,
                risk=risk,
                reason=f"tool is explicitly allowed: {tool_name}",
                metadata=base_metadata,
            )

        if tool_name in self.always_ask_elevated_tools:
            return self._ask_elevated(
                request,
                mode=mode,
                risk=risk,
                reason=f"tool requires elevated confirmation: {tool_name}",
                metadata=base_metadata,
            )

        if tool_name in self.always_ask_tools:
            return self._ask(
                request,
                mode=mode,
                risk=risk,
                reason=f"tool requires confirmation: {tool_name}",
                metadata=base_metadata,
            )

        if mode_denies_risk(
            mode,
            risk,
        ):
            return self._deny(
                request,
                mode=mode,
                risk=risk,
                reason=(
                    f"{mode.value} mode denies {risk.value} risk operations"
                ),
                metadata=base_metadata,
            )

        if mode_auto_allows_risk(
            mode,
            risk,
        ):
            return self._allow(
                request,
                mode=mode,
                risk=risk,
                reason=(
                    f"{mode.value} mode automatically allows "
                    f"{risk.value} risk operations"
                ),
                metadata=base_metadata,
            )

        if risk_requires_elevated_confirmation(risk):
            return self._ask_elevated(
                request,
                mode=mode,
                risk=risk,
                reason=(
                    f"{risk.value} risk operation requires elevated confirmation"
                ),
                metadata=base_metadata,
            )

        return self._ask(
            request,
            mode=mode,
            risk=risk,
            reason=f"{risk.value} risk operation requires confirmation",
            metadata=base_metadata,
        )

    def evaluate_tool(
        self,
        tool_name: str,
        *,
        action: str | None = None,
        mode: PermissionMode | str = DEFAULT_PERMISSION_MODE,
        risk: RiskLevel | str | Any | None = None,
        arguments: dict[str, Any] | None = None,
        call_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        request = PermissionRequest(
            tool_name=tool_name,
            action=action,
            mode=mode,
            risk=risk,
            arguments=arguments or {},
            call_id=call_id,
            reason=reason,
            metadata=metadata or {},
        )

        return self.evaluate(request)

    def evaluate_tool_call(
        self,
        tool_call: Any,
        *,
        mode: PermissionMode | str = DEFAULT_PERMISSION_MODE,
        risk: RiskLevel | str | Any | None = None,
        action: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        """
        从 ToolCall-like 对象创建权限请求。

        兼容：
            tool_call.tool_name
            tool_call.name
            tool_call.arguments
            tool_call.call_id
        """
        tool_name = (
            getattr(tool_call, "tool_name", None)
            or getattr(tool_call, "name", None)
        )

        if not tool_name:
            raise ValueError("tool_call has no tool_name or name")

        arguments = getattr(
            tool_call,
            "arguments",
            {},
        )

        call_id = getattr(
            tool_call,
            "call_id",
            None,
        )

        return self.evaluate_tool(
            str(tool_name),
            action=action,
            mode=mode,
            risk=risk,
            arguments=dict(arguments or {}),
            call_id=call_id,
            reason=reason,
            metadata=metadata or {},
        )

    def _allow(
        self,
        request: PermissionRequest,
        *,
        mode: PermissionMode,
        risk: RiskLevel,
        reason: str,
        metadata: dict[str, Any],
    ) -> PermissionDecision:
        return PermissionDecision(
            decision=PermissionDecisionType.ALLOW,
            request=request,
            mode=mode,
            risk=risk,
            reason=reason,
            requires_confirmation=False,
            requires_elevated_confirmation=False,
            metadata=metadata,
        )

    def _deny(
        self,
        request: PermissionRequest,
        *,
        mode: PermissionMode,
        risk: RiskLevel,
        reason: str,
        metadata: dict[str, Any],
    ) -> PermissionDecision:
        return PermissionDecision(
            decision=PermissionDecisionType.DENY,
            request=request,
            mode=mode,
            risk=risk,
            reason=reason,
            requires_confirmation=False,
            requires_elevated_confirmation=False,
            metadata=metadata,
        )

    def _ask(
        self,
        request: PermissionRequest,
        *,
        mode: PermissionMode,
        risk: RiskLevel,
        reason: str,
        metadata: dict[str, Any],
    ) -> PermissionDecision:
        return PermissionDecision(
            decision=PermissionDecisionType.ASK,
            request=request,
            mode=mode,
            risk=risk,
            reason=reason,
            requires_confirmation=True,
            requires_elevated_confirmation=False,
            metadata=metadata,
        )

    def _ask_elevated(
        self,
        request: PermissionRequest,
        *,
        mode: PermissionMode,
        risk: RiskLevel,
        reason: str,
        metadata: dict[str, Any],
    ) -> PermissionDecision:
        return PermissionDecision(
            decision=PermissionDecisionType.ASK_ELEVATED,
            request=request,
            mode=mode,
            risk=risk,
            reason=reason,
            requires_confirmation=True,
            requires_elevated_confirmation=True,
            metadata=metadata,
        )


def create_default_permission_policy() -> PermissionPolicy:
    return PermissionPolicy()


def evaluate_permission(
    tool_name: str,
    *,
    action: str | None = None,
    mode: PermissionMode | str = DEFAULT_PERMISSION_MODE,
    risk: RiskLevel | str | Any | None = None,
    arguments: dict[str, Any] | None = None,
    call_id: str | None = None,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
    policy: PermissionPolicy | None = None,
) -> PermissionDecision:
    """
    快捷权限判断函数。

    后面 runtime graph 里可以直接用：

        decision = evaluate_permission(
            tool_call.tool_name,
            mode=current_mode,
            arguments=tool_call.arguments,
        )
    """
    permission_policy = policy or create_default_permission_policy()

    return permission_policy.evaluate_tool(
        tool_name,
        action=action,
        mode=mode,
        risk=risk,
        arguments=arguments or {},
        call_id=call_id,
        reason=reason,
        metadata=metadata or {},
    )


def render_permission_decision(decision: PermissionDecision) -> str:
    """渲染权限决策结果，给日志 / ToolLog 用。"""
    return (
        f"{decision.decision.value}: "
        f"tool={decision.request.tool_name}, "
        f"mode={decision.mode.value}, "
        f"risk={decision.risk.value}, "
        f"reason={decision.reason}"
    )


def demo() -> None:
    policy = create_default_permission_policy()

    examples = [
        ("file_read", "default"),
        ("file_write", "default"),
        ("bash", "default"),
        ("file_write", "readonly"),
        ("file_write", "accept_edits"),
        ("bash", "accept_edits"),
        ("bash", "bypass"),
    ]

    for tool_name, mode in examples:
        decision = policy.evaluate_tool(
            tool_name,
            mode=mode,
        )
        print(render_permission_decision(decision))


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())