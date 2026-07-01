from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pywork.permission.audit import PermissionAuditLog, PermissionAuditRecord
from pywork.permission.bash_permissions import (
    BashPermissionResult,
    evaluate_bash_permission,
)
from pywork.permission.file_permissions import (
    FilePermissionResult,
    evaluate_file_permission,
)
from pywork.permission.mode import DEFAULT_PERMISSION_MODE, PermissionMode
from pywork.permission.policy import (
    PermissionDecision,
    PermissionDecisionType,
    PermissionPolicy,
    PermissionRequest,
)
from pywork.permission.powershell_permissions import (
    PowerShellPermissionResult,
    evaluate_powershell_permission,
)
from pywork.permission.risk import RiskLevel, risk_score
from pywork.permission.session_overrides import (
    PermissionGateState,
    PermissionSessionOverrideAction,
)


FILE_TOOL_OPERATIONS: dict[str, str] = {
    "file_read": "read",
    "glob": "list",
    "grep": "search",
    "file_write": "write",
    "file_edit": "edit",
}


SHELL_TOOL_NAMES: set[str] = {
    "bash",
    "powershell",
}


@dataclass(slots=True, frozen=True)
class PermissionGateRuleResult:
    """
    某个具体规则检查器的结果。

    例如：
    - file_permissions.py 的文件路径检查
    - bash_permissions.py 的命令安全检查
    - powershell_permissions.py 的命令安全检查

    hard_decision:
        True 表示这个规则结果必须参与最终决策。
        比如 shell 的 unknown / rm / Remove-Item，或者文件敏感路径。
    """

    source: str
    decision: PermissionDecisionType
    risk: RiskLevel
    reason: str

    matched_rules: tuple[str, ...] = ()
    hard_decision: bool = False
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
    def requires_elevated_confirmation(self) -> bool:
        return self.decision == PermissionDecisionType.ASK_ELEVATED

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["decision"] = self.decision.value
        data["risk"] = self.risk.value
        return data


@dataclass(slots=True, frozen=True)
class PermissionGateResult:
    """
    PermissionGate 最终输出。

    decision:
        最终权限决策。Runtime 后面只看这个即可。

    policy_decision:
        policy.py 的基础决策。

    rule_result:
        具体规则层结果，可能来自 file/bash/powershell，也可能是 None。

    audit_record:
        本次写入的审计记录。
    """

    decision: PermissionDecision
    policy_decision: PermissionDecision
    rule_result: PermissionGateRuleResult | None = None
    audit_record: PermissionAuditRecord | None = None

    @property
    def allowed(self) -> bool:
        return self.decision.allowed

    @property
    def denied(self) -> bool:
        return self.decision.denied

    @property
    def should_ask(self) -> bool:
        return self.decision.should_ask

    @property
    def requires_elevated_confirmation(self) -> bool:
        return self.decision.requires_elevated_confirmation

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "policy_decision": self.policy_decision.to_dict(),
            "rule_result": self.rule_result.to_dict() if self.rule_result else None,
            "audit_record": (
                self.audit_record.to_dict()
                if self.audit_record is not None
                else None
            ),
            "allowed": self.allowed,
            "denied": self.denied,
            "should_ask": self.should_ask,
            "requires_elevated_confirmation": self.requires_elevated_confirmation,
        }


class PermissionGateError(Exception):
    """PermissionGate 基础异常。"""


class PermissionGateValidationError(PermissionGateError):
    """PermissionGate 参数异常。"""


def normalize_tool_name(tool_name: str) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def stronger_decision(
    left: PermissionDecisionType,
    right: PermissionDecisionType,
) -> PermissionDecisionType:
    """
    选择更严格的决策。

    allow < ask < ask_elevated < deny
    """
    order = {
        PermissionDecisionType.ALLOW: 0,
        PermissionDecisionType.ASK: 1,
        PermissionDecisionType.ASK_ELEVATED: 2,
        PermissionDecisionType.DENY: 3,
    }

    return left if order[left] >= order[right] else right


def max_risk_level(
    left: RiskLevel,
    right: RiskLevel,
) -> RiskLevel:
    return left if risk_score(left) >= risk_score(right) else right


def get_tool_call_name(tool_call: Any) -> str:
    tool_name = (
        getattr(tool_call, "tool_name", None)
        or getattr(tool_call, "name", None)
    )

    if not tool_name:
        raise PermissionGateValidationError("tool call has no tool_name or name")

    return str(tool_name)


def get_tool_call_arguments(tool_call: Any) -> dict[str, Any]:
    arguments = getattr(
        tool_call,
        "arguments",
        {},
    )

    if arguments is None:
        return {}

    if not isinstance(arguments, dict):
        return dict(arguments)

    return dict(arguments)


def get_tool_call_id(tool_call: Any) -> str | None:
    call_id = getattr(
        tool_call,
        "call_id",
        None,
    )

    if call_id is None:
        return None

    return str(call_id)


def infer_action_for_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> str | None:
    normalized_name = normalize_tool_name(tool_name)

    if normalized_name in FILE_TOOL_OPERATIONS:
        return FILE_TOOL_OPERATIONS[normalized_name]

    if normalized_name in SHELL_TOOL_NAMES:
        return "execute"

    return None


def infer_file_path_for_tool(
    tool_name: str,
    arguments: dict[str, Any],
) -> str:
    normalized_name = normalize_tool_name(tool_name)

    if normalized_name in {
        "file_read",
        "file_write",
        "file_edit",
    }:
        path = arguments.get("path")

        if path is None:
            raise PermissionGateValidationError(
                f"{normalized_name} requires path argument"
            )

        return str(path)

    if normalized_name == "glob":
        return str(
            arguments.get("path")
            or arguments.get("root")
            or arguments.get("cwd")
            or "."
        )

    if normalized_name == "grep":
        return str(
            arguments.get("path")
            or arguments.get("directory")
            or arguments.get("cwd")
            or "."
        )

    return "."


def file_permission_to_gate_rule(
    result: FilePermissionResult,
) -> PermissionGateRuleResult:
    """
    文件规则结果转成 Gate 规则结果。

    普通写文件 ask 是 soft：
        accept_edits 模式可以自动允许。

    deny / ask_elevated 是 hard：
        敏感文件、保护目录、workspace 外路径必须影响最终决策。
    """
    hard_decision = result.decision in {
        PermissionDecisionType.DENY,
        PermissionDecisionType.ASK_ELEVATED,
    }

    return PermissionGateRuleResult(
        source="file_permissions",
        decision=result.decision,
        risk=result.risk,
        reason=result.reason,
        matched_rules=result.matched_rules,
        hard_decision=hard_decision,
        metadata={
            "operation": result.operation.value,
            "path": result.path,
            "absolute_path": result.absolute_path,
            "target_path": result.target_path,
            "absolute_target_path": result.absolute_target_path,
        },
    )


def bash_permission_to_gate_rule(
    result: BashPermissionResult,
) -> PermissionGateRuleResult:
    """
    Bash 规则结果转成 Gate 规则结果。

    shell 规则里，只要不是 allow，就认为是 hard。
    因为 shell 命令有副作用，不能让 accept_edits 之类的文件模式放行它。
    """
    return PermissionGateRuleResult(
        source="bash_permissions",
        decision=result.decision,
        risk=result.risk,
        reason=result.reason,
        matched_rules=result.matched_rules,
        hard_decision=result.decision != PermissionDecisionType.ALLOW,
        metadata={
            "command": result.command,
            "executable": result.executable,
            "tokens": list(result.tokens),
        },
    )


def powershell_permission_to_gate_rule(
    result: PowerShellPermissionResult,
) -> PermissionGateRuleResult:
    """
    PowerShell 规则结果转成 Gate 规则结果。

    shell 规则里，只要不是 allow，就认为是 hard。
    """
    return PermissionGateRuleResult(
        source="powershell_permissions",
        decision=result.decision,
        risk=result.risk,
        reason=result.reason,
        matched_rules=result.matched_rules,
        hard_decision=result.decision != PermissionDecisionType.ALLOW,
        metadata={
            "command": result.command,
            "executable": result.executable,
            "canonical_executable": result.canonical_executable,
            "tokens": list(result.tokens),
        },
    )


class PermissionGate:
    """
    Runtime 工具执行前的统一权限入口。

    核心职责：
    - 根据工具名调用具体规则检查器
    - 根据具体规则得到更准确的 risk
    - 交给 PermissionPolicy 结合 permission mode 做最终判断
    - 合并细规则和基础策略
    - 写入 audit 日志
    """

    def __init__(
        self,
        workspace_path: str | Path,
        *,
        policy: PermissionPolicy | None = None,
        audit_log: PermissionAuditLog | None = None,
        audit_enabled: bool = True,
        session_id: str | None = None,
        session_state: PermissionGateState | None = None,
    ) -> None:
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.policy = policy or PermissionPolicy()
        self.audit_enabled = audit_enabled
        self.session_id = session_id
        self.session_state = session_state or PermissionGateState()

        if audit_log is not None:
            self.audit_log = audit_log
        elif audit_enabled:
            self.audit_log = PermissionAuditLog(self.workspace_path)
        else:
            self.audit_log = None

    def check(
        self,
        tool_call: Any,
        *,
        mode: PermissionMode | str = DEFAULT_PERMISSION_MODE,
        action: str | None = None,
        risk: RiskLevel | str | Any | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionGateResult:
        """
        检查 ToolCall-like 对象。

        兼容：
        - tool_call.tool_name
        - tool_call.name
        - tool_call.arguments
        - tool_call.call_id
        """
        tool_name = get_tool_call_name(tool_call)
        arguments = get_tool_call_arguments(tool_call)
        call_id = get_tool_call_id(tool_call)

        return self.check_tool(
            tool_name,
            arguments=arguments,
            mode=mode,
            action=action,
            risk=risk,
            call_id=call_id,
            session_id=session_id,
            metadata=metadata,
        )

    def check_tool(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any] | None = None,
        mode: PermissionMode | str = DEFAULT_PERMISSION_MODE,
        action: str | None = None,
        risk: RiskLevel | str | Any | None = None,
        call_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionGateResult:
        """
        根据工具名和参数做权限检查。

        这个函数不执行工具，只返回权限结果。
        """
        normalized_name = normalize_tool_name(tool_name)
        arguments = dict(arguments or {})
        effective_action = action or infer_action_for_tool(
            normalized_name,
            arguments,
        )

        rule_result = self.evaluate_specific_rule(
            normalized_name,
            arguments=arguments,
            action=effective_action,
            metadata=metadata,
        )

        effective_risk = risk
        if rule_result is not None:
            effective_risk = rule_result.risk

        policy_decision = self.policy.evaluate_tool(
            normalized_name,
            action=effective_action,
            mode=mode,
            risk=effective_risk,
            arguments=arguments,
            call_id=call_id,
            metadata=metadata or {},
        )

        final_decision = self.combine_decisions(
            policy_decision,
            rule_result,
        )

        final_decision = self.apply_session_override(
            final_decision,
            rule_result,
        )

        audit_record = self.record_policy_decision(
            final_decision,
            session_id=session_id,
            metadata={
                "policy_decision": policy_decision.decision.value,
                "policy_reason": policy_decision.reason,
                "rule_result": (
                    rule_result.to_dict()
                    if rule_result is not None
                    else None
                ),
            },
        )

        return PermissionGateResult(
            decision=final_decision,
            policy_decision=policy_decision,
            rule_result=rule_result,
            audit_record=audit_record,
        )

    def evaluate_specific_rule(
        self,
        tool_name: str,
        *,
        arguments: dict[str, Any],
        action: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionGateRuleResult | None:
        """
        调用具体工具规则。

        file_read / glob / grep / file_write / file_edit:
            file_permissions.py

        bash:
            bash_permissions.py

        powershell:
            powershell_permissions.py
        """
        normalized_name = normalize_tool_name(tool_name)

        if normalized_name in FILE_TOOL_OPERATIONS:
            operation = FILE_TOOL_OPERATIONS[normalized_name]
            path = infer_file_path_for_tool(
                normalized_name,
                arguments,
            )

            target_path = (
                arguments.get("target_path")
                or arguments.get("to")
                or arguments.get("destination")
            )

            result = evaluate_file_permission(
                path,
                operation=operation,
                workspace_path=self.workspace_path,
                target_path=target_path,
                metadata=metadata or {},
            )

            return file_permission_to_gate_rule(result)

        if normalized_name == "bash":
            command = arguments.get("command")

            if command is None:
                raise PermissionGateValidationError("bash requires command argument")

            result = evaluate_bash_permission(
                str(command),
                cwd=arguments.get("cwd"),
                metadata=metadata or {},
            )

            return bash_permission_to_gate_rule(result)

        if normalized_name == "powershell":
            command = arguments.get("command")

            if command is None:
                raise PermissionGateValidationError(
                    "powershell requires command argument"
                )

            result = evaluate_powershell_permission(
                str(command),
                cwd=arguments.get("cwd"),
                metadata=metadata or {},
            )

            return powershell_permission_to_gate_rule(result)

        return None

    def combine_decisions(
        self,
        policy_decision: PermissionDecision,
        rule_result: PermissionGateRuleResult | None,
    ) -> PermissionDecision:
        """
        合并 policy.py 和具体规则层结果。

        合并原则：

        1. 没有具体规则：
           使用 policy_decision。

        2. 具体规则 deny：
           最终 deny。明显危险操作不允许靠 bypass 误放行。

        3. policy deny：
           最终 deny。例如 readonly / plan 模式禁止写文件。

        4. 具体规则 ask_elevated：
           最终 ask_elevated。

        5. shell 规则 ask：
           最终至少 ask。因为 shell 副作用不可预测。

        6. 普通文件写入 ask：
           允许 accept_edits 模式降级为 allow。

        7. 其他情况：
           使用更严格的决策。
        """
        if rule_result is None:
            return policy_decision

        final_type = policy_decision.decision
        final_risk = max_risk_level(
            policy_decision.risk,
            rule_result.risk,
        )

        reasons = [
            f"policy: {policy_decision.reason}",
            f"{rule_result.source}: {rule_result.reason}",
        ]

        if rule_result.decision == PermissionDecisionType.DENY:
            final_type = PermissionDecisionType.DENY
        elif policy_decision.decision == PermissionDecisionType.DENY:
            final_type = PermissionDecisionType.DENY
        elif rule_result.decision == PermissionDecisionType.ASK_ELEVATED:
            final_type = PermissionDecisionType.ASK_ELEVATED
        elif policy_decision.decision == PermissionDecisionType.ASK_ELEVATED:
            final_type = PermissionDecisionType.ASK_ELEVATED
        elif rule_result.hard_decision and rule_result.decision == PermissionDecisionType.ASK:
            final_type = PermissionDecisionType.ASK
        elif not rule_result.hard_decision and policy_decision.decision == PermissionDecisionType.ALLOW:
            final_type = PermissionDecisionType.ALLOW
        else:
            final_type = stronger_decision(
                policy_decision.decision,
                rule_result.decision,
            )

        return self.build_decision(
            request=policy_decision.request,
            decision_type=final_type,
            risk=final_risk,
            reason=" | ".join(reasons),
            metadata={
                **policy_decision.metadata,
                "policy_decision": policy_decision.decision.value,
                "policy_risk": policy_decision.risk.value,
                "rule_source": rule_result.source,
                "rule_decision": rule_result.decision.value,
                "rule_risk": rule_result.risk.value,
                "rule_reason": rule_result.reason,
                "rule_matched_rules": list(rule_result.matched_rules),
                "rule_metadata": rule_result.metadata,
            },
        )

    def apply_session_override(
        self,
        decision: PermissionDecision,
        rule_result: PermissionGateRuleResult | None,
    ) -> PermissionDecision:
        """
        Apply session-level Always Allow / Always Deny decisions.
        """
        override = self.session_state.get_override(
            decision,
            rule_result=rule_result,
        )

        if override is None:
            return decision

        if override.action == PermissionSessionOverrideAction.DENY:
            return self.build_decision(
                request=decision.request,
                decision_type=PermissionDecisionType.DENY,
                risk=decision.risk,
                reason=f"session override denied operation: {override.reason or 'always deny'}",
                metadata={
                    **decision.metadata,
                    "session_override": override.key.to_dict(),
                    "session_override_action": override.action.value,
                },
            )

        if override.action == PermissionSessionOverrideAction.ALLOW:
            if decision.denied:
                return decision

            return self.build_decision(
                request=decision.request,
                decision_type=PermissionDecisionType.ALLOW,
                risk=decision.risk,
                reason=f"session override allowed operation: {override.reason or 'always allow'}",
                metadata={
                    **decision.metadata,
                    "session_override": override.key.to_dict(),
                    "session_override_action": override.action.value,
                },
            )

        return decision

    def build_decision(
        self,
        *,
        request: PermissionRequest,
        decision_type: PermissionDecisionType,
        risk: RiskLevel,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionDecision:
        return PermissionDecision(
            decision=decision_type,
            request=request,
            mode=policy_mode_from_request(request),
            risk=risk,
            reason=reason,
            requires_confirmation=decision_type in {
                PermissionDecisionType.ASK,
                PermissionDecisionType.ASK_ELEVATED,
            },
            requires_elevated_confirmation=(
                decision_type == PermissionDecisionType.ASK_ELEVATED
            ),
            metadata=metadata or {},
        )

    def record_policy_decision(
        self,
        decision: PermissionDecision,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PermissionAuditRecord | None:
        if not self.audit_enabled or self.audit_log is None:
            return None

        return self.audit_log.record_policy_decision(
            decision,
            session_id=session_id or self.session_id,
            metadata=metadata,
        )


def policy_mode_from_request(
    request: PermissionRequest,
) -> PermissionMode:
    from pywork.permission.mode import normalize_permission_mode

    return normalize_permission_mode(
        request.mode,
        default=DEFAULT_PERMISSION_MODE,
    )


def create_permission_gate(
    workspace_path: str | Path,
    *,
    session_id: str | None = None,
    audit_enabled: bool = True,
    session_state: PermissionGateState | None = None,
) -> PermissionGate:
    return PermissionGate(
        workspace_path,
        session_id=session_id,
        audit_enabled=audit_enabled,
        session_state=session_state,
    )


def render_permission_gate_result(
    result: PermissionGateResult,
) -> str:
    rule = result.rule_result

    parts = [
        f"{result.decision.decision.value}:",
        f"tool={result.decision.request.tool_name}",
        f"mode={result.decision.mode.value}",
        f"risk={result.decision.risk.value}",
        f"reason={result.decision.reason}",
    ]

    if rule is not None:
        parts.append(f"rule={rule.source}")
        parts.append(f"matched={','.join(rule.matched_rules)}")

    return " ".join(parts)


def demo() -> None:
    workspace = Path.cwd()
    gate = PermissionGate(
        workspace,
        session_id="demo_session",
        audit_enabled=False,
    )

    examples = [
        (
            "file_write",
            {
                "path": "src/utils/helper.py",
                "content": "print('hello')\n",
            },
        ),
        (
            "bash",
            {
                "command": "uv run pytest tests",
            },
        ),
        (
            "bash",
            {
                "command": "rm -rf /",
            },
        ),
        (
            "powershell",
            {
                "command": "Remove-Item demo.txt",
            },
        ),
    ]

    for tool_name, arguments in examples:
        result = gate.check_tool(
            tool_name,
            arguments=arguments,
            mode="default",
        )
        print(render_permission_gate_result(result))


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
