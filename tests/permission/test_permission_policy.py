from __future__ import annotations

from dataclasses import dataclass

from pywork.permission.policy import (
    PermissionDecisionType,
    PermissionPolicy,
    PermissionPolicyConfig,
    PermissionRequest,
    evaluate_permission,
    render_permission_decision,
)
from pywork.permission.risk import RiskLevel


def test_default_mode_allows_read_tools() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "file_read",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ALLOW
    assert decision.allowed
    assert decision.risk == RiskLevel.LOW


def test_default_mode_asks_for_file_write() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "file_write",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ASK
    assert decision.should_ask
    assert decision.requires_confirmation
    assert not decision.requires_elevated_confirmation
    assert decision.risk == RiskLevel.HIGH


def test_default_mode_asks_elevated_for_bash() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "bash",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert decision.should_ask
    assert decision.requires_confirmation
    assert decision.requires_elevated_confirmation
    assert decision.risk == RiskLevel.CRITICAL


def test_readonly_mode_denies_write_and_shell() -> None:
    policy = PermissionPolicy()

    write_decision = policy.evaluate_tool(
        "file_write",
        mode="readonly",
    )
    bash_decision = policy.evaluate_tool(
        "bash",
        mode="readonly",
    )

    assert write_decision.decision == PermissionDecisionType.DENY
    assert write_decision.denied

    assert bash_decision.decision == PermissionDecisionType.DENY
    assert bash_decision.denied


def test_readonly_mode_allows_read() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "grep",
        mode="readonly",
    )

    assert decision.decision == PermissionDecisionType.ALLOW
    assert decision.allowed


def test_plan_mode_denies_write_and_shell() -> None:
    policy = PermissionPolicy()

    write_decision = policy.evaluate_tool(
        "file_edit",
        mode="plan",
    )
    shell_decision = policy.evaluate_tool(
        "powershell",
        mode="plan",
    )

    assert write_decision.decision == PermissionDecisionType.DENY
    assert shell_decision.decision == PermissionDecisionType.DENY


def test_accept_edits_allows_file_write_and_edit() -> None:
    policy = PermissionPolicy()

    write_decision = policy.evaluate_tool(
        "file_write",
        mode="accept_edits",
    )
    edit_decision = policy.evaluate_tool(
        "file_edit",
        mode="accept_edits",
    )

    assert write_decision.decision == PermissionDecisionType.ALLOW
    assert edit_decision.decision == PermissionDecisionType.ALLOW


def test_accept_edits_still_asks_elevated_for_shell() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "powershell",
        mode="accept_edits",
    )

    assert decision.decision == PermissionDecisionType.ASK_ELEVATED
    assert decision.requires_elevated_confirmation


def test_bypass_allows_critical() -> None:
    policy = PermissionPolicy()

    decision = policy.evaluate_tool(
        "bash",
        mode="bypass",
    )

    assert decision.decision == PermissionDecisionType.ALLOW
    assert decision.allowed


def test_explicit_always_deny_tool() -> None:
    policy = PermissionPolicy(
        PermissionPolicyConfig(
            always_deny_tools={"file_read"},
        )
    )

    decision = policy.evaluate_tool(
        "file_read",
        mode="bypass",
    )

    # bypass 优先级最高，所以这里仍然 allow
    assert decision.decision == PermissionDecisionType.ALLOW

    decision = policy.evaluate_tool(
        "file_read",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.DENY


def test_explicit_always_allow_tool() -> None:
    policy = PermissionPolicy(
        PermissionPolicyConfig(
            always_allow_tools={"custom_tool"},
        )
    )

    decision = policy.evaluate_tool(
        "custom_tool",
        mode="default",
        risk="critical",
    )

    assert decision.decision == PermissionDecisionType.ALLOW


def test_explicit_always_ask_tool() -> None:
    policy = PermissionPolicy(
        PermissionPolicyConfig(
            always_ask_tools={"file_read"},
        )
    )

    decision = policy.evaluate_tool(
        "file_read",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ASK


def test_explicit_always_ask_elevated_tool() -> None:
    policy = PermissionPolicy(
        PermissionPolicyConfig(
            always_ask_elevated_tools={"file_read"},
        )
    )

    decision = policy.evaluate_tool(
        "file_read",
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ASK_ELEVATED


def test_tool_risk_override() -> None:
    policy = PermissionPolicy(
        PermissionPolicyConfig(
            tool_risk_overrides={
                "custom_tool": "high",
            },
        )
    )

    decision = policy.evaluate_tool(
        "custom_tool",
        mode="default",
    )

    assert decision.risk == RiskLevel.HIGH
    assert decision.decision == PermissionDecisionType.ASK


def test_request_to_dict() -> None:
    request = PermissionRequest(
        tool_name="file_write",
        action="write",
        mode="default",
        risk="high",
        arguments={
            "path": "demo.txt",
        },
        call_id="call_1",
    )

    data = request.to_dict()

    assert data["tool_name"] == "file_write"
    assert data["action"] == "write"
    assert data["mode"] == "default"
    assert data["risk"] == "high"
    assert data["arguments"]["path"] == "demo.txt"
    assert data["call_id"] == "call_1"


def test_decision_to_dict() -> None:
    decision = evaluate_permission(
        "file_write",
        mode="default",
        arguments={
            "path": "demo.txt",
        },
    )

    data = decision.to_dict()

    assert data["decision"] == "ask"
    assert data["allowed"] is False
    assert data["should_ask"] is True
    assert data["mode"] == "default"
    assert data["risk"] == "high"


@dataclass
class FakeToolCall:
    tool_name: str
    arguments: dict[str, str]
    call_id: str


def test_evaluate_tool_call_like_object() -> None:
    policy = PermissionPolicy()

    call = FakeToolCall(
        tool_name="file_write",
        arguments={
            "path": "demo.txt",
        },
        call_id="call_123",
    )

    decision = policy.evaluate_tool_call(
        call,
        mode="default",
    )

    assert decision.decision == PermissionDecisionType.ASK
    assert decision.request.tool_name == "file_write"
    assert decision.request.arguments["path"] == "demo.txt"
    assert decision.request.call_id == "call_123"


def test_render_permission_decision() -> None:
    decision = evaluate_permission(
        "bash",
        mode="default",
    )

    rendered = render_permission_decision(decision)

    assert "ask_elevated" in rendered
    assert "tool=bash" in rendered
    assert "mode=default" in rendered
    assert "risk=critical" in rendered