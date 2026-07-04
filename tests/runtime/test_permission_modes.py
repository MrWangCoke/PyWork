from __future__ import annotations

from pywork.runtime.graph import (
    PERMISSION_MODE_ACCEPT_EDITS,
    PERMISSION_MODE_BYPASS,
    PERMISSION_MODE_DEFAULT,
    PERMISSION_MODE_PLAN,
    PERMISSION_MODE_READONLY,
    evaluate_permission,
    max_allowed_risk_for_permission_mode,
    normalize_permission_mode,
)
from pywork.schemas.tool_schema import ToolRiskLevel, create_tool_call
from pywork.tools.registry import create_default_registry


def test_normalize_permission_mode_aliases() -> None:
    assert normalize_permission_mode("default") == PERMISSION_MODE_DEFAULT
    assert normalize_permission_mode("accept-edits") == PERMISSION_MODE_ACCEPT_EDITS
    assert normalize_permission_mode("accept_edits") == PERMISSION_MODE_ACCEPT_EDITS
    assert normalize_permission_mode("plan") == PERMISSION_MODE_PLAN
    assert normalize_permission_mode("readonly") == PERMISSION_MODE_READONLY
    assert normalize_permission_mode("read-only") == PERMISSION_MODE_READONLY
    assert normalize_permission_mode("read_only") == PERMISSION_MODE_READONLY
    assert normalize_permission_mode("safe") == PERMISSION_MODE_READONLY
    assert normalize_permission_mode("bypass") == PERMISSION_MODE_BYPASS
    assert normalize_permission_mode("bypass_permissions") == PERMISSION_MODE_BYPASS
    assert normalize_permission_mode("dangerous") == PERMISSION_MODE_BYPASS
    assert normalize_permission_mode("unknown") == PERMISSION_MODE_DEFAULT


def test_max_allowed_risk_for_permission_modes() -> None:
    assert max_allowed_risk_for_permission_mode("default") == ToolRiskLevel.LOW
    assert max_allowed_risk_for_permission_mode("accept_edits") == ToolRiskLevel.MEDIUM
    assert max_allowed_risk_for_permission_mode("plan") == ToolRiskLevel.SAFE
    assert max_allowed_risk_for_permission_mode("readonly") == ToolRiskLevel.SAFE
    assert max_allowed_risk_for_permission_mode("bypass") == ToolRiskLevel.DANGEROUS
    assert max_allowed_risk_for_permission_mode("bypass_permissions") == ToolRiskLevel.DANGEROUS


def test_readonly_allows_safe_tool_but_blocks_low_tool() -> None:
    registry = create_default_registry()

    safe_call = create_tool_call(
        "echo",
        {
            "text": "hello",
        },
    )
    low_call = create_tool_call(
        "agent",
        {
            "action": "list_agents",
        },
    )

    safe_decision = evaluate_permission(
        safe_call,
        registry=registry,
        permission_mode="readonly",
    )
    low_decision = evaluate_permission(
        low_call,
        registry=registry,
        permission_mode="readonly",
    )

    assert safe_decision.allowed is True
    assert low_decision.allowed is False
    assert low_decision.requires_confirmation is True


def test_bypass_alias_allows_high_risk_tool() -> None:
    registry = create_default_registry()

    call = create_tool_call(
        "file_write",
        {
            "path": "tmp.txt",
            "content": "hello",
        },
    )

    decision = evaluate_permission(
        call,
        registry=registry,
        permission_mode="bypass",
    )

    assert decision.allowed is True


def test_plan_mode_still_blocks_safe_tool_execution() -> None:
    registry = create_default_registry()

    call = create_tool_call(
        "echo",
        {
            "text": "hello",
        },
    )

    decision = evaluate_permission(
        call,
        registry=registry,
        permission_mode="plan",
    )

    assert decision.allowed is False
    assert decision.requires_confirmation is True
    assert "plan mode does not execute tools" in decision.reason