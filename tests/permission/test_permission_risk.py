from __future__ import annotations

import pytest

from pywork.permission.risk import (
    RiskLevel,
    compare_risk,
    get_risk_info,
    get_tool_risk,
    max_risk,
    min_risk,
    normalize_risk_level,
    render_risk_label,
    risk_above,
    risk_at_least,
    risk_at_most,
    risk_requires_confirmation,
    risk_requires_elevated_confirmation,
    risk_score,
    risk_to_dict,
)


def test_normalize_risk_level() -> None:
    assert normalize_risk_level("safe") == RiskLevel.SAFE
    assert normalize_risk_level("low") == RiskLevel.LOW
    assert normalize_risk_level("medium") == RiskLevel.MEDIUM
    assert normalize_risk_level("high") == RiskLevel.HIGH
    assert normalize_risk_level("critical") == RiskLevel.CRITICAL


def test_normalize_risk_level_aliases() -> None:
    assert normalize_risk_level("dangerous") == RiskLevel.CRITICAL
    assert normalize_risk_level("danger") == RiskLevel.CRITICAL
    assert normalize_risk_level("read-only") == RiskLevel.LOW
    assert normalize_risk_level("readonly") == RiskLevel.LOW
    assert normalize_risk_level("none") == RiskLevel.SAFE


def test_normalize_risk_level_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_risk_level("unknown")


def test_normalize_risk_level_unknown_with_default() -> None:
    assert (
        normalize_risk_level(
            "unknown",
            default=RiskLevel.MEDIUM,
        )
        == RiskLevel.MEDIUM
    )


def test_risk_score_order() -> None:
    assert risk_score(RiskLevel.SAFE) < risk_score(RiskLevel.LOW)
    assert risk_score(RiskLevel.LOW) < risk_score(RiskLevel.MEDIUM)
    assert risk_score(RiskLevel.MEDIUM) < risk_score(RiskLevel.HIGH)
    assert risk_score(RiskLevel.HIGH) < risk_score(RiskLevel.CRITICAL)


def test_compare_risk() -> None:
    assert compare_risk("safe", "low") == -1
    assert compare_risk("high", "medium") == 1
    assert compare_risk("critical", "dangerous") == 0


def test_risk_comparison_helpers() -> None:
    assert risk_at_least("high", "medium")
    assert risk_above("critical", "high")
    assert risk_at_most("low", "medium")
    assert not risk_at_least("low", "high")


def test_max_min_risk() -> None:
    assert max_risk("safe", "medium", "high") == RiskLevel.HIGH
    assert min_risk("safe", "medium", "high") == RiskLevel.SAFE


def test_get_risk_info() -> None:
    info = get_risk_info("critical")

    assert info.level == RiskLevel.CRITICAL
    assert info.score == 4
    assert info.requires_confirmation
    assert info.requires_elevated_confirmation


def test_confirmation_rules() -> None:
    assert not risk_requires_confirmation("safe")
    assert not risk_requires_confirmation("low")
    assert risk_requires_confirmation("medium")
    assert risk_requires_confirmation("high")
    assert risk_requires_confirmation("critical")

    assert not risk_requires_elevated_confirmation("high")
    assert risk_requires_elevated_confirmation("critical")


def test_get_tool_risk_default_mapping() -> None:
    assert get_tool_risk("echo") == RiskLevel.SAFE
    assert get_tool_risk("file_read") == RiskLevel.LOW
    assert get_tool_risk("glob") == RiskLevel.LOW
    assert get_tool_risk("grep") == RiskLevel.LOW
    assert get_tool_risk("file_write") == RiskLevel.HIGH
    assert get_tool_risk("file_edit") == RiskLevel.HIGH
    assert get_tool_risk("bash") == RiskLevel.CRITICAL
    assert get_tool_risk("powershell") == RiskLevel.CRITICAL


def test_get_tool_risk_unknown_defaults_to_medium() -> None:
    assert get_tool_risk("unknown_tool") == RiskLevel.MEDIUM


def test_get_tool_risk_extra_mapping() -> None:
    assert (
        get_tool_risk(
            "custom_tool",
            extra_mapping={
                "custom_tool": "high",
            },
        )
        == RiskLevel.HIGH
    )


def test_risk_to_dict() -> None:
    data = risk_to_dict("high")

    assert data["level"] == "high"
    assert data["score"] == 3
    assert data["requires_confirmation"]
    assert not data["requires_elevated_confirmation"]


def test_render_risk_label() -> None:
    assert render_risk_label("critical") == "Critical(critical)"