from __future__ import annotations

import pytest

from pywork.permission.mode import (
    PermissionMode,
    choose_looser_mode,
    choose_stricter_mode,
    get_mode_auto_allow_max_risk,
    get_mode_deny_above_risk,
    get_permission_mode_info,
    mode_allows_shell,
    mode_allows_write,
    mode_auto_allows_risk,
    mode_denies_risk,
    mode_is_bypass,
    mode_is_planning,
    mode_is_readonly,
    mode_may_ask_for_risk,
    normalize_permission_mode,
    permission_mode_to_dict,
    render_permission_mode_label,
)
from pywork.permission.risk import RiskLevel


def test_normalize_permission_mode() -> None:
    assert normalize_permission_mode("default") == PermissionMode.DEFAULT
    assert normalize_permission_mode("readonly") == PermissionMode.READONLY
    assert normalize_permission_mode("plan") == PermissionMode.PLAN
    assert normalize_permission_mode("accept_edits") == PermissionMode.ACCEPT_EDITS
    assert normalize_permission_mode("bypass") == PermissionMode.BYPASS


def test_normalize_permission_mode_aliases() -> None:
    assert normalize_permission_mode("read-only") == PermissionMode.READONLY
    assert normalize_permission_mode("read_only") == PermissionMode.READONLY
    assert normalize_permission_mode("ro") == PermissionMode.READONLY
    assert normalize_permission_mode("planning") == PermissionMode.PLAN
    assert normalize_permission_mode("dry-run") == PermissionMode.PLAN
    assert normalize_permission_mode("accept edits") == PermissionMode.ACCEPT_EDITS
    assert normalize_permission_mode("auto-edit") == PermissionMode.ACCEPT_EDITS
    assert normalize_permission_mode("unsafe") == PermissionMode.BYPASS
    assert normalize_permission_mode("no-confirm") == PermissionMode.BYPASS


def test_normalize_permission_mode_unknown() -> None:
    with pytest.raises(ValueError):
        normalize_permission_mode("unknown")


def test_normalize_permission_mode_unknown_with_default() -> None:
    assert (
        normalize_permission_mode(
            "unknown",
            default=PermissionMode.DEFAULT,
        )
        == PermissionMode.DEFAULT
    )


def test_get_permission_mode_info() -> None:
    info = get_permission_mode_info("default")

    assert info.mode == PermissionMode.DEFAULT
    assert info.auto_allow_max_risk == RiskLevel.LOW
    assert info.allows_read
    assert not info.allows_write
    assert not info.allows_shell


def test_mode_flags() -> None:
    assert mode_is_readonly("readonly")
    assert mode_is_planning("plan")
    assert mode_is_bypass("bypass")

    assert not mode_is_readonly("default")
    assert not mode_is_planning("default")
    assert not mode_is_bypass("default")


def test_mode_capabilities() -> None:
    assert not mode_allows_write("default")
    assert not mode_allows_shell("default")

    assert not mode_allows_write("readonly")
    assert not mode_allows_shell("readonly")

    assert not mode_allows_write("plan")
    assert not mode_allows_shell("plan")

    assert mode_allows_write("accept_edits")
    assert not mode_allows_shell("accept_edits")

    assert mode_allows_write("bypass")
    assert mode_allows_shell("bypass")


def test_get_mode_risk_limits() -> None:
    assert get_mode_auto_allow_max_risk("default") == RiskLevel.LOW
    assert get_mode_auto_allow_max_risk("readonly") == RiskLevel.LOW
    assert get_mode_auto_allow_max_risk("plan") == RiskLevel.LOW
    assert get_mode_auto_allow_max_risk("accept_edits") == RiskLevel.HIGH
    assert get_mode_auto_allow_max_risk("bypass") == RiskLevel.CRITICAL

    assert get_mode_deny_above_risk("readonly") == RiskLevel.LOW
    assert get_mode_deny_above_risk("plan") == RiskLevel.LOW
    assert get_mode_deny_above_risk("default") is None


def test_mode_auto_allows_risk() -> None:
    assert mode_auto_allows_risk("default", "safe")
    assert mode_auto_allows_risk("default", "low")
    assert not mode_auto_allows_risk("default", "medium")
    assert not mode_auto_allows_risk("default", "high")

    assert mode_auto_allows_risk("accept_edits", "high")
    assert not mode_auto_allows_risk("accept_edits", "critical")

    assert mode_auto_allows_risk("bypass", "critical")


def test_mode_denies_risk() -> None:
    assert not mode_denies_risk("default", "critical")

    assert not mode_denies_risk("readonly", "safe")
    assert not mode_denies_risk("readonly", "low")
    assert mode_denies_risk("readonly", "medium")
    assert mode_denies_risk("readonly", "high")
    assert mode_denies_risk("readonly", "critical")

    assert mode_denies_risk("plan", "medium")
    assert mode_denies_risk("plan", "high")


def test_mode_may_ask_for_risk() -> None:
    assert not mode_may_ask_for_risk("default", "low")
    assert mode_may_ask_for_risk("default", "medium")
    assert mode_may_ask_for_risk("default", "high")
    assert mode_may_ask_for_risk("default", "critical")

    assert not mode_may_ask_for_risk("readonly", "high")
    assert not mode_may_ask_for_risk("plan", "high")

    assert not mode_may_ask_for_risk("accept_edits", "high")
    assert mode_may_ask_for_risk("accept_edits", "critical")

    assert not mode_may_ask_for_risk("bypass", "critical")


def test_choose_stricter_mode() -> None:
    assert choose_stricter_mode("default", "readonly") == PermissionMode.READONLY
    assert choose_stricter_mode("bypass", "accept_edits") == PermissionMode.ACCEPT_EDITS
    assert choose_stricter_mode("plan", "readonly") == PermissionMode.READONLY


def test_choose_looser_mode() -> None:
    assert choose_looser_mode("default", "readonly") == PermissionMode.DEFAULT
    assert choose_looser_mode("bypass", "accept_edits") == PermissionMode.BYPASS
    assert choose_looser_mode("plan", "readonly") == PermissionMode.PLAN


def test_permission_mode_to_dict() -> None:
    data = permission_mode_to_dict("accept_edits")

    assert data["mode"] == "accept_edits"
    assert data["auto_allow_max_risk"] == "high"
    assert data["allows_write"]
    assert not data["allows_shell"]


def test_render_permission_mode_label() -> None:
    assert render_permission_mode_label("readonly") == "Readonly(readonly)"