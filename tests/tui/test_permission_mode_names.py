from __future__ import annotations

from pywork.tui.app import (
    PERMISSION_MODE_CYCLE,
    PyWorkApp,
    normalize_tui_permission_mode,
)


def test_tui_permission_cycle_uses_canonical_bypass_name() -> None:
    assert "bypass_permissions" in PERMISSION_MODE_CYCLE
    assert "bypass" not in PERMISSION_MODE_CYCLE


def test_normalize_tui_permission_mode_aliases() -> None:
    assert normalize_tui_permission_mode("default") == "default"
    assert normalize_tui_permission_mode("accept-edits") == "accept_edits"
    assert normalize_tui_permission_mode("accept_edits") == "accept_edits"
    assert normalize_tui_permission_mode("plan") == "plan"
    assert normalize_tui_permission_mode("readonly") == "readonly"
    assert normalize_tui_permission_mode("read-only") == "readonly"
    assert normalize_tui_permission_mode("safe") == "readonly"
    assert normalize_tui_permission_mode("bypass") == "bypass_permissions"
    assert normalize_tui_permission_mode("bypass_permissions") == "bypass_permissions"
    assert normalize_tui_permission_mode("unknown") == "default"


def test_app_reads_legacy_bypass_as_bypass_permissions() -> None:
    app = PyWorkApp(
        config={
            "permissions": {
                "mode": "bypass",
            }
        }
    )

    assert app.get_permission_mode() == "bypass_permissions"


def test_set_permission_mode_normalizes_legacy_bypass() -> None:
    app = PyWorkApp()

    app.set_permission_mode("bypass")

    assert app.get_permission_mode() == "bypass_permissions"
    assert app.config["permissions"]["mode"] == "bypass_permissions"


def test_set_permission_mode_normalizes_read_only() -> None:
    app = PyWorkApp()

    app.set_permission_mode("read-only")

    assert app.get_permission_mode() == "readonly"
    assert app.config["permissions"]["mode"] == "readonly"