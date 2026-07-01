from __future__ import annotations

from pywork.schemas.tool_schema import ToolRiskLevel
from pywork.tools.bash import BashTool
from pywork.tools.file_edit import FileEditTool
from pywork.tools.file_write import FileWriteTool
from pywork.tools.powershell import PowerShellTool
from pywork.tools.registry import (
    ToolRegistry,
    create_default_registry,
    reset_default_registry,
)


def test_default_registry_contains_high_risk_tools() -> None:
    registry = create_default_registry()

    names = registry.list_names()

    assert "file_write" in names
    assert "file_edit" in names
    assert "bash" in names
    assert "powershell" in names


def test_default_registry_high_risk_tool_instances() -> None:
    registry = create_default_registry()

    assert isinstance(registry.require("file_write"), FileWriteTool)
    assert isinstance(registry.require("file_edit"), FileEditTool)
    assert isinstance(registry.require("bash"), BashTool)
    assert isinstance(registry.require("powershell"), PowerShellTool)


def test_default_registry_high_risk_tool_risk_levels() -> None:
    registry = create_default_registry()

    assert registry.require("file_write").get_risk_level() == ToolRiskLevel.HIGH
    assert registry.require("file_edit").get_risk_level() == ToolRiskLevel.HIGH

    assert registry.require("bash").get_risk_level() == ToolRiskLevel.DANGEROUS
    assert registry.require("powershell").get_risk_level() == ToolRiskLevel.DANGEROUS


def test_default_registry_exposes_high_risk_tool_definitions() -> None:
    registry = create_default_registry()

    definitions = registry.list_definitions()

    names = set()

    for definition in definitions:
        if "function" in definition:
            names.add(definition["function"]["name"])
        else:
            names.add(definition["name"])

    assert "file_write" in names
    assert "file_edit" in names
    assert "bash" in names
    assert "powershell" in names


def test_registry_entries_have_permission_metadata() -> None:
    registry = create_default_registry()

    file_write_entry = registry.get_entry("file_write")
    file_edit_entry = registry.get_entry("file_edit")
    bash_entry = registry.get_entry("bash")
    powershell_entry = registry.get_entry("powershell")

    assert file_write_entry is not None
    assert file_edit_entry is not None
    assert bash_entry is not None
    assert powershell_entry is not None

    assert file_write_entry.metadata["requires_permission_gate"] is True
    assert file_write_entry.metadata["requires_diff_preview"] is True

    assert file_edit_entry.metadata["requires_permission_gate"] is True
    assert file_edit_entry.metadata["requires_diff_preview"] is True

    assert bash_entry.metadata["requires_permission_gate"] is True
    assert bash_entry.metadata["requires_command_safety_check"] is True

    assert powershell_entry.metadata["requires_permission_gate"] is True
    assert powershell_entry.metadata["requires_command_safety_check"] is True


def test_reset_default_registry_contains_high_risk_tools() -> None:
    registry = reset_default_registry()

    assert registry.has("file_write")
    assert registry.has("file_edit")
    assert registry.has("bash")
    assert registry.has("powershell")


def test_manual_registry_can_register_high_risk_tools() -> None:
    registry = ToolRegistry()

    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(BashTool())
    registry.register(PowerShellTool())

    assert registry.has("file_write")
    assert registry.has("file_edit")
    assert registry.has("bash")
    assert registry.has("powershell")