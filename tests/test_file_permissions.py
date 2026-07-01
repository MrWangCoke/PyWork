from __future__ import annotations

from pathlib import Path

from pywork.permission.file_permissions import (
    FileOperation,
    evaluate_file_permission,
    normalize_file_operation,
    render_file_permission_result,
)
from pywork.permission.policy import PermissionDecisionType
from pywork.permission.risk import RiskLevel


def test_normal_file_read_allowed(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "README.md",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ALLOW
    assert result.risk == RiskLevel.LOW
    assert result.path == "README.md"
    assert result.allowed


def test_normal_file_write_asks(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/demo.py",
        operation="write",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK
    assert result.risk == RiskLevel.HIGH
    assert result.should_ask


def test_normal_file_edit_asks(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/demo.py",
        operation="edit",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK
    assert result.risk == RiskLevel.HIGH


def test_delete_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/demo.py",
        operation="delete",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert result.requires_elevated_confirmation
    assert "destructive_operation" in result.matched_rules


def test_outside_workspace_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "../outside.txt",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert result.denied
    assert "outside_workspace" in result.matched_rules


def test_target_outside_workspace_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "src/demo.py",
        operation="move",
        workspace_path=tmp_path,
        target_path="../outside.py",
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "target_outside_workspace" in result.matched_rules


def test_git_edit_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".git/config",
        operation="edit",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "denied_dir:.git" in result.matched_rules


def test_git_read_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".git/config",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "denied_dir:.git" in result.matched_rules


def test_pywork_file_history_write_denied(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".pywork/file_history/index.jsonl",
        operation="write",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.DENY
    assert result.risk == RiskLevel.CRITICAL
    assert "denied_dir:.pywork/file_history" in result.matched_rules


def test_sensitive_file_read_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        ".env",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "sensitive_file" in result.matched_rules


def test_sensitive_file_write_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "secrets/api.key",
        operation="write",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "sensitive_file" in result.matched_rules


def test_important_project_file_edit_asks_elevated(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "pyproject.toml",
        operation="edit",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ASK_ELEVATED
    assert result.risk == RiskLevel.CRITICAL
    assert "important_project_file" in result.matched_rules


def test_important_project_file_read_allowed(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "pyproject.toml",
        operation="read",
        workspace_path=tmp_path,
    )

    assert result.decision == PermissionDecisionType.ALLOW
    assert result.risk == RiskLevel.LOW


def test_normalize_file_operation_aliases() -> None:
    assert normalize_file_operation("file_read") == FileOperation.READ
    assert normalize_file_operation("glob") == FileOperation.LIST
    assert normalize_file_operation("grep") == FileOperation.SEARCH
    assert normalize_file_operation("file_write") == FileOperation.WRITE
    assert normalize_file_operation("file_edit") == FileOperation.EDIT
    assert normalize_file_operation("rm") == FileOperation.DELETE
    assert normalize_file_operation("mv") == FileOperation.MOVE


def test_render_file_permission_result(tmp_path: Path) -> None:
    result = evaluate_file_permission(
        "README.md",
        operation="read",
        workspace_path=tmp_path,
    )

    rendered = render_file_permission_result(result)

    assert "allow" in rendered
    assert "operation=read" in rendered
    assert "path=README.md" in rendered
    assert "risk=low" in rendered