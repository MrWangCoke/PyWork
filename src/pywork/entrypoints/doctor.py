from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pywork.bootstrap.config_loader import ConfigLoadResult, load_config
from pywork.bootstrap.dependency_check import (
    DependencyCheckResult,
    check_all_dependencies,
    has_required_failures,
)
from pywork.bootstrap.env import (
    EnvironmentInfo,
    PythonVersionCheck,
    check_python_version,
    collect_environment_info,
)
from pywork.bootstrap.workspace_loader import (
    WorkspaceError,
    WorkspaceInfo,
    load_workspace,
)


console = Console()


def workspace_to_dict(workspace: WorkspaceInfo) -> dict[str, Any]:
    return asdict(workspace)


def env_to_dict(env: EnvironmentInfo) -> dict[str, Any]:
    return asdict(env)


def version_check_to_dict(version_check: PythonVersionCheck) -> dict[str, Any]:
    return asdict(version_check)


def config_result_to_dict(config_result: ConfigLoadResult) -> dict[str, Any]:
    return {
        "config": config_result.config,
        "sources": [
            {
                "path": str(source.path),
                "exists": source.exists,
                "loaded": source.loaded,
                "error": source.error,
            }
            for source in config_result.sources
        ],
    }

def dependency_result_to_dict(
    result: DependencyCheckResult,
) -> dict[str, Any]:
    return asdict(result)


def print_python_section(
    env: EnvironmentInfo,
    version_check: PythonVersionCheck,
) -> None:
    table = Table(title="Python / OS", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Python", env.python_version)
    table.add_row("Executable", env.python_executable)
    table.add_row("Implementation", env.python_implementation)
    table.add_row("Version OK", "yes" if version_check.ok else "no")
    table.add_row("Version Reason", version_check.reason)
    table.add_row("OS", f"{env.os_system} {env.os_release}")
    table.add_row("Machine", env.machine)
    table.add_row("Shell", str(env.shell))
    table.add_row("Terminal", str(env.terminal))
    table.add_row("Virtual Env", "yes" if env.is_virtual_env else "no")
    table.add_row("Virtual Env Path", str(env.virtual_env_path))

    console.print(table)


def print_workspace_section(workspace: WorkspaceInfo) -> None:
    table = Table(title="Workspace", show_header=True, header_style="bold green")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Requested Path", workspace.requested_path)
    table.add_row("Workspace Path", workspace.workspace_path)
    table.add_row("Project Root", workspace.project_root)
    table.add_row(".pywork Dir", workspace.pywork_dir)
    table.add_row("Exists", str(workspace.exists))
    table.add_row("Is Directory", str(workspace.is_directory))
    table.add_row("Has Git", "yes" if workspace.has_git else "no")
    table.add_row("Has .pywork", "yes" if workspace.has_pywork_dir else "no")
    table.add_row("Has README", "yes" if workspace.has_readme else "no")
    table.add_row("Markers", ", ".join(workspace.detected_markers) or "none")

    console.print(table)


def print_config_section(config_result: ConfigLoadResult) -> None:
    table = Table(title="Config Sources", show_header=True, header_style="bold magenta")
    table.add_column("Status")
    table.add_column("Path")

    for source in config_result.sources:
        if source.loaded:
            status = "[green]loaded[/green]"
        elif source.exists:
            status = "[red]failed[/red]"
        else:
            status = "[dim]missing[/dim]"

        table.add_row(status, str(source.path))

    console.print(table)

    config = config_result.config

    summary = Table(title="Config Summary", show_header=True, header_style="bold magenta")
    summary.add_column("Field")
    summary.add_column("Value")

    summary.add_row("App", str(config.get("app", {}).get("name", "pywork")))
    summary.add_row("Provider", str(config.get("default", {}).get("provider", "unknown")))
    summary.add_row("Model", str(config.get("default", {}).get("model", "unknown")))
    summary.add_row("Permission Mode", str(config.get("permissions", {}).get("mode", "default")))
    summary.add_row("MCP Enabled", str(config.get("mcp", {}).get("enabled", False)))

    console.print(summary)


def print_dependency_section(results: list[DependencyCheckResult]) -> None:
    table = Table(title="Dependencies", show_header=True, header_style="bold yellow")
    table.add_column("Status")
    table.add_column("Name")
    table.add_column("Kind")
    table.add_column("Level")
    table.add_column("Detail")

    for result in results:
        if result.ok:
            status = "[green]OK[/green]"
        elif result.level == "optional":
            status = "[yellow]MISS[/yellow]"
        else:
            status = "[red]FAIL[/red]"

        table.add_row(
            status,
            result.name,
            result.kind,
            result.level,
            result.detail,
        )

    console.print(table)


def build_doctor_payload(
    *,
    env: EnvironmentInfo,
    version_check: PythonVersionCheck,
    workspace: WorkspaceInfo | None,
    config_result: ConfigLoadResult | None,
    dependencies: list[DependencyCheckResult],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    required_dependency_failed = has_required_failures(dependencies)

    ok = (
        version_check.ok
        and workspace is not None
        and config_result is not None
        and not required_dependency_failed
        and not errors
    )

    return {
        "ok": ok,
        "python": {
            "environment": env_to_dict(env),
            "version_check": version_check_to_dict(version_check),
        },
        "workspace": workspace_to_dict(workspace) if workspace else None,
        "config": config_result_to_dict(config_result) if config_result else None,
        "dependencies": [
            dependency_result_to_dict(result)
            for result in dependencies
        ],
        "errors": errors,
        "warnings": warnings,
    }


def print_doctor_report(payload: dict[str, Any]) -> None:
    ok = bool(payload["ok"])

    console.print(
        Panel.fit(
            "[bold cyan]PyWork Doctor[/bold cyan]\n"
            + ("[green]Status: OK[/green]" if ok else "[red]Status: FAILED[/red]"),
            border_style="cyan" if ok else "red",
        )
    )

    if payload["errors"]:
        error_table = Table(title="Errors", show_header=True, header_style="bold red")
        error_table.add_column("Error")

        for error in payload["errors"]:
            error_table.add_row(str(error))

        console.print(error_table)

    if payload["warnings"]:
        warning_table = Table(title="Warnings", show_header=True, header_style="bold yellow")
        warning_table.add_column("Warning")

        for warning in payload["warnings"]:
            warning_table.add_row(str(warning))

        console.print(warning_table)


def run_doctor(
    workspace: str | Path = ".",
    *,
    json_output: bool = False,
) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    env = collect_environment_info()
    version_check = check_python_version()

    if not version_check.ok:
        errors.append(version_check.reason)

    workspace_info: WorkspaceInfo | None = None
    try:
        workspace_info = load_workspace(workspace)
    except WorkspaceError as exc:
        errors.append(f"Workspace error: {exc}")

    config_result: ConfigLoadResult | None = None
    if workspace_info is not None:
        try:
            config_result = load_config(
                workspace=Path(workspace_info.project_root),
            )
        except Exception as exc:
            errors.append(f"Config error: {exc}")

    dependencies = check_all_dependencies()

    for dependency in dependencies:
        if not dependency.ok and dependency.level == "optional":
            warnings.append(
                f"Optional dependency missing: {dependency.name} - {dependency.detail}"
            )

    for dependency in dependencies:
        if not dependency.ok and dependency.level == "required":
            errors.append(
                f"Required dependency failed: {dependency.name} - {dependency.detail}"
            )

    payload = build_doctor_payload(
        env=env,
        version_check=version_check,
        workspace=workspace_info,
        config_result=config_result,
        dependencies=dependencies,
        errors=errors,
        warnings=warnings,
    )

    if json_output:
        console.print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    else:
        print_doctor_report(payload)
        print_python_section(env, version_check)

        if workspace_info is not None:
            print_workspace_section(workspace_info)

        if config_result is not None:
            print_config_section(config_result)

        print_dependency_section(dependencies)

    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywork-doctor",
        description="Run PyWork environment diagnostics.",
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace path.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raise SystemExit(
        run_doctor(
            workspace=args.workspace,
            json_output=args.json,
        )
    )


if __name__ == "__main__":
    main()