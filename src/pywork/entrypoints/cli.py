from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pywork.bootstrap.config_loader import ConfigLoadResult, load_config
from pywork.bootstrap.env import check_python_version
from pywork.bootstrap.workspace_loader import (
    WorkspaceError,
    WorkspaceInfo,
    load_workspace,
)


VERSION = "0.1.0"

console = Console()


def print_startup_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]PyWork[/bold cyan]\n"
            "Python TUI Coding Agent Workspace\n"
            "[dim]pywork .[/dim]",
            border_style="cyan",
        )
    )


def print_workspace_summary(workspace: WorkspaceInfo) -> None:
    table = Table(title="Workspace", show_header=True, header_style="bold cyan")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("Workspace", workspace.workspace_path)
    table.add_row("Project Root", workspace.project_root)
    table.add_row(".pywork Dir", workspace.pywork_dir)
    table.add_row("Git", "yes" if workspace.has_git else "no")
    table.add_row("README", "yes" if workspace.has_readme else "no")
    table.add_row("Markers", ", ".join(workspace.detected_markers) or "none")

    console.print(table)


def print_config_summary(config_result: ConfigLoadResult) -> None:
    config = config_result.config

    app_config = config.get("app", {})
    model_config = config.get("default", {})
    permissions_config = config.get("permissions", {})
    tools_config = config.get("tools", {})
    mcp_config = config.get("mcp", {})

    table = Table(title="Config", show_header=True, header_style="bold green")
    table.add_column("Field")
    table.add_column("Value")

    table.add_row("App", str(app_config.get("name", "pywork")))
    table.add_row("Model Provider", str(model_config.get("provider", "unknown")))
    table.add_row("Model", str(model_config.get("model", "unknown")))
    table.add_row("Base URL", str(model_config.get("base_url", "not set")))
    table.add_row("Permission Mode", str(permissions_config.get("mode", "default")))
    table.add_row("File Tools", str(tools_config.get("enable_file_tools", False)))
    table.add_row("Shell Tools", str(tools_config.get("enable_shell_tools", False)))
    table.add_row("Git Tools", str(tools_config.get("enable_git_tools", False)))
    table.add_row("MCP", str(mcp_config.get("enabled", False)))

    console.print(table)


def workspace_to_dict(workspace: WorkspaceInfo) -> dict[str, Any]:
    return {
        "requested_path": workspace.requested_path,
        "workspace_path": workspace.workspace_path,
        "project_root": workspace.project_root,
        "exists": workspace.exists,
        "is_directory": workspace.is_directory,
        "is_file": workspace.is_file,
        "has_git": workspace.has_git,
        "has_pywork_dir": workspace.has_pywork_dir,
        "has_readme": workspace.has_readme,
        "detected_markers": workspace.detected_markers,
        "pywork_dir": workspace.pywork_dir,
        "cwd": workspace.cwd,
    }


def launch_tui_or_fallback(
    workspace: WorkspaceInfo,
    config_result: ConfigLoadResult,
    *,
    no_tui: bool,
) -> None:
    if no_tui:
        console.print("[yellow]TUI disabled by --no-tui.[/yellow]")
        return

    try:
        from pywork.tui.repl_launcher import launch_repl
    except Exception:
        launch_repl = None

    if launch_repl is not None:
        launch_repl(
            workspace=workspace,
            config=config_result.config,
        )
        return

    console.print(
        Panel(
            "[yellow]TUI launcher is not implemented yet.[/yellow]\n\n"
            "Current startup is ready:\n"
            "- workspace loaded\n"
            "- config loaded\n"
            "- CLI entry works\n\n"
            "Next step:\n"
            "[bold]tui/app.py[/bold] and [bold]tui/repl_launcher.py[/bold]",
            title="PyWork Startup",
            border_style="yellow",
        )
    )


def run_startup(
    workspace_path: str | Path = ".",
    *,
    no_tui: bool = False,
    json_output: bool = False,
) -> int:
    version_check = check_python_version()

    if not version_check.ok:
        console.print(f"[red]{version_check.reason}[/red]")
        return 1

    try:
        workspace = load_workspace(workspace_path)
    except WorkspaceError as exc:
        console.print(f"[red]Workspace error:[/red] {exc}")
        return 1

    try:
        config_result = load_config(
            workspace=Path(workspace.project_root),
        )
    except Exception as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        return 1

    if json_output:
        payload = {
            "version": VERSION,
            "workspace": workspace_to_dict(workspace),
            "config": config_result.config,
        }

        console.print(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print_startup_banner()
    print_workspace_summary(workspace)
    print_config_summary(config_result)

    launch_tui_or_fallback(
        workspace=workspace,
        config_result=config_result,
        no_tui=no_tui,
    )

    return 0


def run_init_entry(
    workspace: str | Path = ".",
    *,
    force: bool = False,
    quiet: bool = False,
    create_rules_file: bool = True,
) -> int:
    try:
        from pywork.entrypoints.init import run_init
    except Exception as exc:
        console.print(
            Panel(
                "[yellow]entrypoints/init.py is not ready yet.[/yellow]\n\n"
                f"Import error: {exc}",
                title="PyWork Init",
                border_style="yellow",
            )
        )
        return 1

    return run_init(
        workspace=workspace,
        force=force,
        quiet=quiet,
        create_rules_file=create_rules_file,
    )


def run_doctor_entry(
    workspace: str | Path = ".",
    *,
    json_output: bool = False,
) -> int:
    try:
        from pywork.entrypoints.doctor import run_doctor
    except Exception as exc:
        console.print(
            Panel(
                "[yellow]entrypoints/doctor.py is not ready yet.[/yellow]\n\n"
                f"Import error: {exc}",
                title="PyWork Doctor",
                border_style="yellow",
            )
        )
        return 1

    return run_doctor(
        workspace=workspace,
        json_output=json_output,
    )


def build_startup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywork",
        description="PyWork: a Python TUI coding agent workspace.",
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace path. Use '.' for the current directory.",
    )

    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Run startup only and do not launch TUI.",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print startup information as JSON.",
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="Initialize PyWork project files.",
    )

    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run environment diagnostics.",
    )

    parser.add_argument(
        "--version",
        action="store_true",
        help="Print PyWork version.",
    )

    return parser


def build_init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywork init",
        description="Initialize a PyWork project.",
    )

    parser.add_argument(
        "workspace",
        nargs="?",
        default=".",
        help="Workspace path.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated files if they already exist.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print init report.",
    )

    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Do not create PYWORK.md.",
    )

    return parser


def build_doctor_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywork doctor",
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
        help="Print doctor result as JSON.",
    )

    return parser


def print_version() -> int:
    console.print(f"pywork {VERSION}")
    return 0


def handle_startup_command(argv: list[str]) -> int:
    parser = build_startup_parser()
    args = parser.parse_args(argv)

    if args.version:
        return print_version()

    if args.init:
        return run_init_entry(
            workspace=args.workspace,
        )

    if args.doctor:
        return run_doctor_entry(
            workspace=args.workspace,
            json_output=args.json,
        )

    return run_startup(
        workspace_path=args.workspace,
        no_tui=args.no_tui,
        json_output=args.json,
    )


def handle_init_command(argv: list[str]) -> int:
    parser = build_init_parser()
    args = parser.parse_args(argv)

    return run_init_entry(
        workspace=args.workspace,
        force=args.force,
        quiet=args.quiet,
        create_rules_file=not args.no_rules,
    )


def handle_doctor_command(argv: list[str]) -> int:
    parser = build_doctor_parser()
    args = parser.parse_args(argv)

    return run_doctor_entry(
        workspace=args.workspace,
        json_output=args.json,
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args:
        raise SystemExit(
            run_startup(
                workspace_path=".",
            )
        )

    command = args[0]

    if command == "version":
        raise SystemExit(print_version())

    if command == "init":
        raise SystemExit(handle_init_command(args[1:]))

    if command == "doctor":
        raise SystemExit(handle_doctor_command(args[1:]))

    raise SystemExit(handle_startup_command(args))


if __name__ == "__main__":
    main()