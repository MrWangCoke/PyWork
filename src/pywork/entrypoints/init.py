from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pywork.bootstrap.workspace_loader import WorkspaceError, load_workspace


console = Console()

InitActionKind = Literal["directory", "file"]
InitActionStatus = Literal["created", "exists", "overwritten"]


PYWORK_SUBDIRS = [
    "sessions",
    "memory",
    "tasks",
    "agents",
    "artifacts",
    "logs",
    "cache",
]


DEFAULT_PROJECT_CONFIG = """[app]
name = "pywork"
default_workspace = "."
permission_mode = "default"

[default]
provider = "openai_compatible"
model = "deepseek-chat"
base_url = "https://api.deepseek.com"
api_key_env = "DEEPSEEK_API_KEY"

[permissions]
mode = "default"

[risk]
allow_read = true
require_approval_for_write = true
require_approval_for_shell = true

[tools]
enable_file_tools = true
enable_shell_tools = true
enable_git_tools = true
enable_mcp_tools = false

[mcp]
enabled = false
"""


DEFAULT_MEMORY = """# PyWork Memory

This file stores project-level memory for PyWork.

Use it for stable project facts, for example:

- Project structure
- Important commands
- Coding rules
- Long-term decisions
"""


DEFAULT_RULES = """# PYWORK.md

Project instructions for PyWork agents.

## Rules

- Read files before editing them.
- Explain risky operations before running them.
- Do not delete files unless the user clearly asks.
- Prefer small, reviewable changes.
- Run tests or checks when possible.

## Project Notes

Add project-specific notes here.
"""


DEFAULT_GITIGNORE = """# PyWork runtime files
sessions/
logs/
cache/
artifacts/
"""


@dataclass(frozen=True)
class InitAction:
    kind: InitActionKind
    path: str
    status: InitActionStatus


@dataclass(frozen=True)
class InitResult:
    workspace_path: str
    project_root: str
    pywork_dir: str
    actions: list[InitAction]


class InitError(RuntimeError):
    pass


def ensure_directory(path: Path) -> InitAction:
    if path.exists() and not path.is_dir():
        raise InitError(f"Path exists but is not a directory: {path}")

    existed = path.exists()
    path.mkdir(parents=True, exist_ok=True)

    return InitAction(
        kind="directory",
        path=str(path),
        status="exists" if existed else "created",
    )


def write_file(
    path: Path,
    content: str,
    *,
    force: bool = False,
) -> InitAction:
    if path.exists() and path.is_dir():
        raise InitError(f"Path exists but is a directory: {path}")

    if path.exists() and not force:
        return InitAction(
            kind="file",
            path=str(path),
            status="exists",
        )

    status: InitActionStatus = "overwritten" if path.exists() else "created"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return InitAction(
        kind="file",
        path=str(path),
        status=status,
    )


def initialize_project(
    workspace: str | Path = ".",
    *,
    force: bool = False,
    create_rules_file: bool = True,
) -> InitResult:
    workspace_info = load_workspace(
        workspace,
        create_pywork_dir=False,
    )

    project_root = Path(workspace_info.project_root)
    pywork_dir = project_root / ".pywork"

    actions: list[InitAction] = []

    actions.append(ensure_directory(pywork_dir))

    for subdir in PYWORK_SUBDIRS:
        actions.append(ensure_directory(pywork_dir / subdir))

    actions.append(
        write_file(
            pywork_dir / "config.toml",
            DEFAULT_PROJECT_CONFIG,
            force=force,
        )
    )

    actions.append(
        write_file(
            pywork_dir / "MEMORY.md",
            DEFAULT_MEMORY,
            force=force,
        )
    )

    actions.append(
        write_file(
            pywork_dir / ".gitignore",
            DEFAULT_GITIGNORE,
            force=force,
        )
    )

    if create_rules_file:
        actions.append(
            write_file(
                project_root / "PYWORK.md",
                DEFAULT_RULES,
                force=force,
            )
        )

    return InitResult(
        workspace_path=workspace_info.workspace_path,
        project_root=str(project_root),
        pywork_dir=str(pywork_dir),
        actions=actions,
    )


def init_result_to_dict(result: InitResult) -> dict[str, object]:
    return {
        "workspace_path": result.workspace_path,
        "project_root": result.project_root,
        "pywork_dir": result.pywork_dir,
        "actions": [asdict(action) for action in result.actions],
    }


def print_init_report(result: InitResult) -> None:
    console.print(
        Panel(
            f"[green]PyWork project initialized.[/green]\n\n"
            f"Workspace:    {result.workspace_path}\n"
            f"Project Root: {result.project_root}\n"
            f"PyWork Dir:   {result.pywork_dir}",
            title="PyWork Init",
            border_style="green",
        )
    )

    table = Table(title="Init Actions", show_header=True)
    table.add_column("Status")
    table.add_column("Kind")
    table.add_column("Path")

    for action in result.actions:
        if action.status == "created":
            status = "[green]created[/green]"
        elif action.status == "overwritten":
            status = "[yellow]overwritten[/yellow]"
        else:
            status = "[dim]exists[/dim]"

        table.add_row(status, action.kind, action.path)

    console.print(table)


def run_init(
    workspace: str | Path = ".",
    *,
    force: bool = False,
    quiet: bool = False,
    create_rules_file: bool = True,
    json_output: bool = False,
) -> int:
    try:
        result = initialize_project(
            workspace=workspace,
            force=force,
            create_rules_file=create_rules_file,
        )
    except WorkspaceError as exc:
        console.print(f"[red]Workspace error:[/red] {exc}")
        return 1
    except InitError as exc:
        console.print(f"[red]Init error:[/red] {exc}")
        return 1
    except Exception as exc:
        console.print(f"[red]Unexpected init error:[/red] {exc}")
        return 1

    if json_output:
        console.print(
            json.dumps(
                init_result_to_dict(result),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if not quiet:
        print_init_report(result)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pywork-init",
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
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )

    parser.add_argument(
        "--no-rules",
        action="store_true",
        help="Do not create PYWORK.md.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raise SystemExit(
        run_init(
            workspace=args.workspace,
            force=args.force,
            quiet=args.quiet,
            create_rules_file=not args.no_rules,
            json_output=args.json,
        )
    )


if __name__ == "__main__":
    main()