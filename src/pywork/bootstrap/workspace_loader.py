from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_MARKERS = [
    ".pywork",
    ".git",
    "pyproject.toml",
    "package.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "package-lock.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "settings.gradle",
    "pubspec.yaml",
    "README.md",
    "README.MD",
]


@dataclass(frozen=True)
class WorkspaceInfo:
    requested_path: str
    workspace_path: str
    project_root: str

    exists: bool
    is_directory: bool
    is_file: bool

    has_git: bool
    has_pywork_dir: bool
    has_readme: bool

    detected_markers: list[str]
    pywork_dir: str

    cwd: str


class WorkspaceError(RuntimeError):
    pass


def resolve_workspace_path(path: str | Path | None = None) -> Path:
    if path is None:
        return Path.cwd().resolve()

    return Path(path).expanduser().resolve()


def validate_workspace_path(path: Path) -> None:
    if not path.exists():
        raise WorkspaceError(f"Workspace path does not exist: {path}")

    if path.is_file():
        raise WorkspaceError(
            f"Workspace path must be a directory, got file: {path}"
        )

    if not path.is_dir():
        raise WorkspaceError(f"Workspace path is not a directory: {path}")


def find_project_root(start: Path) -> Path:
    """
    Find the nearest project root by walking upward.

    Priority:
    1. .pywork/
    2. .git/
    3. known project marker files
    4. fallback to start
    """
    start = start.resolve()

    for parent in [start, *start.parents]:
        if (parent / ".pywork").is_dir():
            return parent

    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent

    for parent in [start, *start.parents]:
        for marker in PROJECT_MARKERS:
            marker_path = parent / marker
            if marker_path.exists():
                return parent

    return start


def detect_project_markers(project_root: Path) -> list[str]:
    markers: list[str] = []

    for marker in PROJECT_MARKERS:
        if (project_root / marker).exists():
            markers.append(marker)

    return markers


def has_readme(project_root: Path) -> bool:
    return any(
        (project_root / name).exists()
        for name in ["README.md", "README.MD", "readme.md", "Readme.md"]
    )


def ensure_pywork_dir(project_root: Path) -> Path:
    pywork_dir = project_root / ".pywork"
    pywork_dir.mkdir(parents=True, exist_ok=True)

    subdirs = [
        "sessions",
        "memory",
        "tasks",
        "agents",
        "artifacts",
        "logs",
        "cache",
    ]

    for subdir in subdirs:
        (pywork_dir / subdir).mkdir(parents=True, exist_ok=True)

    return pywork_dir


def load_workspace(
    path: str | Path | None = None,
    create_pywork_dir: bool = True,
) -> WorkspaceInfo:
    requested = resolve_workspace_path(path)
    validate_workspace_path(requested)

    project_root = find_project_root(requested)
    markers = detect_project_markers(project_root)

    if create_pywork_dir:
        pywork_dir = ensure_pywork_dir(project_root)
    else:
        pywork_dir = project_root / ".pywork"

    return WorkspaceInfo(
        requested_path=str(requested),
        workspace_path=str(requested),
        project_root=str(project_root),

        exists=requested.exists(),
        is_directory=requested.is_dir(),
        is_file=requested.is_file(),

        has_git=(project_root / ".git").exists(),
        has_pywork_dir=pywork_dir.exists(),
        has_readme=has_readme(project_root),

        detected_markers=markers,
        pywork_dir=str(pywork_dir),

        cwd=str(Path.cwd().resolve()),
    )


def workspace_as_dict(
    path: str | Path | None = None,
    create_pywork_dir: bool = True,
) -> dict[str, Any]:
    return asdict(
        load_workspace(
            path=path,
            create_pywork_dir=create_pywork_dir,
        )
    )


def print_workspace_report(workspace: WorkspaceInfo) -> None:
    print("PyWork Workspace Report")
    print("=" * 32)

    print(f"Requested Path:   {workspace.requested_path}")
    print(f"Workspace Path:   {workspace.workspace_path}")
    print(f"Project Root:     {workspace.project_root}")
    print(f"PyWork Dir:       {workspace.pywork_dir}")
    print()

    print(f"Exists:           {workspace.exists}")
    print(f"Is Directory:     {workspace.is_directory}")
    print(f"Is File:          {workspace.is_file}")
    print()

    print(f"Has Git:          {workspace.has_git}")
    print(f"Has .pywork:      {workspace.has_pywork_dir}")
    print(f"Has README:       {workspace.has_readme}")
    print()

    print("Detected Markers:")
    if workspace.detected_markers:
        for marker in workspace.detected_markers:
            print(f"  - {marker}")
    else:
        print("  - none")

    print()
    print(f"CWD:              {workspace.cwd}")


def main() -> int:
    # 支持命令行传路径：
    # uv run python -m pywork.bootstrap.workspace_loader .
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else os.curdir

    try:
        workspace = load_workspace(path)
    except WorkspaceError as exc:
        print(f"Workspace error: {exc}")
        return 1

    print_workspace_report(workspace)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())