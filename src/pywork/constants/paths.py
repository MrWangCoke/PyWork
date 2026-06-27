from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pywork.constants.app import (
    PROJECT_CONFIG_DIR_NAME,
    PROJECT_MEMORY_FILE_NAME,
    PROJECT_RULES_FILE_NAME,
)


# ---------------------------------------------------------------------
# Environment variables
# ---------------------------------------------------------------------

PYWORK_HOME_ENV = "PYWORK_HOME"
PYWORK_CONFIG_DIR_ENV = "PYWORK_CONFIG_DIR"
PYWORK_WORKSPACE_ENV = "PYWORK_WORKSPACE"


# ---------------------------------------------------------------------
# Repository-level paths
# ---------------------------------------------------------------------

REPO_CONFIG_DIR_NAME = "config"

DEFAULT_CONFIG_FILE_NAMES = (
    "default.toml",
    "models.toml",
    "permissions.toml",
    "tools.toml",
    "mcp.toml",
)


# ---------------------------------------------------------------------
# Project-level paths
# ---------------------------------------------------------------------

PROJECT_CONFIG_FILE_NAME = "config.toml"
PROJECT_DATABASE_FILE_NAME = "pywork.sqlite3"
PROJECT_GITIGNORE_FILE_NAME = ".gitignore"

PROJECT_SESSIONS_DIR_NAME = "sessions"
PROJECT_MEMORY_DIR_NAME = "memory"
PROJECT_TASKS_DIR_NAME = "tasks"
PROJECT_AGENTS_DIR_NAME = "agents"
PROJECT_ARTIFACTS_DIR_NAME = "artifacts"
PROJECT_LOGS_DIR_NAME = "logs"
PROJECT_CACHE_DIR_NAME = "cache"
PROJECT_TMP_DIR_NAME = "tmp"
PROJECT_CHECKPOINTS_DIR_NAME = "checkpoints"

PROJECT_RUNTIME_DIR_NAMES = (
    PROJECT_SESSIONS_DIR_NAME,
    PROJECT_MEMORY_DIR_NAME,
    PROJECT_TASKS_DIR_NAME,
    PROJECT_AGENTS_DIR_NAME,
    PROJECT_ARTIFACTS_DIR_NAME,
    PROJECT_LOGS_DIR_NAME,
    PROJECT_CACHE_DIR_NAME,
    PROJECT_TMP_DIR_NAME,
    PROJECT_CHECKPOINTS_DIR_NAME,
)


# ---------------------------------------------------------------------
# Common instruction files
# ---------------------------------------------------------------------

INSTRUCTION_FILE_NAMES = (
    PROJECT_RULES_FILE_NAME,
    "CLAUDE.md",
    "AGENTS.md",
    "README.md",
)


# ---------------------------------------------------------------------
# Project markers
# ---------------------------------------------------------------------

PROJECT_MARKER_NAMES = (
    PROJECT_CONFIG_DIR_NAME,
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
)


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    pywork_dir: Path

    config_file: Path
    memory_file: Path
    rules_file: Path
    database_file: Path
    gitignore_file: Path

    sessions_dir: Path
    memory_dir: Path
    tasks_dir: Path
    agents_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    cache_dir: Path
    tmp_dir: Path
    checkpoints_dir: Path

    instruction_files: tuple[Path, ...]


@dataclass(frozen=True)
class UserPaths:
    home_dir: Path
    pywork_home: Path
    user_config_file: Path
    user_logs_dir: Path
    user_cache_dir: Path


def get_package_root() -> Path:
    """
    src/pywork
    """
    return Path(__file__).resolve().parents[1]


def get_source_root() -> Path:
    """
    src
    """
    return Path(__file__).resolve().parents[2]


def get_repo_root() -> Path:
    """
    Development repo root.

    In current project layout:
        src/pywork/constants/paths.py
        parents[3] == project root
    """
    return Path(__file__).resolve().parents[3]


def get_repo_config_dir() -> Path:
    return get_repo_root() / REPO_CONFIG_DIR_NAME


def get_repo_default_config_paths() -> tuple[Path, ...]:
    config_dir = get_repo_config_dir()
    return tuple(config_dir / file_name for file_name in DEFAULT_CONFIG_FILE_NAMES)


def get_home_dir() -> Path:
    return Path.home()


def get_pywork_home() -> Path:
    env_value = os.environ.get(PYWORK_HOME_ENV)

    if env_value:
        return Path(env_value).expanduser().resolve()

    return get_home_dir() / PROJECT_CONFIG_DIR_NAME


def get_user_paths() -> UserPaths:
    pywork_home = get_pywork_home()

    return UserPaths(
        home_dir=get_home_dir(),
        pywork_home=pywork_home,
        user_config_file=pywork_home / PROJECT_CONFIG_FILE_NAME,
        user_logs_dir=pywork_home / PROJECT_LOGS_DIR_NAME,
        user_cache_dir=pywork_home / PROJECT_CACHE_DIR_NAME,
    )


def get_user_config_file() -> Path:
    return get_user_paths().user_config_file


def resolve_workspace_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()

    env_value = os.environ.get(PYWORK_WORKSPACE_ENV)

    if env_value:
        return Path(env_value).expanduser().resolve()

    return Path.cwd().resolve()


def find_project_root(start: str | Path | None = None) -> Path:
    current = resolve_workspace_path(start)

    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        for marker in PROJECT_MARKER_NAMES:
            if (parent / marker).exists():
                return parent

    return current


def get_project_pywork_dir(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / PROJECT_CONFIG_DIR_NAME


def get_project_config_file(project_root: str | Path) -> Path:
    return get_project_pywork_dir(project_root) / PROJECT_CONFIG_FILE_NAME


def get_project_memory_file(project_root: str | Path) -> Path:
    return get_project_pywork_dir(project_root) / PROJECT_MEMORY_FILE_NAME


def get_project_rules_file(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / PROJECT_RULES_FILE_NAME


def get_project_database_file(project_root: str | Path) -> Path:
    return get_project_pywork_dir(project_root) / PROJECT_DATABASE_FILE_NAME


def get_project_runtime_dir(project_root: str | Path, name: str) -> Path:
    return get_project_pywork_dir(project_root) / name


def get_instruction_files(project_root: str | Path) -> tuple[Path, ...]:
    root = Path(project_root).expanduser().resolve()

    return tuple(root / file_name for file_name in INSTRUCTION_FILE_NAMES)


def build_project_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = find_project_root(project_root)
    pywork_dir = root / PROJECT_CONFIG_DIR_NAME

    return ProjectPaths(
        project_root=root,
        pywork_dir=pywork_dir,

        config_file=pywork_dir / PROJECT_CONFIG_FILE_NAME,
        memory_file=pywork_dir / PROJECT_MEMORY_FILE_NAME,
        rules_file=root / PROJECT_RULES_FILE_NAME,
        database_file=pywork_dir / PROJECT_DATABASE_FILE_NAME,
        gitignore_file=pywork_dir / PROJECT_GITIGNORE_FILE_NAME,

        sessions_dir=pywork_dir / PROJECT_SESSIONS_DIR_NAME,
        memory_dir=pywork_dir / PROJECT_MEMORY_DIR_NAME,
        tasks_dir=pywork_dir / PROJECT_TASKS_DIR_NAME,
        agents_dir=pywork_dir / PROJECT_AGENTS_DIR_NAME,
        artifacts_dir=pywork_dir / PROJECT_ARTIFACTS_DIR_NAME,
        logs_dir=pywork_dir / PROJECT_LOGS_DIR_NAME,
        cache_dir=pywork_dir / PROJECT_CACHE_DIR_NAME,
        tmp_dir=pywork_dir / PROJECT_TMP_DIR_NAME,
        checkpoints_dir=pywork_dir / PROJECT_CHECKPOINTS_DIR_NAME,

        instruction_files=get_instruction_files(root),
    )


def ensure_user_dirs() -> UserPaths:
    paths = get_user_paths()

    paths.pywork_home.mkdir(parents=True, exist_ok=True)
    paths.user_logs_dir.mkdir(parents=True, exist_ok=True)
    paths.user_cache_dir.mkdir(parents=True, exist_ok=True)

    return paths


def ensure_project_dirs(project_root: str | Path | None = None) -> ProjectPaths:
    paths = build_project_paths(project_root)

    paths.pywork_dir.mkdir(parents=True, exist_ok=True)

    for dir_name in PROJECT_RUNTIME_DIR_NAMES:
        (paths.pywork_dir / dir_name).mkdir(parents=True, exist_ok=True)

    return paths


def path_to_string(path: Path) -> str:
    return str(path)


def project_paths_to_dict(paths: ProjectPaths) -> dict[str, Any]:
    data = asdict(paths)

    return {
        key: (
            [str(item) for item in value]
            if isinstance(value, tuple)
            else str(value)
        )
        for key, value in data.items()
    }


def user_paths_to_dict(paths: UserPaths) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in asdict(paths).items()
    }


def main() -> int:
    print("PyWork Paths")
    print("=" * 32)

    print()
    print("Repo:")
    print(f"  package_root: {get_package_root()}")
    print(f"  source_root:  {get_source_root()}")
    print(f"  repo_root:    {get_repo_root()}")
    print(f"  config_dir:   {get_repo_config_dir()}")

    print()
    print("Default config files:")
    for path in get_repo_default_config_paths():
        print(f"  - {path}")

    print()
    print("User:")
    for key, value in user_paths_to_dict(get_user_paths()).items():
        print(f"  {key}: {value}")

    print()
    print("Project:")
    for key, value in project_paths_to_dict(build_project_paths(".")).items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())