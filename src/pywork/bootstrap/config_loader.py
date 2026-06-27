from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_FILE_NAMES = [
    "default.toml",
    "models.toml",
    "permissions.toml",
    "tools.toml",
    "mcp.toml",
]


@dataclass(frozen=True)
class ConfigSource:
    path: Path
    exists: bool
    loaded: bool
    error: str | None = None


@dataclass(frozen=True)
class ConfigLoadResult:
    config: dict[str, Any]
    sources: list[ConfigSource]


class ConfigError(RuntimeError):
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override into base.

    Example:
        base = {"app": {"name": "pywork", "debug": False}}
        override = {"app": {"debug": True}}
        result = {"app": {"name": "pywork", "debug": True}}
    """
    result = dict(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_toml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    if not path.is_file():
        raise ConfigError(f"Config path is not a file: {path}")

    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML file: {path}\n{exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Failed to read config file: {path}\n{exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file must contain a TOML table: {path}")

    return data


def find_project_root(start: Path | None = None) -> Path:
    """
    Find the nearest directory that looks like a project root.

    Priority:
    1. directory containing .pywork/
    2. directory containing pyproject.toml
    3. directory containing .git/
    4. current working directory
    """
    current = (start or Path.cwd()).resolve()

    if current.is_file():
        current = current.parent

    for parent in [current, *current.parents]:
        if (parent / ".pywork").exists():
            return parent

    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent

    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent

    return current


def get_repo_config_dir() -> Path:
    """
    During development, config/ is at project root.

    File path:
        src/pywork/bootstrap/config_loader.py

    Project root:
        parents[3]
    """
    return Path(__file__).resolve().parents[3] / "config"


def get_user_config_path() -> Path:
    return Path.home() / ".pywork" / "config.toml"


def get_project_config_path(workspace: Path | None = None) -> Path:
    project_root = find_project_root(workspace)
    return project_root / ".pywork" / "config.toml"


def discover_config_paths(
    workspace: Path | None = None,
    include_user: bool = True,
    include_project: bool = True,
) -> list[Path]:
    paths: list[Path] = []

    repo_config_dir = get_repo_config_dir()

    for file_name in CONFIG_FILE_NAMES:
        paths.append(repo_config_dir / file_name)

    if include_user:
        paths.append(get_user_config_path())

    if include_project:
        paths.append(get_project_config_path(workspace))

    return paths


def env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_env_overrides() -> dict[str, Any]:
    """
    Supported environment variables:

    PYWORK_MODEL_PROVIDER
    PYWORK_MODEL_NAME
    PYWORK_BASE_URL
    PYWORK_PERMISSION_MODE
    PYWORK_LOG_LEVEL
    PYWORK_ENABLE_MCP
    """
    overrides: dict[str, Any] = {}

    model_provider = os.environ.get("PYWORK_MODEL_PROVIDER")
    model_name = os.environ.get("PYWORK_MODEL_NAME")
    base_url = os.environ.get("PYWORK_BASE_URL")

    if model_provider or model_name or base_url:
        overrides.setdefault("default", {})

        if model_provider:
            overrides["default"]["provider"] = model_provider

        if model_name:
            overrides["default"]["model"] = model_name

        if base_url:
            overrides["default"]["base_url"] = base_url

    permission_mode = os.environ.get("PYWORK_PERMISSION_MODE")
    if permission_mode:
        overrides.setdefault("permissions", {})
        overrides["permissions"]["mode"] = permission_mode

    log_level = os.environ.get("PYWORK_LOG_LEVEL")
    if log_level:
        overrides.setdefault("app", {})
        overrides["app"]["log_level"] = log_level

    enable_mcp = os.environ.get("PYWORK_ENABLE_MCP")
    if enable_mcp is not None:
        overrides.setdefault("mcp", {})
        overrides["mcp"]["enabled"] = env_bool(enable_mcp)

    return overrides


def load_config(
    workspace: Path | None = None,
    include_user: bool = True,
    include_project: bool = True,
    include_env: bool = True,
) -> ConfigLoadResult:
    config: dict[str, Any] = {}
    sources: list[ConfigSource] = []

    for path in discover_config_paths(
        workspace=workspace,
        include_user=include_user,
        include_project=include_project,
    ):
        if not path.exists():
            sources.append(
                ConfigSource(
                    path=path,
                    exists=False,
                    loaded=False,
                )
            )
            continue

        try:
            data = load_toml_file(path)
            config = deep_merge(config, data)
            sources.append(
                ConfigSource(
                    path=path,
                    exists=True,
                    loaded=True,
                )
            )
        except ConfigError as exc:
            sources.append(
                ConfigSource(
                    path=path,
                    exists=True,
                    loaded=False,
                    error=str(exc),
                )
            )
            raise

    if include_env:
        config = deep_merge(config, get_env_overrides())

    return ConfigLoadResult(
        config=config,
        sources=sources,
    )


def get_config_value(
    config: dict[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    """
    Example:
        get_config_value(config, "app.name")
        get_config_value(config, "permissions.mode")
    """
    current: Any = config

    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def print_config_report(result: ConfigLoadResult) -> None:
    print("PyWork Config Report")
    print("=" * 32)

    print("Sources:")
    for source in result.sources:
        status = "loaded" if source.loaded else "missing"
        if source.exists and not source.loaded:
            status = "failed"

        print(f"- {status}: {source.path}")

        if source.error:
            print(f"  error: {source.error}")

    print()
    print("Effective config:")
    print(result.config)


def main() -> int:
    result = load_config()
    print_config_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())