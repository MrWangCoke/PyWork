from __future__ import annotations

from dataclasses import dataclass


APP_NAME = "pywork"
APP_DISPLAY_NAME = "PyWork"
APP_VERSION = "0.1.0"
APP_COMMAND = "pywork"

APP_DESCRIPTION = "A Python TUI coding agent workspace."
APP_FULL_DESCRIPTION = (
    "PyWork is a Python implementation of a TUI coding agent workspace, "
    "with workspace loading, config management, permissions, tools, "
    "multi-agent runtime, and sandbox support."
)

APP_AUTHOR = "PyWork Contributors"

PROJECT_CONFIG_DIR_NAME = ".pywork"
PROJECT_RULES_FILE_NAME = "PYWORK.md"
PROJECT_MEMORY_FILE_NAME = "MEMORY.md"

DEFAULT_ENCODING = "utf-8"

DEFAULT_LOG_LEVEL = "INFO"

SUPPORTED_PYTHON_MIN_VERSION = (3, 12)
SUPPORTED_PYTHON_MAX_VERSION_EXCLUSIVE = (3, 14)


@dataclass(frozen=True)
class AppInfo:
    name: str
    display_name: str
    version: str
    command: str
    description: str
    full_description: str
    author: str


def get_app_info() -> AppInfo:
    return AppInfo(
        name=APP_NAME,
        display_name=APP_DISPLAY_NAME,
        version=APP_VERSION,
        command=APP_COMMAND,
        description=APP_DESCRIPTION,
        full_description=APP_FULL_DESCRIPTION,
        author=APP_AUTHOR,
    )


def get_version_string() -> str:
    return f"{APP_NAME} {APP_VERSION}"


def main() -> int:
    info = get_app_info()

    print(f"Name:        {info.name}")
    print(f"Display:     {info.display_name}")
    print(f"Version:     {info.version}")
    print(f"Command:     {info.command}")
    print(f"Description: {info.description}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())