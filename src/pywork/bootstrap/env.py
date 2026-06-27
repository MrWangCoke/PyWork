from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


MIN_PYTHON_VERSION = (3, 12)
MAX_PYTHON_VERSION_EXCLUSIVE = (3, 14)


@dataclass(frozen=True)
class PythonVersionCheck:
    ok: bool
    current: str
    required: str
    reason: str


@dataclass(frozen=True)
class EnvironmentInfo:
    python_version: str
    python_version_tuple: tuple[int, int, int]
    python_implementation: str
    python_executable: str

    os_name: str
    os_system: str
    os_release: str
    os_version: str
    machine: str
    processor: str

    is_windows: bool
    is_macos: bool
    is_linux: bool
    is_wsl: bool

    cwd: str
    home: str
    path_separator: str

    shell: str | None
    terminal: str | None

    default_encoding: str
    filesystem_encoding: str

    is_virtual_env: bool
    virtual_env_path: str | None


def get_python_version_tuple() -> tuple[int, int, int]:
    info = sys.version_info
    return info.major, info.minor, info.micro


def format_version(version: tuple[int, int] | tuple[int, int, int]) -> str:
    return ".".join(str(part) for part in version)


def check_python_version() -> PythonVersionCheck:
    current_tuple = get_python_version_tuple()

    min_ok = current_tuple >= (*MIN_PYTHON_VERSION, 0)
    max_ok = current_tuple < (*MAX_PYTHON_VERSION_EXCLUSIVE, 0)
    ok = min_ok and max_ok

    current = format_version(current_tuple)
    required = (
        f">={format_version(MIN_PYTHON_VERSION)},"
        f"<{format_version(MAX_PYTHON_VERSION_EXCLUSIVE)}"
    )

    if ok:
        reason = "Python version is supported."
    elif not min_ok:
        reason = (
            f"Python {current} is too old. "
            f"PyWork requires Python {required}."
        )
    else:
        reason = (
            f"Python {current} is newer than the tested range. "
            f"PyWork currently expects Python {required}."
        )

    return PythonVersionCheck(
        ok=ok,
        current=current,
        required=required,
        reason=reason,
    )


def detect_wsl() -> bool:
    if platform.system().lower() != "linux":
        return False

    try:
        release = platform.release().lower()
        version = platform.version().lower()

        if "microsoft" in release or "microsoft" in version:
            return True

        os_release = Path("/proc/sys/kernel/osrelease")
        if os_release.exists():
            return "microsoft" in os_release.read_text(encoding="utf-8").lower()

    except OSError:
        return False

    return False


def detect_shell() -> str | None:
    if platform.system().lower() == "windows":
        shell = os.environ.get("ComSpec")
        ps_module_path = os.environ.get("PSModulePath")

        if ps_module_path:
            return os.environ.get("SHELL") or "powershell"

        return shell

    return os.environ.get("SHELL")


def detect_terminal() -> str | None:
    candidates = [
        "WT_SESSION",          # Windows Terminal
        "TERM_PROGRAM",        # macOS / modern terminals
        "TERM",                # Unix terminal type
        "ConEmuPID",           # ConEmu
        "VSCODE_PID",          # VS Code integrated terminal
    ]

    values: list[str] = []

    for key in candidates:
        value = os.environ.get(key)
        if value:
            values.append(f"{key}={value}")

    if not values:
        return None

    return "; ".join(values)


def is_running_in_virtual_env() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or sys.prefix != sys.base_prefix
        or bool(os.environ.get("VIRTUAL_ENV"))
        or bool(os.environ.get("CONDA_PREFIX"))
    )


def get_virtual_env_path() -> str | None:
    return (
        os.environ.get("VIRTUAL_ENV")
        or os.environ.get("CONDA_PREFIX")
        or (sys.prefix if is_running_in_virtual_env() else None)
    )


def collect_environment_info() -> EnvironmentInfo:
    system = platform.system()
    version_tuple = get_python_version_tuple()

    return EnvironmentInfo(
        python_version=platform.python_version(),
        python_version_tuple=version_tuple,
        python_implementation=platform.python_implementation(),
        python_executable=sys.executable,

        os_name=os.name,
        os_system=system,
        os_release=platform.release(),
        os_version=platform.version(),
        machine=platform.machine(),
        processor=platform.processor(),

        is_windows=system.lower() == "windows",
        is_macos=system.lower() == "darwin",
        is_linux=system.lower() == "linux",
        is_wsl=detect_wsl(),

        cwd=str(Path.cwd()),
        home=str(Path.home()),
        path_separator=os.sep,

        shell=detect_shell(),
        terminal=detect_terminal(),

        default_encoding=sys.getdefaultencoding(),
        filesystem_encoding=sys.getfilesystemencoding(),

        is_virtual_env=is_running_in_virtual_env(),
        virtual_env_path=get_virtual_env_path(),
    )


def environment_as_dict() -> dict[str, Any]:
    return asdict(collect_environment_info())


def print_environment_report() -> None:
    version_check = check_python_version()
    env = collect_environment_info()

    print("PyWork Environment Report")
    print("=" * 32)

    print(f"Python:        {env.python_version}")
    print(f"Implementation:{env.python_implementation}")
    print(f"Executable:    {env.python_executable}")
    print(f"Version OK:    {version_check.ok}")
    print(f"Required:      {version_check.required}")
    print(f"Reason:        {version_check.reason}")
    print()

    print(f"OS:            {env.os_system}")
    print(f"OS Release:    {env.os_release}")
    print(f"Machine:       {env.machine}")
    print(f"Processor:     {env.processor}")
    print(f"WSL:           {env.is_wsl}")
    print()

    print(f"CWD:           {env.cwd}")
    print(f"Home:          {env.home}")
    print(f"Shell:         {env.shell}")
    print(f"Terminal:      {env.terminal}")
    print()

    print(f"Virtual Env:   {env.is_virtual_env}")
    print(f"Venv Path:     {env.virtual_env_path}")
    print(f"Encoding:      {env.default_encoding}")
    print(f"FS Encoding:   {env.filesystem_encoding}")


def main() -> int:
    print_environment_report()
    return 0 if check_python_version().ok else 1


if __name__ == "__main__":
    raise SystemExit(main())