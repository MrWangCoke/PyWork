from __future__ import annotations

import importlib
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal


DependencyKind = Literal["python", "command"]
DependencyLevel = Literal["required", "optional"]


@dataclass(frozen=True)
class DependencySpec:
    name: str
    kind: DependencyKind
    level: DependencyLevel
    import_name: str | None = None
    command: str | None = None
    version_args: tuple[str, ...] = ("--version",)
    description: str = ""


@dataclass(frozen=True)
class DependencyCheckResult:
    name: str
    kind: DependencyKind
    level: DependencyLevel
    ok: bool
    detail: str


PYTHON_DEPENDENCIES: list[DependencySpec] = [
    DependencySpec("typer", "python", "required", import_name="typer", description="CLI framework"),
    DependencySpec("rich", "python", "required", import_name="rich", description="Terminal rendering"),
    DependencySpec("textual", "python", "required", import_name="textual", description="TUI framework"),
    DependencySpec("pydantic", "python", "required", import_name="pydantic", description="Schema models"),
    DependencySpec(
        "pydantic-settings",
        "python",
        "required",
        import_name="pydantic_settings",
        description="Settings management",
    ),
    DependencySpec("aiosqlite", "python", "required", import_name="aiosqlite", description="SQLite async"),
    DependencySpec("httpx", "python", "required", import_name="httpx", description="HTTP client"),
    DependencySpec("orjson", "python", "required", import_name="orjson", description="Fast JSON"),
    DependencySpec("psutil", "python", "required", import_name="psutil", description="Process control"),
    DependencySpec("openai", "python", "required", import_name="openai", description="OpenAI SDK"),
    DependencySpec("anthropic", "python", "required", import_name="anthropic", description="Anthropic SDK"),
    DependencySpec(
        "langchain-core",
        "python",
        "required",
        import_name="langchain_core",
        description="LangChain core abstractions",
    ),
    DependencySpec(
        "langchain-openai",
        "python",
        "required",
        import_name="langchain_openai",
        description="LangChain OpenAI adapter",
    ),
    DependencySpec("langgraph", "python", "required", import_name="langgraph", description="Agent graph runtime"),
    DependencySpec(
        "langgraph-checkpoint-sqlite",
        "python",
        "required",
        import_name="langgraph.checkpoint.sqlite",
        description="LangGraph SQLite checkpoint",
    ),
    DependencySpec(
        "langgraph-supervisor",
        "python",
        "optional",
        import_name="langgraph_supervisor",
        description="Supervisor/worker multi-agent helper",
    ),
    DependencySpec(
        "deepagents",
        "python",
        "optional",
        import_name="deepagents",
        description="Deep agent helpers",
    ),
    DependencySpec("mcp", "python", "optional", import_name="mcp", description="MCP Python SDK"),
    DependencySpec("GitPython", "python", "required", import_name="git", description="Git Python library"),
    DependencySpec("unidiff", "python", "required", import_name="unidiff", description="Unified diff parser"),
    DependencySpec("tree-sitter", "python", "optional", import_name="tree_sitter", description="Code parser"),
    DependencySpec("nbformat", "python", "optional", import_name="nbformat", description="Notebook support"),
]


COMMAND_DEPENDENCIES: list[DependencySpec] = [
    DependencySpec("git", "command", "required", command="git", description="Git operations"),
    DependencySpec("ripgrep", "command", "required", command="rg", description="Fast code search"),
    DependencySpec("python", "command", "required", command="python", description="Python executable"),
    DependencySpec(
        "powershell",
        "command",
        "optional",
        command="powershell",
        version_args=("-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"),
        description="Windows PowerShell",
    ),
    DependencySpec(
        "pwsh",
        "command",
        "optional",
        command="pwsh",
        version_args=("-NoProfile", "-Command", "$PSVersionTable.PSVersion.ToString()"),
    description="PowerShell Core",
    ),
    DependencySpec("docker", "command", "optional", command="docker", description="Container sandbox"),
    DependencySpec("wsl", "command", "optional", command="wsl", description="Windows WSL sandbox"),
]


def check_python_dependency(spec: DependencySpec) -> DependencyCheckResult:
    import_name = spec.import_name or spec.name

    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return DependencyCheckResult(
            name=spec.name,
            kind=spec.kind,
            level=spec.level,
            ok=False,
            detail=f"import failed: {exc}",
        )

    version = getattr(module, "__version__", None)

    if version:
        detail = f"import ok, version={version}"
    else:
        detail = "import ok"

    return DependencyCheckResult(
        name=spec.name,
        kind=spec.kind,
        level=spec.level,
        ok=True,
        detail=detail,
    )


def run_version_command(command: str, args: tuple[str, ...]) -> str:
    try:
        completed = subprocess.run(
            [command, *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        return f"version check failed: {exc}"

    output = (completed.stdout or completed.stderr).strip()
    first_line = output.splitlines()[0] if output else "version output empty"
    return first_line


def check_command_dependency(spec: DependencySpec) -> DependencyCheckResult:
    command = spec.command or spec.name
    resolved = shutil.which(command)

    if not resolved:
        return DependencyCheckResult(
            name=spec.name,
            kind=spec.kind,
            level=spec.level,
            ok=False,
            detail=f"command not found: {command}",
        )

    version_detail = run_version_command(command, spec.version_args)

    return DependencyCheckResult(
        name=spec.name,
        kind=spec.kind,
        level=spec.level,
        ok=True,
        detail=f"{resolved} | {version_detail}",
    )


def check_dependency(spec: DependencySpec) -> DependencyCheckResult:
    if spec.kind == "python":
        return check_python_dependency(spec)

    if spec.kind == "command":
        return check_command_dependency(spec)

    raise ValueError(f"Unknown dependency kind: {spec.kind}")


def check_all_dependencies() -> list[DependencyCheckResult]:
    specs = [*PYTHON_DEPENDENCIES, *COMMAND_DEPENDENCIES]
    return [check_dependency(spec) for spec in specs]


def has_required_failures(results: list[DependencyCheckResult]) -> bool:
    return any(result.level == "required" and not result.ok for result in results)


def print_dependency_report(results: list[DependencyCheckResult]) -> None:
    print("PyWork Dependency Report")
    print("=" * 32)

    print()
    print("Python packages:")
    for result in results:
        if result.kind != "python":
            continue

        mark = "OK" if result.ok else "FAIL"
        print(f"[{mark}] {result.name:<30} {result.level:<8} {result.detail}")

    print()
    print("External commands:")
    for result in results:
        if result.kind != "command":
            continue

        mark = "OK" if result.ok else "FAIL"
        print(f"[{mark}] {result.name:<30} {result.level:<8} {result.detail}")

    print()

    required_failed = [
        result
        for result in results
        if result.level == "required" and not result.ok
    ]

    optional_failed = [
        result
        for result in results
        if result.level == "optional" and not result.ok
    ]

    if required_failed:
        print("Required dependencies failed:")
        for result in required_failed:
            print(f"  - {result.name}: {result.detail}")

    if optional_failed:
        print("Optional dependencies missing:")
        for result in optional_failed:
            print(f"  - {result.name}: {result.detail}")

    if not required_failed:
        print("All required dependencies are available.")


def main() -> int:
    results = check_all_dependencies()
    print_dependency_report(results)

    return 1 if has_required_failures(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())