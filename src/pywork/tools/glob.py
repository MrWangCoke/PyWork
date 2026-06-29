from __future__ import annotations

import asyncio
import fnmatch
import inspect
from pathlib import Path, PureWindowsPath
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.file_read import get_context_workspace_path, make_relative_path
from pywork.tools.tool import (
    BaseTool,
    ToolExecutionContext,
    ToolExecutionError,
    ToolValidationError,
)


DEFAULT_MAX_RESULTS = 200

DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/**",
    "**/.git/**",
    ".venv",
    ".venv/**",
    "**/.venv/**",
    "venv",
    "venv/**",
    "**/venv/**",
    "__pycache__",
    "__pycache__/**",
    "**/__pycache__/**",
    "node_modules",
    "node_modules/**",
    "**/node_modules/**",
    ".mypy_cache",
    ".mypy_cache/**",
    "**/.mypy_cache/**",
    ".pytest_cache",
    ".pytest_cache/**",
    "**/.pytest_cache/**",
    "dist",
    "dist/**",
    "build",
    "build/**",
]


def coerce_int(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{name} must be an integer") from exc

    if minimum is not None and result < minimum:
        raise ToolValidationError(f"{name} must be >= {minimum}")

    if maximum is not None and result > maximum:
        raise ToolValidationError(f"{name} must be <= {maximum}")

    return result


def coerce_bool(
    value: Any,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "y", "on"}:
            return True

        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    return bool(value)


def coerce_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []

    if isinstance(value, list):
        result: list[str] = []

        for item in value:
            text = str(item).strip()

            if text:
                result.append(text)

        return result

    raise ToolValidationError("ignore_patterns must be a string or list of strings")


def normalize_glob_pattern(pattern: str) -> str:
    pattern = pattern.strip().replace("\\", "/")

    if not pattern:
        raise ToolValidationError("pattern cannot be empty")

    if "\x00" in pattern:
        raise ToolValidationError("pattern contains null byte")

    if Path(pattern).is_absolute() or PureWindowsPath(pattern).is_absolute():
        raise ToolValidationError("pattern must be relative to workspace/base path")

    if PureWindowsPath(pattern).drive:
        raise ToolValidationError("pattern must not contain a Windows drive")

    parts = [
        part
        for part in pattern.split("/")
        if part
    ]

    if any(part == ".." for part in parts):
        raise ToolValidationError("pattern must not contain '..'")

    return pattern


def resolve_base_path_in_workspace(
    path_value: str | None,
    *,
    workspace_path: Path,
) -> Path:
    raw_path = (path_value or ".").strip() or "."
    candidate = Path(raw_path).expanduser()

    if not candidate.is_absolute():
        candidate = workspace_path / candidate

    resolved = candidate.resolve()

    try:
        resolved.relative_to(workspace_path)
    except ValueError as exc:
        raise ToolValidationError(
            f"base path is outside workspace: {raw_path}"
        ) from exc

    if not resolved.exists():
        raise ToolValidationError(f"base path does not exist: {raw_path}")

    if not resolved.is_dir():
        raise ToolValidationError(f"base path is not a directory: {raw_path}")

    return resolved


def is_hidden_path(
    path: Path,
    *,
    workspace_path: Path,
) -> bool:
    relative = make_relative_path(path, workspace_path=workspace_path)

    for part in relative.split("/"):
        if part.startswith(".") and part not in {".", ".."}:
            return True

    return False


def matches_ignore_pattern(
    relative_path: str,
    *,
    ignore_patterns: list[str],
) -> bool:
    relative_path = relative_path.replace("\\", "/")

    for pattern in ignore_patterns:
        pattern = pattern.replace("\\", "/").strip()

        if not pattern:
            continue

        if fnmatch.fnmatchcase(relative_path, pattern):
            return True

        if fnmatch.fnmatchcase(Path(relative_path).name, pattern):
            return True

    return False


def path_kind(path: Path) -> str:
    if path.is_file():
        return "file"

    if path.is_dir():
        return "directory"

    if path.is_symlink():
        return "symlink"

    return "other"


def format_match_path(
    path: Path,
    *,
    workspace_path: Path,
    absolute_paths: bool,
) -> str:
    if absolute_paths:
        text = str(path)

        if path.is_dir():
            return text + "\\"

        return text

    text = make_relative_path(
        path,
        workspace_path=workspace_path,
    )

    if path.is_dir():
        return text.rstrip("/") + "/"

    return text


class GlobTool(BaseTool):
    """
    根据 glob 模式匹配 workspace 内的文件。
    """

    name: ClassVar[str] = "glob"
    description: ClassVar[str] = (
        "Find files or directories inside the current workspace using a glob pattern."
    )
    risk_level: ClassVar[ToolRiskLevel] = ToolRiskLevel.SAFE

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern relative to the base path, for example '**/*.py' or 'src/**/*.py'.",
            },
            "path": {
                "type": "string",
                "description": "Base directory relative to workspace. Defaults to workspace root.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of paths to return. Defaults to 200.",
                "minimum": 1,
                "maximum": 10000,
            },
            "include_files": {
                "type": "boolean",
                "description": "Whether to include files. Defaults to true.",
            },
            "include_dirs": {
                "type": "boolean",
                "description": "Whether to include directories. Defaults to false.",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Whether to include dot files and dot directories. Defaults to false.",
            },
            "absolute_paths": {
                "type": "boolean",
                "description": "Whether to return absolute paths. Defaults to false.",
            },
            "follow_symlinks": {
                "type": "boolean",
                "description": "Whether to include symlinks. Defaults to false.",
            },
            "use_default_ignores": {
                "type": "boolean",
                "description": "Whether to ignore .git, .venv, node_modules, __pycache__, etc. Defaults to true.",
            },
            "ignore_patterns": {
                "type": "array",
                "description": "Extra ignore glob patterns relative to workspace.",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": ["pattern"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = call.arguments

        workspace_path = get_context_workspace_path(context)

        pattern = normalize_glob_pattern(
            str(args.get("pattern", "")).strip()
        )

        base_path = resolve_base_path_in_workspace(
            args.get("path"),
            workspace_path=workspace_path,
        )

        max_results = coerce_int(
            args.get("max_results"),
            name="max_results",
            default=DEFAULT_MAX_RESULTS,
            minimum=1,
            maximum=10000,
        )

        include_files = coerce_bool(
            args.get("include_files"),
            default=True,
        )

        include_dirs = coerce_bool(
            args.get("include_dirs"),
            default=False,
        )

        include_hidden = coerce_bool(
            args.get("include_hidden"),
            default=False,
        )

        absolute_paths = coerce_bool(
            args.get("absolute_paths"),
            default=False,
        )

        follow_symlinks = coerce_bool(
            args.get("follow_symlinks"),
            default=False,
        )

        use_default_ignores = coerce_bool(
            args.get("use_default_ignores"),
            default=True,
        )

        ignore_patterns = coerce_string_list(
            args.get("ignore_patterns")
        )

        if use_default_ignores:
            ignore_patterns = DEFAULT_IGNORE_PATTERNS + ignore_patterns

        result_data = await asyncio.to_thread(
            self._glob,
            base_path,
            workspace_path=workspace_path,
            pattern=pattern,
            max_results=max_results,
            include_files=include_files,
            include_dirs=include_dirs,
            include_hidden=include_hidden,
            absolute_paths=absolute_paths,
            follow_symlinks=follow_symlinks,
            ignore_patterns=ignore_patterns,
        )

        return ToolResult.success_result(
            call=call,
            content=result_data["content"],
            data=result_data,
        )

    def _glob(
        self,
        base_path: Path,
        *,
        workspace_path: Path,
        pattern: str,
        max_results: int,
        include_files: bool,
        include_dirs: bool,
        include_hidden: bool,
        absolute_paths: bool,
        follow_symlinks: bool,
        ignore_patterns: list[str],
    ) -> dict[str, Any]:
        matches: list[dict[str, Any]] = []
        truncated = False

        try:
            iterator = base_path.glob(pattern)
        except ValueError as exc:
            raise ToolValidationError(f"invalid glob pattern: {pattern}") from exc

        try:
            for item in iterator:
                try:
                    resolved = item.resolve()
                except OSError:
                    continue

                try:
                    resolved.relative_to(workspace_path)
                except ValueError:
                    continue

                if item.is_symlink() and not follow_symlinks:
                    continue

                kind = path_kind(item)

                if kind == "file" and not include_files:
                    continue

                if kind == "directory" and not include_dirs:
                    continue

                if kind not in {"file", "directory"}:
                    continue

                relative_path = make_relative_path(
                    item,
                    workspace_path=workspace_path,
                )

                if not include_hidden and is_hidden_path(
                    item,
                    workspace_path=workspace_path,
                ):
                    continue

                if matches_ignore_pattern(
                    relative_path,
                    ignore_patterns=ignore_patterns,
                ):
                    continue

                matches.append(
                    {
                        "path": format_match_path(
                            item,
                            workspace_path=workspace_path,
                            absolute_paths=absolute_paths,
                        ),
                        "relative_path": relative_path,
                        "absolute_path": str(item),
                        "kind": kind,
                    }
                )

                if len(matches) >= max_results:
                    truncated = True
                    break

        except OSError as exc:
            raise ToolExecutionError(f"glob failed: {exc}") from exc

        matches.sort(
            key=lambda item: (
                item["kind"],
                item["relative_path"].lower(),
            )
        )

        paths = [
            item["path"]
            for item in matches
        ]

        base_relative = make_relative_path(
            base_path,
            workspace_path=workspace_path,
        )

        content = self._format_content(
            pattern=pattern,
            base_path=base_relative,
            paths=paths,
            truncated=truncated,
            max_results=max_results,
        )

        return {
            "pattern": pattern,
            "base_path": base_relative,
            "workspace_path": str(workspace_path),
            "matches": matches,
            "paths": paths,
            "count": len(paths),
            "max_results": max_results,
            "truncated": truncated,
            "include_files": include_files,
            "include_dirs": include_dirs,
            "include_hidden": include_hidden,
            "absolute_paths": absolute_paths,
            "follow_symlinks": follow_symlinks,
            "ignore_patterns": ignore_patterns,
            "content": content,
        }

    def _format_content(
        self,
        *,
        pattern: str,
        base_path: str,
        paths: list[str],
        truncated: bool,
        max_results: int,
    ) -> str:
        header = [
            f"# glob: {pattern}",
            f"# base: {base_path}",
            f"# matches: {len(paths)}",
        ]

        if truncated:
            header.append(f"# truncated: true, max_results={max_results}")

        if not paths:
            return "\n".join(header + ["", "No matches found."])

        return "\n".join(header + ["", *paths])

    def render_result(self, result: ToolResult) -> str:
        if not result.success:
            return result.error or "glob failed"

        return result.content


def create_demo_context() -> ToolExecutionContext:
    signature = inspect.signature(ToolExecutionContext)
    kwargs: dict[str, Any] = {}

    if "workspace_path" in signature.parameters:
        kwargs["workspace_path"] = Path.cwd()

    if "project_root" in signature.parameters:
        kwargs["project_root"] = Path.cwd()

    if "config" in signature.parameters:
        kwargs["config"] = {}

    if "metadata" in signature.parameters:
        kwargs["metadata"] = {}

    return ToolExecutionContext(**kwargs)


async def demo() -> None:
    tool = GlobTool()
    context = create_demo_context()

    call = tool.create_call(
        {
            "pattern": "src/pywork/**/*.py",
            "max_results": 40,
            "include_dirs": False,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    print(tool.render_result(result))


def main() -> int:
    asyncio.run(demo())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())