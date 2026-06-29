from __future__ import annotations

import asyncio
import inspect
import json
import shutil
from pathlib import Path
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.file_read import get_context_workspace_path, limit_content_chars, make_relative_path
from pywork.tools.glob import (
    coerce_bool,
    coerce_int,
    coerce_string_list,
    resolve_base_path_in_workspace,
)
from pywork.tools.tool import (
    BaseTool,
    ToolExecutionContext,
    ToolExecutionError,
    ToolValidationError,
)


DEFAULT_MAX_RESULTS = 100
DEFAULT_MAX_CHARS = 80_000
DEFAULT_TIMEOUT_SECONDS = 20


def require_ripgrep() -> str:
    rg_path = shutil.which("rg")

    if not rg_path:
        raise ToolExecutionError(
            "ripgrep is not installed or not found in PATH. "
            "Please install ripgrep and make sure `rg` is available."
        )

    return rg_path


def normalize_search_pattern(pattern: str) -> str:
    pattern = pattern.strip()

    if not pattern:
        raise ToolValidationError("pattern cannot be empty")

    if "\x00" in pattern:
        raise ToolValidationError("pattern contains null byte")

    return pattern


def normalize_rg_path(path: Path) -> str:
    return str(path)


def build_rg_command(
    *,
    rg_path: str,
    pattern: str,
    search_path: Path,
    max_results: int,
    context_lines: int,
    case_sensitive: bool,
    fixed_strings: bool,
    word_regexp: bool,
    include_hidden: bool,
    use_gitignore: bool,
    glob_patterns: list[str],
) -> list[str]:
    command = [
        rg_path,
        "--json",
        "--line-number",
        "--column",
        "--color",
        "never",
        "--no-heading",
        "--with-filename",
        "--max-count",
        str(max_results),
    ]

    if context_lines > 0:
        command.extend(["--context", str(context_lines)])

    if not case_sensitive:
        command.append("--ignore-case")

    if fixed_strings:
        command.append("--fixed-strings")

    if word_regexp:
        command.append("--word-regexp")

    if include_hidden:
        command.append("--hidden")

    if not use_gitignore:
        command.append("--no-ignore")

    for glob_pattern in glob_patterns:
        command.extend(["--glob", glob_pattern])

    command.extend(
        [
            "--regexp",
            pattern,
            normalize_rg_path(search_path),
        ]
    )

    return command


def parse_rg_json_line(
    line: str,
    *,
    workspace_path: Path,
) -> dict[str, Any] | None:
    line = line.strip()

    if not line:
        return None

    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None

    event_type = payload.get("type")
    data = payload.get("data", {})

    if event_type not in {"match", "context"}:
        return None

    raw_path = data.get("path", {}).get("text", "")

    if not raw_path:
        return None

    path = Path(raw_path)

    try:
        resolved = path.resolve()
    except OSError:
        resolved = path

    try:
        resolved.relative_to(workspace_path)
    except ValueError:
        return None

    relative_path = make_relative_path(
        resolved,
        workspace_path=workspace_path,
    )

    line_number = int(data.get("line_number") or 0)

    line_text = data.get("lines", {}).get("text", "")

    line_text = line_text.rstrip("\n").rstrip("\r")

    submatches = data.get("submatches", []) or []

    column = None
    match_text = ""

    if submatches:
        first_match = submatches[0]
        column = int(first_match.get("start", 0)) + 1
        match_text = first_match.get("match", {}).get("text", "")

    return {
        "type": event_type,
        "path": relative_path,
        "absolute_path": str(resolved),
        "line_number": line_number,
        "column": column,
        "line": line_text,
        "match_text": match_text,
        "submatches": submatches,
    }


def format_grep_matches(
    *,
    pattern: str,
    search_path: str,
    matches: list[dict[str, Any]],
    max_results: int,
    truncated_by_matches: bool,
) -> str:
    header = [
        f"# grep: {pattern}",
        f"# path: {search_path}",
        f"# matches: {len([item for item in matches if item['type'] == 'match'])}",
    ]

    if truncated_by_matches:
        header.append(f"# truncated: true, max_results={max_results}")

    if not matches:
        return "\n".join(header + ["", "No matches found."])

    body: list[str] = []

    for item in matches:
        event_type = item["type"]
        path = item["path"]
        line_number = item["line_number"]
        column = item["column"]
        line = item["line"]

        if event_type == "context":
            body.append(f"{path}:{line_number}:context: {line}")
            continue

        if column is None:
            body.append(f"{path}:{line_number}: {line}")
        else:
            body.append(f"{path}:{line_number}:{column}: {line}")

    return "\n".join(header + ["", *body])


class GrepTool(BaseTool):
    """
    使用 ripgrep 在 workspace 内进行正则搜索。
    """

    name: ClassVar[str] = "grep"
    description: ClassVar[str] = (
        "Search text in files inside the current workspace using ripgrep regular expressions."
    )
    risk_level: ClassVar[ToolRiskLevel] = ToolRiskLevel.SAFE

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path relative to workspace. Defaults to workspace root.",
            },
            "glob": {
                "description": "Optional ripgrep glob pattern or list of patterns, for example '*.py' or ['*.py', '!*.md'].",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of match lines to return. Defaults to 100.",
                "minimum": 1,
                "maximum": 10000,
            },
            "context_lines": {
                "type": "integer",
                "description": "Number of context lines before and after each match. Defaults to 0.",
                "minimum": 0,
                "maximum": 20,
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether search is case-sensitive. Defaults to true.",
            },
            "fixed_strings": {
                "type": "boolean",
                "description": "Treat pattern as a literal string instead of regex. Defaults to false.",
            },
            "word_regexp": {
                "type": "boolean",
                "description": "Only show matches surrounded by word boundaries. Defaults to false.",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Search hidden files and directories. Defaults to false.",
            },
            "use_gitignore": {
                "type": "boolean",
                "description": "Respect .gitignore and ignore files. Defaults to true.",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum number of output characters. Defaults to 80000.",
                "minimum": 100,
                "maximum": 500000,
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Search timeout in seconds. Defaults to 20.",
                "minimum": 1,
                "maximum": 120,
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

        pattern = normalize_search_pattern(
            str(args.get("pattern", ""))
        )

        search_path = resolve_base_path_in_workspace(
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

        context_lines = coerce_int(
            args.get("context_lines"),
            name="context_lines",
            default=0,
            minimum=0,
            maximum=20,
        )

        max_chars = coerce_int(
            args.get("max_chars"),
            name="max_chars",
            default=DEFAULT_MAX_CHARS,
            minimum=100,
            maximum=500000,
        )

        timeout_seconds = coerce_int(
            args.get("timeout_seconds"),
            name="timeout_seconds",
            default=DEFAULT_TIMEOUT_SECONDS,
            minimum=1,
            maximum=120,
        )

        case_sensitive = coerce_bool(
            args.get("case_sensitive"),
            default=True,
        )

        fixed_strings = coerce_bool(
            args.get("fixed_strings"),
            default=False,
        )

        word_regexp = coerce_bool(
            args.get("word_regexp"),
            default=False,
        )

        include_hidden = coerce_bool(
            args.get("include_hidden"),
            default=False,
        )

        use_gitignore = coerce_bool(
            args.get("use_gitignore"),
            default=True,
        )

        glob_patterns = coerce_string_list(
            args.get("glob")
        )

        result_data = await self._run_rg(
            workspace_path=workspace_path,
            search_path=search_path,
            pattern=pattern,
            max_results=max_results,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
            word_regexp=word_regexp,
            include_hidden=include_hidden,
            use_gitignore=use_gitignore,
            glob_patterns=glob_patterns,
            max_chars=max_chars,
            timeout_seconds=timeout_seconds,
        )

        return ToolResult.success_result(
            call=call,
            content=result_data["content"],
            data=result_data,
        )

    async def _run_rg(
        self,
        *,
        workspace_path: Path,
        search_path: Path,
        pattern: str,
        max_results: int,
        context_lines: int,
        case_sensitive: bool,
        fixed_strings: bool,
        word_regexp: bool,
        include_hidden: bool,
        use_gitignore: bool,
        glob_patterns: list[str],
        max_chars: int,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        rg_path = require_ripgrep()

        command = build_rg_command(
            rg_path=rg_path,
            pattern=pattern,
            search_path=search_path,
            max_results=max_results,
            context_lines=context_lines,
            case_sensitive=case_sensitive,
            fixed_strings=fixed_strings,
            word_regexp=word_regexp,
            include_hidden=include_hidden,
            use_gitignore=use_gitignore,
            glob_patterns=glob_patterns,
        )

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(workspace_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError as exc:
                process.kill()
                await process.communicate()
                raise ToolExecutionError(
                    f"ripgrep timed out after {timeout_seconds} seconds"
                ) from exc

        except FileNotFoundError as exc:
            raise ToolExecutionError("ripgrep executable not found") from exc
        except OSError as exc:
            raise ToolExecutionError(f"failed to run ripgrep: {exc}") from exc

        stderr_text = stderr.decode("utf-8", errors="replace").strip()

        if process.returncode not in {0, 1}:
            raise ToolExecutionError(
                f"ripgrep failed with exit code {process.returncode}: {stderr_text}"
            )

        stdout_text = stdout.decode("utf-8", errors="replace")

        matches: list[dict[str, Any]] = []
        match_count = 0
        truncated_by_matches = False

        for line in stdout_text.splitlines():
            item = parse_rg_json_line(
                line,
                workspace_path=workspace_path,
            )

            if item is None:
                continue

            if item["type"] == "match":
                match_count += 1

                if match_count > max_results:
                    truncated_by_matches = True
                    break

            matches.append(item)

        search_relative = make_relative_path(
            search_path,
            workspace_path=workspace_path,
        )

        content = format_grep_matches(
            pattern=pattern,
            search_path=search_relative,
            matches=matches,
            max_results=max_results,
            truncated_by_matches=truncated_by_matches,
        )

        content, truncated_by_chars = limit_content_chars(
            content,
            max_chars=max_chars,
        )

        return {
            "pattern": pattern,
            "path": search_relative,
            "absolute_path": str(search_path),
            "workspace_path": str(workspace_path),
            "matches": matches,
            "match_count": len(
                [
                    item
                    for item in matches
                    if item["type"] == "match"
                ]
            ),
            "event_count": len(matches),
            "max_results": max_results,
            "context_lines": context_lines,
            "case_sensitive": case_sensitive,
            "fixed_strings": fixed_strings,
            "word_regexp": word_regexp,
            "include_hidden": include_hidden,
            "use_gitignore": use_gitignore,
            "glob": glob_patterns,
            "timeout_seconds": timeout_seconds,
            "returncode": process.returncode,
            "stderr": stderr_text,
            "truncated_by_matches": truncated_by_matches,
            "truncated_by_chars": truncated_by_chars,
            "command_preview": self._safe_command_preview(command),
            "content": content,
        }

    def _safe_command_preview(self, command: list[str]) -> list[str]:
        return [
            str(item)
            for item in command
        ]

    def render_result(self, result: ToolResult) -> str:
        if not result.success:
            return result.error or "grep failed"

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
    tool = GrepTool()
    context = create_demo_context()

    call = tool.create_call(
        {
            "pattern": "class .*Tool",
            "path": "src/pywork/tools",
            "glob": "*.py",
            "max_results": 40,
            "context_lines": 1,
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