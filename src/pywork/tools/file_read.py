from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.tool import (
    BaseTool,
    ToolExecutionContext,
    ToolExecutionError,
    ToolValidationError,
)


DEFAULT_MAX_LINES = 200
DEFAULT_MAX_CHARS = 60_000
BINARY_CHECK_BYTES = 4096


def get_context_workspace_path(context: ToolExecutionContext) -> Path:
    """
    从 ToolExecutionContext 中取 workspace 路径。

    为了兼容前面不同阶段的 Context 字段，这里做宽松处理：
    - 优先 workspace_path
    - 其次 project_root
    - 最后 Path.cwd()
    """
    workspace = getattr(context, "workspace_path", None)

    if workspace is None:
        workspace = getattr(context, "project_root", None)

    if workspace is None:
        workspace = Path.cwd()

    return Path(workspace).expanduser().resolve()


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


def resolve_file_path_in_workspace(
    path_value: str,
    *,
    workspace_path: Path,
) -> Path:
    """
    解析文件路径，并确保它没有逃出 workspace。
    """
    raw_path = path_value.strip()

    if not raw_path:
        raise ToolValidationError("path cannot be empty")

    candidate = Path(raw_path).expanduser()

    if not candidate.is_absolute():
        candidate = workspace_path / candidate

    resolved = candidate.resolve()

    try:
        resolved.relative_to(workspace_path)
    except ValueError as exc:
        raise ToolValidationError(
            f"path is outside workspace: {raw_path}"
        ) from exc

    return resolved


def is_probably_binary_file(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:BINARY_CHECK_BYTES]
    except OSError as exc:
        raise ToolExecutionError(f"failed to read file header: {exc}") from exc

    return b"\x00" in chunk


def make_relative_path(
    path: Path,
    *,
    workspace_path: Path,
) -> str:
    try:
        return path.relative_to(workspace_path).as_posix()
    except ValueError:
        return str(path)


def format_lines_with_numbers(
    lines: list[tuple[int, str]],
    *,
    include_line_numbers: bool,
) -> str:
    if not lines:
        return ""

    if not include_line_numbers:
        return "\n".join(line for _, line in lines)

    max_line_no = lines[-1][0]
    width = max(1, len(str(max_line_no)))

    return "\n".join(
        f"{line_no:>{width}} | {line}"
        for line_no, line in lines
    )


def limit_content_chars(
    content: str,
    *,
    max_chars: int,
) -> tuple[str, bool]:
    if len(content) <= max_chars:
        return content, False

    suffix = "\n\n... content truncated by max_chars ..."
    allowed = max(0, max_chars - len(suffix))

    return content[:allowed] + suffix, True


class FileReadTool(BaseTool):
    """
    读取 workspace 内文件，并返回带行号的内容。
    """

    name: ClassVar[str] = "file_read"
    description: ClassVar[str] = (
        "Read a text file inside the current workspace and return content with line numbers."
    )
    risk_level: ClassVar[ToolRiskLevel] = ToolRiskLevel.SAFE

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace, or an absolute path inside the workspace.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-based start line. Defaults to 1.",
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-based end line. If omitted, max_lines is used.",
                "minimum": 1,
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to return. Defaults to 200.",
                "minimum": 1,
                "maximum": 5000,
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum number of characters to return. Defaults to 60000.",
                "minimum": 100,
                "maximum": 500000,
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
            },
            "include_line_numbers": {
                "type": "boolean",
                "description": "Whether to include line numbers. Defaults to true.",
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = call.arguments

        path_value = str(args.get("path", "")).strip()
        workspace_path = get_context_workspace_path(context)

        file_path = resolve_file_path_in_workspace(
            path_value,
            workspace_path=workspace_path,
        )

        start_line = coerce_int(
            args.get("start_line"),
            name="start_line",
            default=1,
            minimum=1,
        )

        max_lines = coerce_int(
            args.get("max_lines"),
            name="max_lines",
            default=DEFAULT_MAX_LINES,
            minimum=1,
            maximum=5000,
        )

        max_chars = coerce_int(
            args.get("max_chars"),
            name="max_chars",
            default=DEFAULT_MAX_CHARS,
            minimum=100,
            maximum=500000,
        )

        end_line_raw = args.get("end_line")

        end_line = None
        if end_line_raw is not None:
            end_line = coerce_int(
                end_line_raw,
                name="end_line",
                default=start_line,
                minimum=1,
            )

            if end_line < start_line:
                raise ToolValidationError("end_line must be >= start_line")

        encoding = str(args.get("encoding") or "utf-8").strip() or "utf-8"

        include_line_numbers = coerce_bool(
            args.get("include_line_numbers"),
            default=True,
        )

        result_data = await asyncio.to_thread(
            self._read_file,
            file_path,
            workspace_path=workspace_path,
            start_line=start_line,
            end_line=end_line,
            max_lines=max_lines,
            max_chars=max_chars,
            encoding=encoding,
            include_line_numbers=include_line_numbers,
        )

        return ToolResult.success_result(
            call=call,
            content=result_data["content"],
            data=result_data,
        )

    def _read_file(
        self,
        file_path: Path,
        *,
        workspace_path: Path,
        start_line: int,
        end_line: int | None,
        max_lines: int,
        max_chars: int,
        encoding: str,
        include_line_numbers: bool,
    ) -> dict[str, Any]:
        if not file_path.exists():
            raise ToolValidationError(f"file does not exist: {file_path}")

        if not file_path.is_file():
            raise ToolValidationError(f"path is not a file: {file_path}")

        if is_probably_binary_file(file_path):
            raise ToolValidationError(f"file appears to be binary: {file_path}")

        file_size = file_path.stat().st_size
        relative_path = make_relative_path(
            file_path,
            workspace_path=workspace_path,
        )

        requested_end_line = end_line

        if requested_end_line is None:
            read_until_line = start_line + max_lines - 1
        else:
            read_until_line = min(
                requested_end_line,
                start_line + max_lines - 1,
            )

        selected_lines: list[tuple[int, str]] = []
        has_more_after = False

        try:
            with file_path.open(
                "r",
                encoding=encoding,
                errors="replace",
                newline=None,
            ) as file:
                for line_no, line in enumerate(file, start=1):
                    if line_no < start_line:
                        continue

                    if line_no > read_until_line:
                        has_more_after = True
                        break

                    selected_lines.append(
                        (
                            line_no,
                            line.rstrip("\n").rstrip("\r"),
                        )
                    )

        except UnicodeError as exc:
            raise ToolExecutionError(
                f"failed to decode file with encoding {encoding!r}: {exc}"
            ) from exc
        except OSError as exc:
            raise ToolExecutionError(f"failed to read file: {exc}") from exc

        body = format_lines_with_numbers(
            selected_lines,
            include_line_numbers=include_line_numbers,
        )

        actual_start_line = selected_lines[0][0] if selected_lines else start_line
        actual_end_line = selected_lines[-1][0] if selected_lines else start_line - 1

        header = (
            f"# file: {relative_path}\n"
            f"# lines: {actual_start_line}-{actual_end_line}\n"
        )

        content = header + body

        content, truncated_by_chars = limit_content_chars(
            content,
            max_chars=max_chars,
        )

        truncated_by_max_lines = False

        if requested_end_line is not None and requested_end_line > read_until_line:
            truncated_by_max_lines = True

        if requested_end_line is None and has_more_after:
            truncated_by_max_lines = True

        return {
            "path": relative_path,
            "absolute_path": str(file_path),
            "workspace_path": str(workspace_path),
            "content": content,
            "start_line": actual_start_line,
            "end_line": actual_end_line,
            "line_count": len(selected_lines),
            "requested_start_line": start_line,
            "requested_end_line": requested_end_line,
            "max_lines": max_lines,
            "max_chars": max_chars,
            "file_size": file_size,
            "encoding": encoding,
            "include_line_numbers": include_line_numbers,
            "truncated_by_max_lines": truncated_by_max_lines,
            "truncated_by_chars": truncated_by_chars,
        }

    def render_result(self, result: ToolResult) -> str:
        if not result.success:
            return result.error or "file_read failed"

        return result.content


def create_demo_context() -> ToolExecutionContext:
    """
    兼容 ToolExecutionContext 字段变化的 demo context 创建函数。
    """
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
    tool = FileReadTool()
    context = create_demo_context()

    call = tool.create_call(
        {
            "path": "src/pywork/tools/file_read.py",
            "start_line": 1,
            "max_lines": 40,
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