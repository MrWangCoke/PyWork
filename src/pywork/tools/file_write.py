from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.file_change import (
    DEFAULT_MAX_DIFF_CHARS,
    DEFAULT_MAX_DIFF_LINES,
    FileChangeApplyResult,
    FileChangePreview,
    apply_file_change_preview,
    build_file_change_preview,
    read_text_file_for_change,
)
from pywork.tools.file_read import (
    coerce_bool,
    coerce_int,
    get_context_workspace_path,
    make_relative_path,
    resolve_file_path_in_workspace,
)
from pywork.tools.tool import BaseTool, ToolExecutionContext, ToolValidationError


class FileWriteTool(BaseTool):
    """
    写入 workspace 内文件。

    现在支持两阶段：
    - preview(): 只生成 diff，不写文件
    - apply(): 真正写入
    - execute(): 兼容旧流程，内部 preview + apply
    """

    name: ClassVar[str] = "file_write"

    description: ClassVar[str] = (
        "Write text content to a file inside the current workspace. "
        "Supports preview/apply flow with unified diff."
    )

    risk_level: ClassVar[ToolRiskLevel] = ToolRiskLevel.HIGH

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path relative to the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Full text content to write.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Whether to overwrite an existing file. Defaults to false.",
            },
            "create_dirs": {
                "type": "boolean",
                "description": "Whether to create parent directories. Defaults to true.",
            },
            "preview_only": {
                "type": "boolean",
                "description": "If true, only return diff preview and do not write the file.",
            },
            "max_diff_lines": {
                "type": "integer",
                "description": "Maximum number of diff lines to return.",
            },
            "max_diff_chars": {
                "type": "integer",
                "description": "Maximum number of diff characters to return.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def preview(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> FileChangePreview:
        args = call.arguments

        path_value = str(args.get("path", "")).strip()

        if not path_value:
            raise ToolValidationError("path cannot be empty")

        content = args.get("content")

        if not isinstance(content, str):
            raise ToolValidationError("content must be a string")

        encoding = str(args.get("encoding") or "utf-8").strip() or "utf-8"

        overwrite = coerce_bool(
            args.get("overwrite"),
            default=False,
        )

        create_dirs = coerce_bool(
            args.get("create_dirs"),
            default=True,
        )

        max_diff_lines = coerce_int(
            args.get("max_diff_lines"),
            name="max_diff_lines",
            default=DEFAULT_MAX_DIFF_LINES,
            minimum=20,
            maximum=20_000,
        )

        max_diff_chars = coerce_int(
            args.get("max_diff_chars"),
            name="max_diff_chars",
            default=DEFAULT_MAX_DIFF_CHARS,
            minimum=500,
            maximum=1_000_000,
        )

        workspace_path = get_context_workspace_path(context)

        file_path = resolve_file_path_in_workspace(
            path_value,
            workspace_path=workspace_path,
        )

        old_exists = file_path.exists()

        if old_exists and not file_path.is_file():
            raise ToolValidationError(f"path exists but is not a file: {file_path}")

        if old_exists and not overwrite:
            raise ToolValidationError(
                "file already exists; pass overwrite=true to replace it"
            )

        if not file_path.parent.exists() and not create_dirs:
            raise ToolValidationError(
                f"parent directory does not exist: {file_path.parent}"
            )

        old_content = read_text_file_for_change(
            file_path,
            encoding=encoding,
        )

        relative_path = make_relative_path(
            file_path,
            workspace_path=workspace_path,
        )

        return build_file_change_preview(
            operation="write",
            path=relative_path,
            file_path=file_path,
            workspace_path=workspace_path,
            old_content=old_content,
            new_content=content,
            old_exists=old_exists,
            encoding=encoding,
            create_dirs=create_dirs,
            overwrite=overwrite,
            max_diff_lines=max_diff_lines,
            max_diff_chars=max_diff_chars,
            metadata={
                "tool_name": self.name,
                "call_id": call.call_id,
            },
        )

    def apply(
        self,
        preview: FileChangePreview,
    ) -> FileChangeApplyResult:
        if preview.operation != "write":
            raise ToolValidationError(
                f"invalid preview operation for file_write: {preview.operation}"
            )

        return apply_file_change_preview(preview)

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        preview = await asyncio.to_thread(
            self.preview,
            call,
            context,
        )

        preview_only = coerce_bool(
            call.arguments.get("preview_only"),
            default=False,
        )

        if preview_only:
            return ToolResult.success_result(
                call=call,
                content=preview.diff_text or preview.diff_summary,
                data={
                    "preview_only": True,
                    "preview": preview.to_dict(),
                },
                metadata={
                    "preview_only": True,
                    "has_changes": preview.has_changes,
                },
            )

        apply_result = await asyncio.to_thread(
            self.apply,
            preview,
        )

        content = apply_result.diff_text or apply_result.diff_summary

        return ToolResult.success_result(
            call=call,
            content=content,
            data={
                "preview": preview.to_dict(),
                "apply_result": apply_result.to_dict(),
                "path": apply_result.path,
                "absolute_path": apply_result.absolute_path,
                "changed": apply_result.changed,
                "applied": apply_result.applied,
                "diff_text": apply_result.diff_text,
                "diff_summary": apply_result.diff_summary,
            },
            metadata={
                "has_changes": preview.has_changes,
                "operation": "write",
            },
        )

    def render_result(
        self,
        result: ToolResult,
    ) -> str:
        if not result.success:
            return result.error or result.content or "file_write failed"

        return result.content or "file written"