from __future__ import annotations

import asyncio
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


class FileEditTool(BaseTool):
    """
    精确替换 workspace 内文本文件内容。

    现在支持两阶段：
    - preview(): 只生成 diff，不写文件
    - apply(): 真正写入
    - execute(): 兼容旧流程，内部 preview + apply
    """

    name: ClassVar[str] = "file_edit"

    description: ClassVar[str] = (
        "Edit a text file inside the current workspace by replacing an exact string. "
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
            "old_string": {
                "type": "string",
                "description": "Exact text to replace.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text.",
            },
            "encoding": {
                "type": "string",
                "description": "Text encoding. Defaults to utf-8.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "Replace all occurrences. Defaults to false.",
            },
            "occurrence": {
                "type": "integer",
                "description": "1-based occurrence to replace. Cannot be used with replace_all.",
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
        "required": ["path", "old_string", "new_string"],
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

        old_string = args.get("old_string")
        new_string = args.get("new_string")

        if not isinstance(old_string, str):
            raise ToolValidationError("old_string must be a string")

        if not isinstance(new_string, str):
            raise ToolValidationError("new_string must be a string")

        if old_string == "":
            raise ToolValidationError("old_string cannot be empty")

        encoding = str(args.get("encoding") or "utf-8").strip() or "utf-8"

        replace_all = coerce_bool(
            args.get("replace_all"),
            default=False,
        )

        occurrence_raw = args.get("occurrence")

        occurrence: int | None = None

        if occurrence_raw is not None:
            occurrence = coerce_int(
                occurrence_raw,
                name="occurrence",
                default=1,
                minimum=1,
            )

        if replace_all and occurrence is not None:
            raise ToolValidationError("occurrence cannot be used with replace_all=true")

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

        if not file_path.exists():
            raise ToolValidationError(f"file does not exist: {file_path}")

        if not file_path.is_file():
            raise ToolValidationError(f"path is not a file: {file_path}")

        old_content = read_text_file_for_change(
            file_path,
            encoding=encoding,
        )

        new_content, replacement_count = replace_text_exactly(
            old_content,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            occurrence=occurrence,
        )

        relative_path = make_relative_path(
            file_path,
            workspace_path=workspace_path,
        )

        return build_file_change_preview(
            operation="edit",
            path=relative_path,
            file_path=file_path,
            workspace_path=workspace_path,
            old_content=old_content,
            new_content=new_content,
            old_exists=True,
            encoding=encoding,
            create_dirs=False,
            overwrite=True,
            max_diff_lines=max_diff_lines,
            max_diff_chars=max_diff_chars,
            metadata={
                "tool_name": self.name,
                "call_id": call.call_id,
                "replacement_count": replacement_count,
                "replace_all": replace_all,
                "occurrence": occurrence,
            },
        )

    def apply(
        self,
        preview: FileChangePreview,
    ) -> FileChangeApplyResult:
        if preview.operation != "edit":
            raise ToolValidationError(
                f"invalid preview operation for file_edit: {preview.operation}"
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
                "operation": "edit",
            },
        )

    def render_result(
        self,
        result: ToolResult,
    ) -> str:
        if not result.success:
            return result.error or result.content or "file_edit failed"

        return result.content or "file edited"


def replace_text_exactly(
    content: str,
    *,
    old_string: str,
    new_string: str,
    replace_all: bool,
    occurrence: int | None,
) -> tuple[str, int]:
    count = content.count(old_string)

    if count == 0:
        raise ToolValidationError("old_string was not found in file")

    if replace_all:
        return content.replace(old_string, new_string), count

    if occurrence is not None:
        if occurrence > count:
            raise ToolValidationError(
                f"occurrence {occurrence} was requested, but only {count} occurrence(s) exist"
            )

        return replace_nth_occurrence(
            content,
            old_string=old_string,
            new_string=new_string,
            occurrence=occurrence,
        ), 1

    if count != 1:
        raise ToolValidationError(
            f"old_string appears {count} times; pass occurrence or replace_all=true"
        )

    return content.replace(old_string, new_string, 1), 1


def replace_nth_occurrence(
    content: str,
    *,
    old_string: str,
    new_string: str,
    occurrence: int,
) -> str:
    start = -1
    search_from = 0

    for _ in range(occurrence):
        start = content.find(old_string, search_from)

        if start < 0:
            raise ToolValidationError("old_string occurrence was not found")

        search_from = start + len(old_string)

    end = start + len(old_string)

    return content[:start] + new_string + content[end:]