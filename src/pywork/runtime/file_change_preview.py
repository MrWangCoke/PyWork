from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pywork.tools.file_change import FileChangePreview
from pywork.tools.file_edit import FileEditTool
from pywork.tools.file_write import FileWriteTool
from pywork.tools.tool import ToolExecutionContext


FILE_CHANGE_TOOL_NAMES: set[str] = {
    "file_write",
    "file_edit",
}


@dataclass(slots=True, frozen=True)
class PreviewToolCall:
    tool_name: str
    arguments: dict[str, Any]
    call_id: str | None = None


def normalize_tool_name(tool_name: str) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def get_decision_request(decision: Any) -> Any:
    request = getattr(decision, "request", None)

    if request is None:
        raise ValueError("permission decision has no request")

    return request


def get_request_arguments(request: Any) -> dict[str, Any]:
    arguments = getattr(request, "arguments", {})

    if isinstance(arguments, dict):
        return dict(arguments)

    try:
        return dict(arguments)
    except Exception:
        return {}


def is_file_change_tool_name(tool_name: str) -> bool:
    return normalize_tool_name(tool_name) in FILE_CHANGE_TOOL_NAMES


def build_preview_context(
    *,
    workspace_path: str | Path,
    permission_mode: str | None = None,
) -> ToolExecutionContext:
    workspace = Path(workspace_path).expanduser().resolve()

    return ToolExecutionContext(
        workspace_path=str(workspace),
        project_root=str(workspace),
        permission_mode=permission_mode or "default",
    )


def build_file_change_preview_for_decision(
    decision: Any,
    *,
    workspace_path: str | Path,
) -> FileChangePreview | None:
    """
    根据 PermissionDecision 生成文件修改 diff preview。

    注意：
    - 只支持 file_write / file_edit
    - 只 preview，不写文件
    - 后面 TUI ApprovalDialog 可以把 preview.diff_text 展示出来
    """
    request = get_decision_request(decision)

    tool_name = normalize_tool_name(
        getattr(request, "tool_name", "")
    )

    if not is_file_change_tool_name(tool_name):
        return None

    arguments = get_request_arguments(request)

    call = PreviewToolCall(
        tool_name=tool_name,
        arguments=arguments,
        call_id=getattr(request, "call_id", None),
    )

    context = build_preview_context(
        workspace_path=workspace_path,
        permission_mode=str(getattr(decision, "mode", "default")),
    )

    if tool_name == "file_write":
        return FileWriteTool().preview(
            call,  # type: ignore[arg-type]
            context,
        )

    if tool_name == "file_edit":
        return FileEditTool().preview(
            call,  # type: ignore[arg-type]
            context,
        )

    return None


def build_file_change_preview_for_gate_result(
    gate_result: Any,
    *,
    workspace_path: str | Path,
) -> FileChangePreview | None:
    decision = getattr(gate_result, "decision", None)

    if decision is None:
        return None

    return build_file_change_preview_for_decision(
        decision,
        workspace_path=workspace_path,
    )