from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Static

from pywork.runtime.events import RuntimeEvent, RuntimeEventType
from pywork.schemas.tool_schema import ToolCall, ToolResult, create_tool_call


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compact_json(
    value: Any,
    *,
    indent: int = 2,
    max_chars: int = 4000,
) -> str:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            indent=indent,
            default=str,
        )
    except TypeError:
        text = str(value)

    if len(text) <= max_chars:
        return text

    suffix = "\n... truncated ..."
    return text[: max(0, max_chars - len(suffix))] + suffix


def get_attr_or_key(value: Any, name: str, default: Any = None) -> Any:
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def format_time(value: datetime) -> str:
    return value.astimezone().strftime("%H:%M:%S")


class ToolLogEntryKind(str, Enum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    CHECKPOINT = "checkpoint"
    STATUS = "status"
    MESSAGE = "message"


@dataclass(frozen=True)
class ToolLogEntry:
    kind: ToolLogEntryKind
    title: str
    body: str = ""

    tool_name: str | None = None
    call_id: str | None = None
    result_id: str | None = None
    success: bool | None = None

    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def icon(self) -> str:
        if self.kind == ToolLogEntryKind.TOOL_CALL:
            return "▶"

        if self.kind == ToolLogEntryKind.TOOL_RESULT:
            return "✓" if self.success else "✗"

        if self.kind == ToolLogEntryKind.ERROR:
            return "!"

        if self.kind == ToolLogEntryKind.CHECKPOINT:
            return "◆"

        if self.kind == ToolLogEntryKind.STATUS:
            return "●"

        return "-"

    def to_text(self) -> str:
        lines = [
            f"{self.icon()} [{format_time(self.created_at)}] {self.title}",
        ]

        if self.tool_name:
            lines.append(f"  tool: {self.tool_name}")

        if self.call_id:
            lines.append(f"  call_id: {self.call_id}")

        if self.result_id:
            lines.append(f"  result_id: {self.result_id}")

        if self.body:
            lines.append("")
            lines.extend(
                f"  {line}"
                for line in self.body.splitlines()
            )

        return "\n".join(lines)


class ToolLog(VerticalScroll):
    """
    TUI 工具日志组件。

    用于实时展示：
    - 工具调用 tool_call
    - 工具结果 tool_result
    - 错误 error
    - 检查点 checkpoint
    - 状态 status

    后面 Runtime Streaming 接入时，可以这样用：

        tool_log.append_runtime_event(event)
    """

    DEFAULT_CSS = """
    ToolLog {
        height: 1fr;
        border: solid $panel;
        padding: 0 1;
        background: $surface;
    }

    ToolLog:focus {
        border: solid $accent;
    }

    #tool-log-content {
        width: 100%;
        height: auto;
        padding: 0 0;
    }
    """

    def __init__(
        self,
        *,
        max_entries: int = 200,
        show_empty_hint: bool = True,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(
            id=id,
            classes=classes,
        )

        self.max_entries = max_entries
        self.show_empty_hint = show_empty_hint
        self.entries: list[ToolLogEntry] = []

    def compose(self) -> ComposeResult:
        yield Static(
            self.render_log(),
            id="tool-log-content",
        )

    def on_mount(self) -> None:
        self.refresh_log()

    def clear(self) -> None:
        self.entries.clear()
        self.refresh_log()

    def get_entries(self) -> list[ToolLogEntry]:
        return list(self.entries)

    def append_entry(self, entry: ToolLogEntry) -> ToolLogEntry:
        self.entries.append(entry)

        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

        self.refresh_log()
        return entry

    def append_tool_call(
        self,
        tool_call: ToolCall | dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        tool_name = str(
            get_attr_or_key(
                tool_call,
                "tool_name",
                "unknown_tool",
            )
        )

        call_id = str(
            get_attr_or_key(
                tool_call,
                "call_id",
                "",
            )
            or ""
        )

        arguments = get_attr_or_key(
            tool_call,
            "arguments",
            {},
        )

        body = "arguments:\n" + compact_json(arguments)

        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.TOOL_CALL,
                title=f"tool_call: {tool_name}",
                body=body,
                tool_name=tool_name,
                call_id=call_id,
                metadata=metadata or {},
            )
        )

    def append_tool_result(
        self,
        tool_result: ToolResult | dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        tool_name = str(
            get_attr_or_key(
                tool_result,
                "tool_name",
                "unknown_tool",
            )
        )

        call_id = str(
            get_attr_or_key(
                tool_result,
                "call_id",
                "",
            )
            or ""
        )

        result_id = str(
            get_attr_or_key(
                tool_result,
                "result_id",
                "",
            )
            or ""
        )

        success = bool(
            get_attr_or_key(
                tool_result,
                "success",
                False,
            )
        )

        content = str(
            get_attr_or_key(
                tool_result,
                "content",
                "",
            )
            or ""
        )

        error = get_attr_or_key(
            tool_result,
            "error",
            None,
        )

        data = get_attr_or_key(
            tool_result,
            "data",
            {},
        )

        body_parts: list[str] = []

        if content:
            body_parts.append("content:")
            body_parts.append(content)

        if error:
            body_parts.append("error:")
            body_parts.append(str(error))

        if data:
            body_parts.append("data:")
            body_parts.append(compact_json(data))

        body = "\n".join(body_parts)

        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.TOOL_RESULT,
                title=f"tool_result: {tool_name} {'success' if success else 'failed'}",
                body=body,
                tool_name=tool_name,
                call_id=call_id,
                result_id=result_id,
                success=success,
                metadata=metadata or {},
            )
        )

    def append_error(
        self,
        error: str,
        *,
        error_type: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        title = "error"

        if error_type:
            title = f"error: {error_type}"

        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.ERROR,
                title=title,
                body=error,
                success=False,
                metadata=metadata or {},
            )
        )

    def append_checkpoint(
        self,
        checkpoint_id: str,
        *,
        data: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        body = ""

        if data:
            body = compact_json(data)

        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.CHECKPOINT,
                title=f"checkpoint: {checkpoint_id}",
                body=body,
                metadata=metadata or {},
            )
        )

    def append_status(
        self,
        status: str,
        *,
        content: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.STATUS,
                title=f"status: {status}",
                body=content,
                metadata=metadata or {},
            )
        )

    def append_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolLogEntry:
        return self.append_entry(
            ToolLogEntry(
                kind=ToolLogEntryKind.MESSAGE,
                title="message",
                body=content,
                metadata=metadata or {},
            )
        )

    def append_runtime_event(
        self,
        event: RuntimeEvent,
    ) -> ToolLogEntry | None:
        """
        接收 RuntimeEvent，并自动转换成 ToolLogEntry。

        后面 streaming.py 接入 TUI 时会用这个方法。
        """
        if event.event_type == RuntimeEventType.TOOL_CALL:
            if event.tool_call is None:
                return None

            return self.append_tool_call(
                event.tool_call,
                metadata={
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "source": event.source.value,
                },
            )

        if event.event_type == RuntimeEventType.TOOL_RESULT:
            if event.tool_result is None:
                return None

            return self.append_tool_result(
                event.tool_result,
                metadata={
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "source": event.source.value,
                },
            )

        if event.event_type == RuntimeEventType.ERROR:
            return self.append_error(
                event.error or event.content,
                error_type=event.error_type,
                metadata={
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "source": event.source.value,
                },
            )

        if event.event_type == RuntimeEventType.CHECKPOINT:
            return self.append_checkpoint(
                event.checkpoint_id or "unknown_checkpoint",
                data=event.data,
                metadata={
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "source": event.source.value,
                },
            )

        if event.event_type == RuntimeEventType.STATUS:
            return self.append_status(
                event.status or "unknown",
                content=event.content,
                metadata={
                    "event_id": event.event_id,
                    "run_id": event.run_id,
                    "source": event.source.value,
                },
            )

        return None

    def render_log(self) -> str:
        if not self.entries:
            if not self.show_empty_hint:
                return ""

            return (
                "Tool Log\n"
                "No tool events yet.\n\n"
                "等待 Runtime 发送 tool_call / tool_result / error / checkpoint 事件。"
            )

        parts = ["Tool Log", ""]

        for index, entry in enumerate(self.entries, start=1):
            parts.append(f"#{index}")
            parts.append(entry.to_text())

            if index != len(self.entries):
                parts.append("-" * 60)

        return "\n".join(parts)

    def refresh_log(self) -> None:
        if not self.is_mounted:
            return

        content = self.query_one("#tool-log-content", Static)
        content.update(self.render_log())

        self.call_after_refresh(self._scroll_to_bottom)

    def _scroll_to_bottom(self) -> None:
        try:
            self.scroll_end(animate=False)
        except Exception:
            pass


class ToolLogDemoApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    ToolLog {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("c", "clear_log", "Clear"),
        ("r", "reload_demo", "Reload Demo"),
    ]

    def compose(self) -> ComposeResult:
        yield ToolLog(id="tool-log")

    def on_mount(self) -> None:
        self.action_reload_demo()

    def action_clear_log(self) -> None:
        tool_log = self.query_one("#tool-log", ToolLog)
        tool_log.clear()

    def action_reload_demo(self) -> None:
        tool_log = self.query_one("#tool-log", ToolLog)
        tool_log.clear()

        tool_log.append_status(
            "thinking",
            content="Runtime is preparing a tool call.",
        )

        call = create_tool_call(
            tool_name="grep",
            arguments={
                "pattern": "class .*Tool",
                "path": "src/pywork/tools",
                "glob": "*.py",
                "max_results": 20,
            },
        )

        tool_log.append_tool_call(call)

        result = ToolResult.success_result(
            call=call,
            content=(
                "src/pywork/tools/file_read.py:174: class FileReadTool(BaseTool)\n"
                "src/pywork/tools/glob.py:250: class GlobTool(BaseTool)\n"
                "src/pywork/tools/grep.py:230: class GrepTool(BaseTool)"
            ),
            data={
                "match_count": 3,
                "path": "src/pywork/tools",
            },
        )

        tool_log.append_tool_result(result)

        tool_log.append_checkpoint(
            "checkpoint_demo_001",
            data={
                "iteration": 1,
                "message": "tool result appended",
            },
        )

        tool_log.append_error(
            "demo error message",
            error_type="DemoError",
        )

        tool_log.append_status(
            "idle",
            content="Runtime finished.",
        )


def main() -> int:
    app = ToolLogDemoApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())