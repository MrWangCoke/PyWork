from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


PermissionMode = Literal[
    "default",
    "accept_edits",
    "bypass_permissions",
    "plan",
]

NotificationLevel = Literal[
    "info",
    "success",
    "warning",
    "error",
]


class UIRuntimeState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    RUNNING_TOOL = "running_tool"
    ERROR = "error"


class UIFocusTarget(str, Enum):
    CHAT = "chat"
    INPUT = "input"
    STATUS_BAR = "status_bar"
    COMMAND_PALETTE = "command_palette"
    MODAL = "modal"


class UIModalKind(str, Enum):
    NONE = "none"
    INFO = "info"
    CONFIRM = "confirm"
    ERROR = "error"
    PERMISSION = "permission"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_ui_id() -> str:
    return f"ui_{uuid4().hex}"


def new_notification_id() -> str:
    return f"notification_{uuid4().hex}"


@dataclass
class UIStatusBarState:
    """
    状态栏显示状态。

    这个状态和 tui/components/status_bar.py 对应。
    """

    model: str = "deepseek-v4-flash"
    provider: str = "deepseek"
    permission_mode: str = "default"

    input_tokens: int = 0
    output_tokens: int = 0

    runtime_state: UIRuntimeState = UIRuntimeState.IDLE
    message: str = "ready"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def set_model(
        self,
        model: str,
        *,
        provider: str | None = None,
    ) -> None:
        self.model = model

        if provider is not None:
            self.provider = provider

    def set_permission_mode(self, mode: str) -> None:
        self.permission_mode = mode

    def set_token_usage(
        self,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        if input_tokens is not None:
            self.input_tokens = max(0, input_tokens)

        if output_tokens is not None:
            self.output_tokens = max(0, output_tokens)

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.input_tokens = max(0, self.input_tokens + input_tokens)
        self.output_tokens = max(0, self.output_tokens + output_tokens)

    def reset_token_usage(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    def set_runtime_state(
        self,
        state: UIRuntimeState | str,
        *,
        message: str = "",
    ) -> None:
        self.runtime_state = UIRuntimeState(state)
        self.message = message

    def set_idle(self, message: str = "ready") -> None:
        self.set_runtime_state(UIRuntimeState.IDLE, message=message)

    def set_thinking(self, message: str = "thinking") -> None:
        self.set_runtime_state(UIRuntimeState.THINKING, message=message)

    def set_running_tool(self, message: str = "running tool") -> None:
        self.set_runtime_state(UIRuntimeState.RUNNING_TOOL, message=message)

    def set_error(self, message: str = "error") -> None:
        self.set_runtime_state(UIRuntimeState.ERROR, message=message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "permission_mode": self.permission_mode,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "runtime_state": self.runtime_state.value,
            "message": self.message,
        }


@dataclass
class UICommandItem:
    """
    命令面板中的一个命令。
    """

    command_id: str
    title: str
    description: str = ""
    key_binding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "title": self.title,
            "description": self.description,
            "key_binding": self.key_binding,
            "metadata": self.metadata,
        }


@dataclass
class UICommandPaletteState:
    """
    命令面板状态。

    后面可以用于：
    - /help
    - /clear
    - /model
    - /permissions
    """

    is_open: bool = False
    query: str = ""
    selected_index: int = 0
    items: list[UICommandItem] = field(default_factory=list)
    opened_at: datetime | None = None

    def open(self) -> None:
        self.is_open = True
        self.opened_at = utc_now()

    def close(self) -> None:
        self.is_open = False
        self.query = ""
        self.selected_index = 0
        self.opened_at = None

    def set_query(self, query: str) -> None:
        self.query = query
        self.selected_index = 0

    def set_items(self, items: list[UICommandItem]) -> None:
        self.items = items
        self.selected_index = 0

    def move_selection(self, delta: int) -> None:
        if not self.items:
            self.selected_index = 0
            return

        self.selected_index = (self.selected_index + delta) % len(self.items)

    def get_selected_item(self) -> UICommandItem | None:
        if not self.items:
            return None

        if self.selected_index < 0 or self.selected_index >= len(self.items):
            return None

        return self.items[self.selected_index]

    def get_filtered_items(self) -> list[UICommandItem]:
        keyword = self.query.strip().lower()

        if not keyword:
            return self.items

        return [
            item
            for item in self.items
            if keyword in item.title.lower()
            or keyword in item.description.lower()
            or keyword in item.command_id.lower()
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_open": self.is_open,
            "query": self.query,
            "selected_index": self.selected_index,
            "opened_at": self.opened_at.isoformat() if self.opened_at else None,
            "items": [
                item.to_dict()
                for item in self.items
            ],
            "selected_item": (
                self.get_selected_item().to_dict()
                if self.get_selected_item()
                else None
            ),
        }


@dataclass
class UINotification:
    """
    UI 通知消息。

    比如：
    - 保存成功
    - 工具执行失败
    - token 已重置
    """

    message: str
    level: NotificationLevel = "info"
    notification_id: str = field(default_factory=new_notification_id)
    created_at: datetime = field(default_factory=utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notification_id": self.notification_id,
            "message": self.message,
            "level": self.level,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class UIState:
    """
    PyWork UI 交互状态。

    这个状态只描述 UI，不负责真正执行模型或工具。
    """

    ui_id: str = field(default_factory=new_ui_id)

    focus_target: UIFocusTarget = UIFocusTarget.INPUT

    input_text: str = ""

    is_busy: bool = False
    is_streaming: bool = False

    status_bar: UIStatusBarState = field(default_factory=UIStatusBarState)

    command_palette: UICommandPaletteState = field(
        default_factory=UICommandPaletteState
    )

    modal_kind: UIModalKind = UIModalKind.NONE
    modal_title: str = ""
    modal_message: str = ""

    last_error: str | None = None

    notifications: list[UINotification] = field(default_factory=list)

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.updated_at = utc_now()

    def set_focus(self, target: UIFocusTarget | str) -> None:
        self.focus_target = UIFocusTarget(target)
        self.touch()

    def focus_input(self) -> None:
        self.set_focus(UIFocusTarget.INPUT)

    def focus_chat(self) -> None:
        self.set_focus(UIFocusTarget.CHAT)

    def set_input_text(self, text: str) -> None:
        self.input_text = text
        self.touch()

    def append_input_text(self, text: str) -> None:
        self.input_text += text
        self.touch()

    def clear_input_text(self) -> None:
        self.input_text = ""
        self.touch()

    def set_idle(self, message: str = "ready") -> None:
        self.is_busy = False
        self.is_streaming = False
        self.status_bar.set_idle(message)
        self.touch()

    def set_thinking(self, message: str = "thinking") -> None:
        self.is_busy = True
        self.status_bar.set_thinking(message)
        self.touch()

    def set_running_tool(self, message: str = "running tool") -> None:
        self.is_busy = True
        self.status_bar.set_running_tool(message)
        self.touch()

    def set_error(self, message: str) -> None:
        self.is_busy = False
        self.is_streaming = False
        self.last_error = message
        self.status_bar.set_error(message)
        self.notify(message, level="error")
        self.touch()

    def clear_error(self) -> None:
        self.last_error = None
        self.touch()

    def start_streaming(self, message: str = "streaming") -> None:
        self.is_busy = True
        self.is_streaming = True
        self.status_bar.set_thinking(message)
        self.touch()

    def stop_streaming(self, message: str = "ready") -> None:
        self.is_busy = False
        self.is_streaming = False
        self.status_bar.set_idle(message)
        self.touch()

    def set_model(
        self,
        model: str,
        *,
        provider: str | None = None,
    ) -> None:
        self.status_bar.set_model(
            model,
            provider=provider,
        )
        self.touch()

    def set_permission_mode(self, mode: str) -> None:
        self.status_bar.set_permission_mode(mode)
        self.touch()

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.status_bar.add_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.touch()

    def reset_token_usage(self) -> None:
        self.status_bar.reset_token_usage()
        self.status_bar.set_idle("tokens reset")
        self.notify("Token usage reset.", level="success")
        self.touch()

    def open_command_palette(
        self,
        *,
        query: str = "",
        items: list[UICommandItem] | None = None,
    ) -> None:
        if items is not None:
            self.command_palette.set_items(items)

        self.command_palette.open()
        self.command_palette.set_query(query)
        self.set_focus(UIFocusTarget.COMMAND_PALETTE)
        self.touch()

    def close_command_palette(self) -> None:
        self.command_palette.close()
        self.focus_input()
        self.touch()

    def open_modal(
        self,
        *,
        kind: UIModalKind | str,
        title: str,
        message: str,
    ) -> None:
        self.modal_kind = UIModalKind(kind)
        self.modal_title = title
        self.modal_message = message
        self.set_focus(UIFocusTarget.MODAL)
        self.touch()

    def close_modal(self) -> None:
        self.modal_kind = UIModalKind.NONE
        self.modal_title = ""
        self.modal_message = ""
        self.focus_input()
        self.touch()

    def notify(
        self,
        message: str,
        *,
        level: NotificationLevel = "info",
        metadata: dict[str, Any] | None = None,
    ) -> UINotification:
        notification = UINotification(
            message=message,
            level=level,
            metadata=metadata or {},
        )

        self.notifications.append(notification)
        self.touch()

        return notification

    def pop_notification(self) -> UINotification | None:
        if not self.notifications:
            return None

        notification = self.notifications.pop(0)
        self.touch()
        return notification

    def clear_notifications(self) -> None:
        self.notifications.clear()
        self.touch()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ui_id": self.ui_id,
            "focus_target": self.focus_target.value,
            "input_text": self.input_text,
            "is_busy": self.is_busy,
            "is_streaming": self.is_streaming,
            "status_bar": self.status_bar.to_dict(),
            "command_palette": self.command_palette.to_dict(),
            "modal": {
                "kind": self.modal_kind.value,
                "title": self.modal_title,
                "message": self.modal_message,
            },
            "last_error": self.last_error,
            "notifications": [
                notification.to_dict()
                for notification in self.notifications
            ],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
            default=str,
        )

    def summary(self) -> dict[str, Any]:
        return {
            "ui_id": self.ui_id,
            "focus_target": self.focus_target.value,
            "is_busy": self.is_busy,
            "is_streaming": self.is_streaming,
            "runtime_state": self.status_bar.runtime_state.value,
            "status_message": self.status_bar.message,
            "model": self.status_bar.model,
            "provider": self.status_bar.provider,
            "permission_mode": self.status_bar.permission_mode,
            "total_tokens": self.status_bar.total_tokens,
            "command_palette_open": self.command_palette.is_open,
            "modal_kind": self.modal_kind.value,
            "notification_count": len(self.notifications),
            "last_error": self.last_error,
            "updated_at": self.updated_at.isoformat(),
        }


def create_ui_state(
    *,
    model: str = "deepseek-v4-flash",
    provider: str = "deepseek",
    permission_mode: str = "default",
    metadata: dict[str, Any] | None = None,
) -> UIState:
    state = UIState(metadata=metadata or {})

    state.set_model(
        model,
        provider=provider,
    )
    state.set_permission_mode(permission_mode)
    state.set_idle("ready")

    return state


def main() -> int:
    state = create_ui_state(
        model="deepseek-v4-flash",
        provider="deepseek",
        permission_mode="default",
    )

    state.set_input_text("hello")
    state.set_thinking("waiting for model response")
    state.add_token_usage(
        input_tokens=2,
        output_tokens=5,
    )
    state.notify("Demo notification.", level="info")

    state.open_command_palette(
        items=[
            UICommandItem(
                command_id="clear_chat",
                title="Clear Chat",
                description="Clear all chat messages.",
                key_binding="Ctrl+L",
            ),
            UICommandItem(
                command_id="reset_tokens",
                title="Reset Tokens",
                description="Reset token usage.",
                key_binding="F5",
            ),
        ]
    )

    print("UI summary:")
    print(json.dumps(state.summary(), ensure_ascii=False, indent=2))

    print("\nUI full state:")
    print(state.to_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())