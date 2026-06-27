from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from pywork.schemas.tool_schema import ToolCall, ToolResult
from pywork.state.session_state import (
    SessionMessage,
    SessionState,
    SessionStatus,
    create_session_state,
)
from pywork.state.ui_state import UIState, create_ui_state
from pywork.tools.registry import ToolRegistry, create_default_registry


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def resolve_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def get_nested_config_value(
    config: dict[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = config

    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


@dataclass
class AppState:
    """
    PyWork 全局应用状态。

    一个进程中通常只存在一个 AppState。

    它负责统一管理：
    - 工作区信息
    - 配置信息
    - 当前会话状态
    - 当前 UI 状态
    - 工具注册表
    - 运行时标记
    """

    workspace_path: str = "."
    project_root: str = "."
    config: dict[str, Any] = field(default_factory=dict)

    session: SessionState = field(default_factory=create_session_state)
    ui: UIState = field(default_factory=create_ui_state)
    tool_registry: ToolRegistry = field(default_factory=create_default_registry)

    is_initialized: bool = False
    is_shutting_down: bool = False

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.workspace_path = resolve_path(self.workspace_path)
        self.project_root = resolve_path(self.project_root)

        self.session.workspace_path = self.workspace_path
        self.session.project_root = self.project_root

        self.sync_ui_from_config()
        self.touch()

    def touch(self) -> None:
        self.updated_at = utc_now()

    def initialize(self) -> None:
        self.is_initialized = True
        self.is_shutting_down = False
        self.session.set_status(SessionStatus.IDLE)
        self.ui.set_idle("ready")
        self.touch()

    def shutdown(self) -> None:
        self.is_shutting_down = True
        self.session.close()
        self.ui.set_idle("shutting down")
        self.touch()

    def set_workspace(
        self,
        *,
        workspace_path: str | Path,
        project_root: str | Path | None = None,
    ) -> None:
        self.workspace_path = resolve_path(workspace_path)
        self.project_root = resolve_path(project_root or workspace_path)

        self.session.workspace_path = self.workspace_path
        self.session.project_root = self.project_root

        self.touch()

    def set_config(self, config: dict[str, Any]) -> None:
        self.config = config
        self.sync_ui_from_config()
        self.touch()

    def get_config_value(
        self,
        dotted_key: str,
        default: Any = None,
    ) -> Any:
        return get_nested_config_value(
            self.config,
            dotted_key,
            default,
        )

    def sync_ui_from_config(self) -> None:
        model = str(
            get_nested_config_value(
                self.config,
                "default.model",
                "deepseek-v4-flash",
            )
        )

        provider = str(
            get_nested_config_value(
                self.config,
                "default.provider",
                "deepseek",
            )
        )

        permission_mode = str(
            get_nested_config_value(
                self.config,
                "permissions.mode",
                get_nested_config_value(
                    self.config,
                    "app.permission_mode",
                    "default",
                ),
            )
        )

        self.ui.set_model(
            model,
            provider=provider,
        )
        self.ui.set_permission_mode(permission_mode)

    def set_tool_registry(self, registry: ToolRegistry) -> None:
        self.tool_registry = registry
        self.touch()

    def reset_tool_registry(self) -> ToolRegistry:
        self.tool_registry = create_default_registry()
        self.touch()
        return self.tool_registry

    def start_new_session(
        self,
        *,
        title: str = "New Session",
        metadata: dict[str, Any] | None = None,
    ) -> SessionState:
        self.session = create_session_state(
            workspace_path=self.workspace_path,
            project_root=self.project_root,
            title=title,
            metadata=metadata or {},
        )

        self.ui.clear_input_text()
        self.ui.clear_error()
        self.ui.clear_notifications()
        self.ui.reset_token_usage()
        self.ui.set_idle("new session")

        self.touch()
        return self.session

    def set_session_status(self, status: SessionStatus | str) -> None:
        self.session.set_status(status)

        if status == SessionStatus.THINKING or str(status) == "thinking":
            self.ui.set_thinking("thinking")
        elif status == SessionStatus.RUNNING_TOOL or str(status) == "running_tool":
            self.ui.set_running_tool("running tool")
        elif status == SessionStatus.ERROR or str(status) == "error":
            self.ui.set_error("session error")
        else:
            self.ui.set_idle(str(status))

        self.touch()

    def add_system_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = self.session.add_system_message(
            content,
            metadata=metadata,
        )
        self.touch()
        return message

    def add_user_message(
        self,
        content: str,
        *,
        token_estimate: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = self.session.add_user_message(
            content,
            token_estimate=token_estimate,
            metadata=metadata,
        )

        if token_estimate > 0:
            self.add_token_usage(
                input_tokens=token_estimate,
            )

        self.touch()
        return message

    def add_assistant_message(
        self,
        content: str,
        *,
        token_estimate: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = self.session.add_assistant_message(
            content,
            token_estimate=token_estimate,
            metadata=metadata,
        )

        if token_estimate > 0:
            self.add_token_usage(
                output_tokens=token_estimate,
            )

        self.touch()
        return message

    def add_tool_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = self.session.add_tool_message(
            content,
            metadata=metadata,
        )
        self.touch()
        return message

    def add_error_message(
        self,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> SessionMessage:
        message = self.session.add_error_message(
            content,
            metadata=metadata,
        )
        self.ui.set_error(content)
        self.touch()
        return message

    def clear_messages(self) -> None:
        self.session.clear_messages()
        self.ui.set_idle("chat cleared")
        self.touch()

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.session.add_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.ui.add_token_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        self.touch()

    def reset_token_usage(self) -> None:
        self.session.reset_token_usage()
        self.ui.reset_token_usage()
        self.touch()

    def add_tool_call(self, call: ToolCall) -> ToolCall:
        self.session.add_tool_call(call)
        self.ui.set_running_tool(call.tool_name)
        self.touch()
        return call

    def add_tool_result(self, result: ToolResult) -> ToolResult:
        self.session.add_tool_result(result)

        if result.success:
            self.ui.set_idle("tool finished")
        else:
            self.ui.set_error(result.error or "tool failed")

        self.touch()
        return result

    def get_status_summary(self) -> dict[str, Any]:
        return {
            "initialized": self.is_initialized,
            "shutting_down": self.is_shutting_down,
            "workspace_path": self.workspace_path,
            "project_root": self.project_root,
            "session": self.session.summary(),
            "ui": self.ui.summary(),
            "tools": self.tool_registry.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace_path": self.workspace_path,
            "project_root": self.project_root,
            "config": self.config,
            "is_initialized": self.is_initialized,
            "is_shutting_down": self.is_shutting_down,
            "session": self.session.to_dict(),
            "ui": self.ui.to_dict(),
            "tool_registry": self.tool_registry.to_dict(),
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


_APP_STATE: AppState | None = None
_APP_STATE_LOCK = RLock()


def create_app_state(
    *,
    workspace_path: str | Path = ".",
    project_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AppState:
    resolved_workspace = resolve_path(workspace_path)
    resolved_root = resolve_path(project_root or workspace_path)

    session = create_session_state(
        workspace_path=resolved_workspace,
        project_root=resolved_root,
        title="PyWork Session",
    )

    ui = create_ui_state(
        model=str(
            get_nested_config_value(
                config or {},
                "default.model",
                "deepseek-v4-flash",
            )
        ),
        provider=str(
            get_nested_config_value(
                config or {},
                "default.provider",
                "deepseek",
            )
        ),
        permission_mode=str(
            get_nested_config_value(
                config or {},
                "permissions.mode",
                get_nested_config_value(
                    config or {},
                    "app.permission_mode",
                    "default",
                ),
            )
        ),
    )

    return AppState(
        workspace_path=resolved_workspace,
        project_root=resolved_root,
        config=config or {},
        session=session,
        ui=ui,
        tool_registry=create_default_registry(),
        metadata=metadata or {},
    )


def init_app_state(
    *,
    workspace_path: str | Path = ".",
    project_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    force: bool = False,
) -> AppState:
    """
    初始化全局 AppState。

    force=False：
        如果已经存在全局状态，就直接返回已有状态。

    force=True：
        重新创建全局状态。
    """
    global _APP_STATE

    with _APP_STATE_LOCK:
        if _APP_STATE is not None and not force:
            return _APP_STATE

        _APP_STATE = create_app_state(
            workspace_path=workspace_path,
            project_root=project_root,
            config=config,
            metadata=metadata,
        )
        _APP_STATE.initialize()

        return _APP_STATE


def get_app_state() -> AppState:
    """
    获取全局 AppState。

    如果还没初始化，则使用默认参数初始化。
    """
    global _APP_STATE

    with _APP_STATE_LOCK:
        if _APP_STATE is None:
            _APP_STATE = init_app_state()

        return _APP_STATE


def reset_app_state(
    *,
    workspace_path: str | Path = ".",
    project_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AppState:
    """
    重置全局 AppState。

    测试和新会话启动时可用。
    """
    return init_app_state(
        workspace_path=workspace_path,
        project_root=project_root,
        config=config,
        metadata=metadata,
        force=True,
    )


def has_app_state() -> bool:
    return _APP_STATE is not None


def clear_app_state() -> None:
    global _APP_STATE

    with _APP_STATE_LOCK:
        _APP_STATE = None


def main() -> int:
    config = {
        "default": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
        },
        "permissions": {
            "mode": "default",
        },
    }

    state = reset_app_state(
        workspace_path=".",
        project_root=".",
        config=config,
        metadata={
            "demo": True,
        },
    )

    state.add_system_message("PyWork app state initialized.")
    state.add_user_message(
        "hello",
        token_estimate=2,
    )
    state.add_assistant_message(
        "你好，我是 PyWork。",
        token_estimate=5,
    )

    call = state.tool_registry.create_call(
        "echo",
        {
            "text": "hello",
        },
    )

    state.add_tool_call(call)

    result = ToolResult.success_result(
        call=call,
        content="hello",
        data={
            "text": "hello",
        },
    )

    state.add_tool_result(result)

    print("AppState summary:")
    print(json.dumps(state.get_status_summary(), ensure_ascii=False, indent=2))

    print("\nAppState full state:")
    print(state.to_json(indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())