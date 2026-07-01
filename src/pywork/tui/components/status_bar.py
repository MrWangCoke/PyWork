from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widget import Widget
from textual.widgets import Static


PermissionMode = Literal[
    "default",
    "accept_edits",
    "bypass_permissions",
    "plan",
]

RuntimeState = Literal[
    "idle",
    "thinking",
    "running_tool",
    "error",
]


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class StatusInfo:
    model: str
    provider: str
    permission_mode: str
    workspace_path: str
    token_usage: TokenUsage
    state: str
    message: str = ""


class StatusBar(Widget):
    """
    PyWork TUI 状态栏。

    显示：
    - 模型
    - 供应商
    - 权限模式
    - Token 用量
    - 当前运行状态
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 2;
        background: $surface;
        color: $text;
        border-top: solid $primary;
    }

    #status-line {
        width: 1fr;
        height: 1;
        padding: 0 1;
        text-overflow: ellipsis;
    }
    """

    def __init__(
        self,
        *,
        model: str = "deepseek-v4-flash",
        provider: str = "deepseek",
        permission_mode: str = "default",
        workspace_path: str = ".",
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)

        self.model = model
        self.provider = provider
        self.permission_mode = permission_mode
        self.workspace_path = workspace_path

        self.input_tokens = 0
        self.output_tokens = 0

        self.state: str = "idle"
        self.message: str = "ready"

    def compose(self) -> ComposeResult:
        yield Static("", id="status-line")

    def on_mount(self) -> None:
        self.refresh_status()

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def get_status_info(self) -> StatusInfo:
        return StatusInfo(
            model=self.model,
            provider=self.provider,
            permission_mode=self.permission_mode,
            workspace_path=self.workspace_path,
            token_usage=TokenUsage(
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
            ),
            state=self.state,
            message=self.message,
        )

    def set_model(
        self,
        model: str,
        *,
        provider: str | None = None,
    ) -> None:
        self.model = model

        if provider is not None:
            self.provider = provider

        self.refresh_status()

    def set_permission_mode(self, mode: str) -> None:
        self.permission_mode = mode
        self.refresh_status()

    def set_workspace_path(self, path: str) -> None:
        self.workspace_path = path
        self.refresh_status()

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

        self.refresh_status()

    def add_token_usage(
        self,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.input_tokens = max(0, self.input_tokens + input_tokens)
        self.output_tokens = max(0, self.output_tokens + output_tokens)

        self.refresh_status()

    def reset_token_usage(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.refresh_status()

    def set_state(
        self,
        state: RuntimeState,
        *,
        message: str = "",
    ) -> None:
        self.state = state
        self.message = message
        self.refresh_status()

    def set_idle(self, message: str = "ready") -> None:
        self.set_state("idle", message=message)

    def set_thinking(self, message: str = "waiting for model response") -> None:
        self.set_state("thinking", message=message)

    def set_running_tool(self, message: str = "running tool") -> None:
        self.set_state("running_tool", message=message)

    def set_error(self, message: str = "runtime not connected") -> None:
        self.set_state("error", message=message)

    def refresh_status(self) -> None:
        if not self.is_mounted:
            return

        line = self.query_one("#status-line", Static)
        line.update(self.render_status_line())

    def render_status_line(self) -> str:
        model_label = self.model if "/" in self.model else f"{self.model}/{self.provider}"

        return (
            f"{self.state}: {self.message}"
            f" | dir {self.workspace_path}"
            f" | model {model_label}"
            f" | mode {self.permission_mode} [Tab]"
            f" | Ctrl+P commands"
        )

    def render_left(self) -> str:
        return (
            f"model: {self.model} "
            f"| provider: {self.provider} "
            f"| permission: {self.permission_mode}"
        )

    def render_right(self) -> str:
        return (
            f"dir: {self.workspace_path} "
            f"| mode: {self.permission_mode} [Tab] "
            f"| Ctrl+P commands "
            f"| {self.state}: {self.message}"
        )


class StatusBarDemoApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #demo-main {
        height: 1fr;
        border: round $primary;
        padding: 1;
    }

    StatusBar {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),

        Binding("1", "idle", "Idle", priority=True),
        Binding("2", "thinking", "Thinking", priority=True),
        Binding("3", "tool", "Tool", priority=True),
        Binding("4", "error", "Error", priority=True),
        Binding("t", "tokens", "Add Tokens", priority=True),
        Binding("r", "reset", "Reset Tokens", priority=True),
        Binding("m", "model", "Switch Model", priority=True),
        Binding("tab", "permission", "Switch Permission", priority=True),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "StatusBar demo\n\n"
            "按 1：idle\n"
            "按 2：thinking\n"
            "按 3：running_tool\n"
            "按 4：error\n"
            "按 t：增加 token\n"
            "按 r：重置 token\n"
            "按 m：切换模型\n"
            "按 p：切换权限模式\n"
            "按 q：退出",
            id="demo-main",
        )
        yield StatusBar(id="status-bar")

    def get_status_bar(self) -> StatusBar:
        return self.query_one("#status-bar", StatusBar)

    def action_idle(self) -> None:
        self.get_status_bar().set_idle("ready")

    def action_thinking(self) -> None:
        self.get_status_bar().set_thinking("waiting for model response")

    def action_tool(self) -> None:
        self.get_status_bar().set_running_tool("bash")

    def action_error(self) -> None:
        self.get_status_bar().set_error("runtime not connected")

    def action_tokens(self) -> None:
        self.get_status_bar().add_token_usage(
            input_tokens=128,
            output_tokens=64,
        )

    def action_reset(self) -> None:
        status_bar = self.get_status_bar()
        status_bar.reset_token_usage()
        status_bar.set_idle("tokens reset")

    def action_model(self) -> None:
        status_bar = self.get_status_bar()

        if status_bar.model == "deepseek-v4-flash":
            status_bar.set_model(
                "qwen3.7-max",
                provider="qwen",
            )
        else:
            status_bar.set_model(
                "deepseek-v4-flash",
                provider="deepseek",
            )

    def action_permission(self) -> None:
        status_bar = self.get_status_bar()

        if status_bar.permission_mode == "default":
            status_bar.set_permission_mode("accept_edits")
        elif status_bar.permission_mode == "accept_edits":
            status_bar.set_permission_mode("plan")
        else:
            status_bar.set_permission_mode("default")


def main() -> int:
    app = StatusBarDemoApp()
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
