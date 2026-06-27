from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

from pywork.tui.components.chat_panel import ChatPanel
from pywork.tui.components.input_box import InputBox, InputSubmitted
from pywork.tui.components.status_bar import StatusBar


@dataclass(frozen=True)
class PyWorkTUIContext:
    workspace_path: str
    project_root: str
    config: dict[str, Any]

@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    usage: str
    aliases: tuple[str, ...] = ()


def get_config_value(
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


def estimate_tokens(text: str) -> int:
    """
    临时 token 估算。

    真正的 Token 统计后面接 LLM Provider 时再换成真实 usage。
    这里先用简单估算：中文/英文混合场景下，大约 2 个字符算 1 token。
    """
    text = text.strip()

    if not text:
        return 0

    return max(1, len(text) // 2)

def get_builtin_slash_commands() -> list[SlashCommand]:
    return [
        SlashCommand(
            name="/help",
            description="列出当前可用命令。",
            usage="/help",
            aliases=("/?", "/h"),
        ),
        SlashCommand(
            name="/clear",
            description="清空当前聊天消息。",
            usage="/clear",
            aliases=("/cls",),
        ),
        SlashCommand(
            name="/status",
            description="显示当前模型、权限模式、Token 用量等状态。",
            usage="/status",
            aliases=("/info",),
        ),
        SlashCommand(
            name="/doctor",
            description="打印环境诊断信息，包括 Python、OS、依赖状态。",
            usage="/doctor",
            aliases=("/diag",),
        ),
        SlashCommand(
            name="/tokens",
            description="显示当前 Token 用量。",
            usage="/tokens",
            aliases=(),
        ),
        SlashCommand(
            name="/reset-token",
            description="重置 Token 用量。",
            usage="/reset-token",
            aliases=("/reset-tokens", "/tokens reset"),
        ),
        SlashCommand(
            name="/exit",
            description="退出 PyWork TUI。",
            usage="/exit",
            aliases=("/quit",),
        ),
    ]


def normalize_slash_command(text: str) -> str:
    return " ".join(text.strip().lower().split())

def is_python_package_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def is_command_available(command: str) -> bool:
    return shutil.which(command) is not None


def render_dependency_status(
    *,
    name: str,
    available: bool,
    required: bool = True,
) -> str:
    icon = "✅" if available else ("❌" if required else "⚠️")
    kind = "required" if required else "optional"
    status = "ok" if available else "missing"

    return f"- {icon} `{name}` — {status} ({kind})"


def render_tui_doctor_report() -> str:
    """
    TUI 内部 doctor 报告。

    注意：
    CLI 版 doctor 已经在 entrypoints/doctor.py。
    这里是给 /doctor 命令显示用的轻量版。
    """
    python_version = platform.python_version()
    python_executable = sys.executable
    os_name = platform.system()
    os_release = platform.release()
    os_version = platform.version()
    machine = platform.machine()
    processor = platform.processor() or "unknown"

    required_python_packages = [
        "typer",
        "rich",
        "textual",
        "pydantic",
        "pydantic_settings",
        "httpx",
        "orjson",
        "psutil",
        "openai",
        "anthropic",
        "langchain_core",
        "langchain_openai",
        "langgraph",
        "aiosqlite",
    ]

    optional_python_packages = [
        "mcp",
        "git",
        "unidiff",
        "tree_sitter",
        "nbformat",
        "langgraph_supervisor",
        "deepagents",
    ]

    required_commands = [
        "git",
        "python",
    ]

    optional_commands = [
        "rg",
        "powershell",
        "pwsh",
        "docker",
        "wsl",
    ]

    lines: list[str] = [
        "PyWork Doctor",
        "",
        "Python:",
        "",
        f"- Version: `{python_version}`",
        f"- Executable: `{python_executable}`",
        f"- Supported: `3.12 <= Python < 3.14`",
        "",
        "OS:",
        "",
        f"- System: `{os_name}`",
        f"- Release: `{os_release}`",
        f"- Version: `{os_version}`",
        f"- Machine: `{machine}`",
        f"- Processor: `{processor}`",
        "",
        "Python dependencies:",
        "",
    ]

    for package in required_python_packages:
        lines.append(
            render_dependency_status(
                name=package,
                available=is_python_package_available(package),
                required=True,
            )
        )

    for package in optional_python_packages:
        lines.append(
            render_dependency_status(
                name=package,
                available=is_python_package_available(package),
                required=False,
            )
        )

    lines.extend(
        [
            "",
            "Commands:",
            "",
        ]
    )

    for command in required_commands:
        lines.append(
            render_dependency_status(
                name=command,
                available=is_command_available(command),
                required=True,
            )
        )

    for command in optional_commands:
        lines.append(
            render_dependency_status(
                name=command,
                available=is_command_available(command),
                required=False,
            )
        )

    missing_required_packages = [
        package
        for package in required_python_packages
        if not is_python_package_available(package)
    ]

    missing_required_commands = [
        command
        for command in required_commands
        if not is_command_available(command)
    ]

    lines.extend(
        [
            "",
            "Summary:",
            "",
        ]
    )

    if not missing_required_packages and not missing_required_commands:
        lines.append("- ✅ Required environment looks OK.")
    else:
        lines.append("- ❌ Required environment has missing items.")

        if missing_required_packages:
            lines.append(
                "- Missing Python packages: "
                + ", ".join(f"`{item}`" for item in missing_required_packages)
            )

        if missing_required_commands:
            lines.append(
                "- Missing commands: "
                + ", ".join(f"`{item}`" for item in missing_required_commands)
            )

    return "\n".join(lines)


class PyWorkApp(App[None]):
    """
    PyWork 主 TUI 应用。

    当前阶段目标：
    1. 启动 Textual App
    2. 渲染消息区
    3. 渲染输入区
    4. 渲染状态栏
    5. 输入后把用户消息显示到 ChatPanel
    6. Runtime 尚未接入时，返回占位助手消息
    """

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-layout {
        height: 1fr;
    }

    ChatPanel {
        height: 1fr;
    }

    InputBox {
        height: 10;
    }

    StatusBar {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", priority=True),
        Binding("ctrl+r", "reset_tokens", "Reset Tokens", priority=True),
        Binding("ctrl+s", "show_status", "Show Status", priority=True),
    ]

    def __init__(
        self,
        *,
        workspace_path: str | Path = ".",
        project_root: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        workspace = Path(workspace_path).expanduser().resolve()
        root = Path(project_root).expanduser().resolve() if project_root else workspace

        self.context = PyWorkTUIContext(
            workspace_path=str(workspace),
            project_root=str(root),
            config=config or {},
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="main-layout"):
            yield ChatPanel(id="chat-panel")
            yield InputBox(id="input-box")

        yield StatusBar(
            id="status-bar",
            model=self.get_model_name(),
            provider=self.get_provider_name(),
            permission_mode=self.get_permission_mode(),
        )

    def on_mount(self) -> None:
        self.title = "PyWork"

        chat_panel = self.get_chat_panel()
        status_bar = self.get_status_bar()

        chat_panel.append_system_message(
            "PyWork TUI started.\n\n"
            f"- Workspace: `{self.context.workspace_path}`\n"
            f"- Project Root: `{self.context.project_root}`\n\n"
            "Runtime is not connected yet. 当前阶段先测试 TUI 输入和消息渲染。"
        )

        status_bar.set_idle("ready")
        self.get_input_box().focus_input()


    def get_chat_panel(self) -> ChatPanel:
        return self.query_one("#chat-panel", ChatPanel)

    def get_input_box(self) -> InputBox:
        return self.query_one("#input-box", InputBox)

    def get_status_bar(self) -> StatusBar:
        return self.query_one("#status-bar", StatusBar)

    def get_model_name(self) -> str:
        return str(
            get_config_value(
                self.context.config,
                "default.model",
                "deepseek-v4-flash",
            )
        )

    def get_provider_name(self) -> str:
        return str(
            get_config_value(
                self.context.config,
                "default.provider",
                "deepseek",
            )
        )

    def get_permission_mode(self) -> str:
        return str(
            get_config_value(
                self.context.config,
                "permissions.mode",
                get_config_value(
                    self.context.config,
                    "app.permission_mode",
                    "default",
                ),
            )
        )
    
    def get_slash_commands(self) -> list[SlashCommand]:
        return get_builtin_slash_commands()

    def render_help_text(self) -> str:
        lines: list[str] = [
            "PyWork 可用命令：",
            "",
        ]

        for command in self.get_slash_commands():
            aliases = ""

            if command.aliases:
                aliases = "，别名：" + " / ".join(command.aliases)

            lines.append(
                f"- `{command.usage}` — {command.description}{aliases}"
            )

        lines.extend(
            [
                "",
                "快捷键：",
                "",
                "- `Ctrl+Enter` / `Ctrl+J`：提交输入",
                "- `Esc`：清空输入框",
                "- `Ctrl+L`：清空聊天",
                "- `F5`：重置 Token",
                "- `F6`：显示状态",
                "- `q`：退出",
            ]
        )

        return "\n".join(lines)

    def handle_slash_command(self, user_text: str) -> bool:
        """
        处理 TUI 内部 /command。

        返回 True：说明已经处理，不再进入普通聊天流程。
        返回 False：说明不是命令，继续作为普通用户输入。
        """
        command_text = normalize_slash_command(user_text)

        if not command_text.startswith("/"):
            return False

        if command_text in {"/help", "/?", "/h"}:
            self.get_chat_panel().append_system_message(self.render_help_text())
            self.get_status_bar().set_idle("help shown")
            self.get_input_box().focus_input()
            return True

        if command_text in {"/clear", "/cls"}:
            self.action_clear_chat()
            return True

        if command_text in {"/status", "/info"}:
            self.action_show_status()
            return True
        
        if command_text in {"/doctor", "/diag"}:
            self.action_show_doctor()
            return True

        if command_text == "/tokens":
            self.action_show_tokens()
            return True

        if command_text in {"/reset-token", "/reset-tokens", "/tokens reset"}:
            self.action_reset_tokens()
            return True

        if command_text in {"/exit", "/quit"}:
            self.exit()
            return True

        self.get_chat_panel().append_error_message(
            f"未知命令：`{user_text}`\n\n输入 `/help` 查看可用命令。"
        )
        self.get_status_bar().set_error("unknown command")
        self.get_input_box().focus_input()
        return True

    def on_input_submitted(self, message: InputSubmitted) -> None:
        """
        InputBox 提交后进入这里。

        当前阶段：
        - 显示用户消息
        - 更新 token 估算
        - 显示占位助手消息

        后面接 runtime/engine.py 后：
        - 这里会调用 Runtime
        - Runtime 再调用 LLM / Tools / Agent
        """
        message.stop()

        user_text = message.value.text

        if not user_text:
            return

        if self.handle_slash_command(user_text):
            return

        chat_panel = self.get_chat_panel()
        status_bar = self.get_status_bar()

        chat_panel.append_user_message(user_text)

        input_tokens = estimate_tokens(user_text)
        status_bar.add_token_usage(input_tokens=input_tokens)

        status_bar.set_thinking("runtime not connected")

        assistant_text = (
            "收到你的输入：\n\n"
            f"> {user_text}\n\n"
            "当前 TUI 已经能接收输入并渲染消息。\n\n"
            "下一阶段接入 `runtime/engine.py` 后，这里会替换成真实模型回复。"
        )

        chat_panel.append_assistant_message(assistant_text)

        output_tokens = estimate_tokens(assistant_text)
        status_bar.add_token_usage(output_tokens=output_tokens)

        status_bar.set_idle("ready")
        self.get_input_box().focus_input()

    def action_clear_chat(self) -> None:
        chat_panel = self.get_chat_panel()
        chat_panel.clear_messages()
        chat_panel.append_system_message("Chat cleared.")
        self.get_status_bar().set_idle("chat cleared")
        self.get_input_box().focus_input()

    def action_reset_tokens(self) -> None:
        status_bar = self.get_status_bar()
        status_bar.reset_token_usage()
        status_bar.set_idle("tokens reset")
        self.get_input_box().focus_input()

    def action_show_tokens(self) -> None:
        status_bar = self.get_status_bar()
        info = status_bar.get_status_info()

        self.get_chat_panel().append_system_message(
            "Token usage:\n\n"
            f"- Input: `{info.token_usage.input_tokens}`\n"
            f"- Output: `{info.token_usage.output_tokens}`\n"
            f"- Total: `{info.token_usage.total_tokens}`"
        )

        status_bar.set_idle("tokens shown")
        self.get_input_box().focus_input()

    def action_show_doctor(self) -> None:
        report = render_tui_doctor_report()

        self.get_chat_panel().append_system_message(report)
        self.get_status_bar().set_idle("doctor shown")
        self.get_input_box().focus_input()

    def action_show_status(self) -> None:
        status_bar = self.get_status_bar()
        info = status_bar.get_status_info()

        self.get_chat_panel().append_system_message(
            "Current status:\n\n"
            f"- Model: `{info.model}`\n"
            f"- Provider: `{info.provider}`\n"
            f"- Permission: `{info.permission_mode}`\n"
            f"- Tokens: `{info.token_usage.input_tokens}` in / "
            f"`{info.token_usage.output_tokens}` out / "
            f"`{info.token_usage.total_tokens}` total\n"
            f"- State: `{info.state}`\n"
            f"- Message: `{info.message}`"
        )

        self.get_input_box().focus_input()


def run_pywork_app(
    *,
    workspace_path: str | Path = ".",
    project_root: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    app = PyWorkApp(
        workspace_path=workspace_path,
        project_root=project_root,
        config=config,
    )
    app.run()


def main() -> int:
    demo_config = {
        "default": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
        },
        "permissions": {
            "mode": "default",
        },
    }

    run_pywork_app(
        workspace_path=".",
        project_root=".",
        config=demo_config,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
