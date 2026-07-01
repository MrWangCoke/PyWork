from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from pywork.runtime.controller import RuntimeController
from pywork.runtime.events import RuntimeEvent
from pywork.runtime.permission_gate import PermissionGateResult
from pywork.state.app_state import create_app_state
from pywork.tui.components.approval_dialog import (
    ApprovalDialog,
    ApprovalDialogResult,
)
from pywork.tui.components.chat_panel import ChatPanel
from pywork.tui.components.input_box import InputBox, InputSubmitted
from pywork.tui.components.status_bar import StatusBar
from pywork.tui.components.tool_log import ToolLog


PERMISSION_MODE_CYCLE: tuple[str, ...] = (
    "default",
    "accept_edits",
    "plan",
    "readonly",
    "bypass",
)


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
    text = text.strip()

    if not text:
        return 0

    return max(1, len(text) // 2)


def get_builtin_slash_commands() -> list[SlashCommand]:
    return [
        SlashCommand(
            name="/help",
            description="List available commands.",
            usage="/help",
            aliases=("/?", "/h"),
        ),
        SlashCommand(
            name="/clear",
            description="Clear chat messages and tool log.",
            usage="/clear",
            aliases=("/cls",),
        ),
        SlashCommand(
            name="/status",
            description="Show model, permission mode, token usage, and state.",
            usage="/status",
            aliases=("/info",),
        ),
        SlashCommand(
            name="/doctor",
            description="Print a lightweight environment diagnostic report.",
            usage="/doctor",
            aliases=("/diag",),
        ),
        SlashCommand(
            name="/tokens",
            description="Show current token usage.",
            usage="/tokens",
            aliases=(),
        ),
        SlashCommand(
            name="/reset-token",
            description="Reset token usage.",
            usage="/reset-token",
            aliases=("/reset-tokens", "/tokens reset"),
        ),
        SlashCommand(
            name="/exit",
            description="Exit PyWork TUI.",
            usage="/exit",
            aliases=("/quit",),
        ),
    ]


def normalize_slash_command(text: str) -> str:
    return " ".join(text.strip().lower().split())


class CommandsDialog(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
    ]

    DEFAULT_CSS = """
    CommandsDialog {
        align: center middle;
    }

    CommandsDialog > Container {
        width: 78%;
        max-width: 96;
        height: auto;
        max-height: 82%;
        border: round #666666;
        background: #181818;
        padding: 1 2;
    }

    CommandsDialog .dialog-title {
        height: 1;
        color: #eeeeee;
        text-style: bold;
        margin-bottom: 1;
    }

    #command-search {
        margin-bottom: 1;
    }

    #command-list {
        height: auto;
        min-height: 8;
        max-height: 24;
    }

    CommandsDialog .dialog-footer {
        height: 1;
        color: #999999;
        margin-top: 1;
    }
    """

    def __init__(self, commands: list[SlashCommand]) -> None:
        super().__init__()
        self.commands = commands
        self.filtered_commands = list(commands)

    def compose(self) -> ComposeResult:
        with Container():
            yield Static("Commands", classes="dialog-title")
            yield Input(
                placeholder="Search commands",
                id="command-search",
            )
            yield ListView(id="command-list")
            yield Static(
                "Type to filter · Up/Down move · Enter run · Esc close",
                classes="dialog-footer",
            )

    async def on_mount(self) -> None:
        await self.refresh_commands("")
        self.query_one("#command-search", Input).focus()

    def command_matches(self, command: SlashCommand, query: str) -> bool:
        if not query:
            return True

        haystack = " ".join(
            [
                command.name,
                command.usage,
                command.description,
                " ".join(command.aliases),
            ]
        ).lower()

        return query.lower() in haystack

    def render_command_row(self, command: SlashCommand) -> str:
        aliases = ""

        if command.aliases:
            aliases = "  " + ", ".join(command.aliases)

        return f"{command.usage:<16} {command.description}{aliases}"

    async def refresh_commands(self, query: str) -> None:
        self.filtered_commands = [
            command
            for command in self.commands
            if self.command_matches(command, query.strip())
        ]

        list_view = self.query_one("#command-list", ListView)
        await list_view.clear()

        if not self.filtered_commands:
            await list_view.append(
                ListItem(Label("No commands found"), disabled=True)
            )
            list_view.index = None
            return

        for command in self.filtered_commands:
            await list_view.append(
                ListItem(Label(self.render_command_row(command)))
            )

        list_view.index = 0

    @on(Input.Changed, "#command-search")
    async def on_search_changed(self, event: Input.Changed) -> None:
        await self.refresh_commands(event.value)

    @on(ListView.Selected, "#command-list")
    def on_command_selected(self, event: ListView.Selected) -> None:
        self.execute_selected()

    def execute_selected(self) -> None:
        if not self.filtered_commands:
            return

        list_view = self.query_one("#command-list", ListView)
        index = list_view.index

        if index is None:
            index = 0

        if index < 0 or index >= len(self.filtered_commands):
            return

        self.dismiss(self.filtered_commands[index].usage)

    def on_key(self, event: events.Key) -> None:
        key = event.key.lower()
        list_view = self.query_one("#command-list", ListView)

        if key == "enter":
            event.prevent_default()
            event.stop()
            self.execute_selected()
            return

        if key in {"down", "ctrl+n"} and self.filtered_commands:
            event.prevent_default()
            event.stop()
            current = list_view.index or 0
            list_view.index = min(current + 1, len(self.filtered_commands) - 1)
            return

        if key in {"up", "ctrl+p"} and self.filtered_commands:
            event.prevent_default()
            event.stop()
            current = list_view.index or 0
            list_view.index = max(current - 1, 0)
            return

    def action_close(self) -> None:
        self.dismiss(None)


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
    icon = "OK" if available else ("MISSING" if required else "OPTIONAL")
    kind = "required" if required else "optional"
    status = "ok" if available else "missing"

    return f"- {icon} `{name}` - {status} ({kind})"


def render_tui_doctor_report() -> str:
    """Return a lightweight doctor report for the TUI /doctor command."""
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
        lines.append("- OK Required environment looks OK.")
    else:
        lines.append("- MISSING Required environment has missing items.")

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
    """Main Textual application for PyWork."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-layout {
        height: 1fr;
    }

    #main-area {
        height: 1fr;
    }

    #chat-panel {
        width: 2fr;
        height: 100%;
        border: round $accent;
    }

    #tool-log {
        width: 1fr;
        height: 100%;
        border: round $surface;
    }

    #input-box {
        height: 12;
    }

    #status-bar {
        dock: bottom;
        height: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+l", "clear_chat", "Clear Chat", priority=True),
        Binding("ctrl+r", "reset_tokens", "Reset Tokens", priority=True),
        Binding("ctrl+s", "show_status", "Show Status", priority=True),
        Binding("ctrl+p", "show_commands", "Commands", priority=True),
        Binding("tab", "cycle_permission_mode", "Switch Mode", priority=True),
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

        self.workspace = workspace
        self.config = config or {}
        self.permission_mode = self.get_configured_permission_mode(self.config)

        self.context = PyWorkTUIContext(
            workspace_path=str(workspace),
            project_root=str(root),
            config=self.config,
        )

        self.runtime_controller: RuntimeController | None = None
        self.runtime_busy = False

        self.chat_panel: ChatPanel | None = None
        self.input_box: InputBox | None = None
        self.status_bar: StatusBar | None = None
        self.tool_log: ToolLog | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="main-layout"):
            with Horizontal(id="main-area"):
                yield ChatPanel(id="chat-panel")
                yield ToolLog(id="tool-log")

            yield InputBox(id="input-box")

        yield StatusBar(
            id="status-bar",
            model=self.get_model_name(),
            provider=self.get_provider_name(),
            permission_mode=self.get_permission_mode(),
            workspace_path=str(self.workspace),
        )

    def on_mount(self) -> None:
        self.title = "PyWork"

        self.chat_panel = self.query_one("#chat-panel", ChatPanel)
        self.input_box = self.query_one("#input-box", InputBox)
        self.status_bar = self.query_one("#status-bar", StatusBar)
        self.tool_log = self.query_one("#tool-log", ToolLog)

        self.runtime_controller = self.create_runtime_controller()

        if self.status_bar is not None:
            self.status_bar.set_model(
                self.get_configured_model_label(),
                provider=self.get_configured_provider_name(),
            )
            self.status_bar.set_workspace_path(str(self.workspace))
            self.status_bar.set_permission_mode(self.get_permission_mode())

        self.chat_panel.append_system_message(
            "PyWork TUI ready. RuntimeController.stream() is connected."
        )

        self.tool_log.append_status("RuntimeController connected.")
        self.status_bar.set_idle("ready")

        with suppress(Exception):
            self.input_box.focus_input()

    def get_runtime_config(self) -> dict[str, Any]:
        runtime_config = dict(self.config)
        runtime_config["permissions"] = {
            **runtime_config.get("permissions", {}),
            "mode": self.permission_mode,
        }
        runtime_config["agent"] = {
            **runtime_config.get("agent", {}),
            "max_iterations": 1000,
            "max_context_messages": 5000,
        }
        runtime_config["llm"] = {
            "default_provider": "qwen",
            "fallback_to_mock": False,
            "providers": {
                "qwen": {
                    "provider": "qwen",
                    "api_format": "openai_compatible",
                    "model": "qwen3.6-flash",
                    "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "api_key_env": "DASHSCOPE_API_KEY",
                    "temperature": 0.2,
                    "max_tokens": 2048,
                }
            },
        }

        return runtime_config

    def create_runtime_controller(self) -> RuntimeController:
        workspace_path = self.context.workspace_path
        project_root = self.context.project_root
        runtime_config = self.get_runtime_config()

        try:
            app_state = create_app_state(
                workspace_path=workspace_path,
                project_root=project_root,
                config=runtime_config,
            )
        except TypeError:
            app_state = create_app_state(
                config=runtime_config,
            )

        return RuntimeController(
            app_state=app_state,
            approval_handler=self.request_tool_approval,
        )

    async def request_tool_approval(
        self,
        gate_result: PermissionGateResult,
    ) -> ApprovalDialogResult | None:
        """
        Open the approval dialog for ask / ask_elevated runtime tool calls.
        """
        decision = gate_result.decision
        title = "Approve Tool Operation"

        if decision.is_elevated:
            title = "Approve Elevated Operation"

        return await self.push_screen_wait(
            ApprovalDialog(
                decision,
                title=title,
            )
        )

    def get_chat_panel(self) -> ChatPanel:
        if self.chat_panel is not None:
            return self.chat_panel

        return self.query_one("#chat-panel", ChatPanel)

    def get_input_box(self) -> InputBox:
        if self.input_box is not None:
            return self.input_box

        return self.query_one("#input-box", InputBox)

    def get_status_bar(self) -> StatusBar:
        if self.status_bar is not None:
            return self.status_bar

        return self.query_one("#status-bar", StatusBar)

    def get_tool_log(self) -> ToolLog:
        if self.tool_log is not None:
            return self.tool_log

        return self.query_one("#tool-log", ToolLog)

    def get_model_name(self) -> str:
        return self.get_configured_model_label()

    def get_provider_name(self) -> str:
        return self.get_configured_provider_name()

    def get_configured_provider_name(self) -> str:
        llm_config = self.get_runtime_config().get("llm", {})

        if isinstance(llm_config, dict):
            default_provider = llm_config.get("default_provider")

            if default_provider:
                return str(default_provider)

        return "mock"

    def get_configured_model_label(self) -> str:
        llm_config = self.get_runtime_config().get("llm", {})

        if isinstance(llm_config, dict):
            default_provider = llm_config.get("default_provider")
            providers = llm_config.get("providers", {})

            if default_provider and isinstance(providers, dict):
                provider_config = providers.get(default_provider, {})

                if isinstance(provider_config, dict):
                    model = provider_config.get("model")

                    if model:
                        return f"{model}/{default_provider}"

        return "mock/local"

    def get_permission_mode(self) -> str:
        return self.permission_mode

    @staticmethod
    def get_configured_permission_mode(config: dict[str, Any]) -> str:
        mode = str(
            get_config_value(
                config,
                "permissions.mode",
                "default",
            )
        )

        if mode in PERMISSION_MODE_CYCLE:
            return mode

        return "default"

    def set_permission_mode(self, mode: str) -> None:
        if mode not in PERMISSION_MODE_CYCLE:
            mode = "default"

        self.permission_mode = mode
        self.config.setdefault("permissions", {})["mode"] = mode

        if self.status_bar is not None:
            self.status_bar.set_permission_mode(mode)

        if not self.runtime_busy:
            self.runtime_controller = self.create_runtime_controller()

    def get_slash_commands(self) -> list[SlashCommand]:
        return get_builtin_slash_commands()

    def render_help_text(self) -> str:
        lines: list[str] = [
            "PyWork available commands:",
            "",
        ]

        for command in self.get_slash_commands():
            aliases = ""

            if command.aliases:
                aliases = ", aliases: " + " / ".join(command.aliases)

            lines.append(
                f"- `{command.usage}` - {command.description}{aliases}"
            )

        lines.extend(
            [
                "",
                "Shortcuts:",
                "",
                "- `Enter`: submit input / confirm dialog",
                "- `Ctrl+P`: show commands",
                "- `Tab`: switch permission mode",
                "- `Esc`: clear input",
                "- `Ctrl+L`: clear chat and tool log",
                "- `Ctrl+R`: reset tokens",
                "- `Ctrl+S`: show status",
            ]
        )

        return "\n".join(lines)

    def handle_slash_command(self, user_text: str) -> bool:
        command_name = user_text.strip().split(maxsplit=1)[0].lower()

        if command_name == "/tool":
            return False

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
            f"Unknown command: `{user_text}`\n\nType `/help` to see available commands."
        )
        self.get_status_bar().set_error("unknown command")
        self.get_input_box().focus_input()
        return True

    def get_submitted_text_from_event(self, event: Any) -> str:
        for attr in ("text", "value", "content", "message"):
            value = getattr(event, attr, None)

            if isinstance(value, str):
                return value.strip()

            text = getattr(value, "text", None)
            if isinstance(text, str):
                return text.strip()

        return ""

    def on_input_submitted(self, message: InputSubmitted) -> None:
        message.stop()

        user_text = self.get_submitted_text_from_event(message)

        if not user_text:
            return

        if self.runtime_busy:
            if self.chat_panel is not None:
                self.chat_panel.append_system_message(
                    "A runtime task is still running. Please wait for it to finish."
                )
            return

        if user_text.startswith("/"):
            handled = self.handle_slash_command(user_text)

            if handled:
                return

        chat_panel = self.get_chat_panel()
        status_bar = self.get_status_bar()
        tool_log = self.get_tool_log()

        chat_panel.append_user_message(user_text)
        status_bar.add_token_usage(input_tokens=estimate_tokens(user_text))
        tool_log.append_status(f"user submitted: {user_text}")

        self.run_worker(
            self.run_runtime_stream(user_text),
            name="runtime-stream",
            group="runtime",
            exclusive=True,
        )

    async def run_runtime_stream(self, user_text: str) -> None:
        if self.runtime_controller is None:
            if self.chat_panel is not None:
                self.chat_panel.append_error_message("RuntimeController is not initialized.")
            return

        self.runtime_busy = True

        if self.status_bar is not None:
            self.status_bar.set_thinking()

        if self.tool_log is not None:
            self.tool_log.append_status("runtime stream started")

        try:
            async for event in self.runtime_controller.stream(user_text):
                self.handle_runtime_event(event)

            result = self.runtime_controller.get_last_stream_result()

            if result is not None:
                if result.success:
                    if result.output and self.chat_panel is not None:
                        self.chat_panel.append_assistant_message(result.output)

                    if result.output and self.status_bar is not None:
                        self.status_bar.add_token_usage(
                            output_tokens=estimate_tokens(result.output)
                        )

                    if self.status_bar is not None:
                        self.status_bar.set_idle()

                else:
                    error_text = result.error or "RuntimeController failed."

                    if self.chat_panel is not None:
                        self.chat_panel.append_error_message(error_text)

                    if self.status_bar is not None:
                        self.status_bar.set_error(error_text)

            else:
                if self.status_bar is not None:
                    self.status_bar.set_idle()

        except Exception as exc:
            error_text = str(exc)

            if self.tool_log is not None:
                self.tool_log.append_error(error_text)

            if self.chat_panel is not None:
                self.chat_panel.append_error_message(error_text)

            if self.status_bar is not None:
                self.status_bar.set_error(error_text)

        finally:
            self.runtime_busy = False

            if self.tool_log is not None:
                self.tool_log.append_status("runtime stream finished")

            if self.input_box is not None:
                with suppress(Exception):
                    self.input_box.focus_input()

    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        if self.tool_log is not None:
            self.tool_log.append_runtime_event(event)

        event_type = getattr(event, "event_type", None)
        event_type_value = getattr(event_type, "value", str(event_type))

        if self.status_bar is None:
            return

        if event_type_value == "status":
            status = str(getattr(event, "status", "") or "")

            if status == "running_tool":
                metadata = getattr(event, "metadata", {}) or {}
                tool_name = metadata.get("tool_name", "tool")
                self.status_bar.set_running_tool(tool_name)

            elif status in {"thinking", "llm_response"}:
                self.status_bar.set_thinking()

            elif status in {"finished", "tool_finished"}:
                self.status_bar.set_idle()

        elif event_type_value == "error":
            content = str(
                getattr(event, "content", "")
                or getattr(event, "message", "")
                or "runtime error"
            )
            self.status_bar.set_error(content)

    def action_clear_chat(self) -> None:
        chat_panel = self.get_chat_panel()
        chat_panel.clear_messages()
        chat_panel.append_system_message("Chat cleared.")
        self.get_tool_log().clear()
        self.get_status_bar().set_idle("chat cleared")
        self.get_input_box().focus_input()
    def action_reset_tokens(self) -> None:
        status_bar = self.get_status_bar()
        status_bar.reset_token_usage()
        self.get_chat_panel().append_system_message("Token usage reset.")
        status_bar.set_idle("tokens reset")
        self.get_input_box().focus_input()

    def action_cycle_permission_mode(self) -> None:
        current_mode = self.get_permission_mode()

        try:
            current_index = PERMISSION_MODE_CYCLE.index(current_mode)
        except ValueError:
            current_index = 0

        next_mode = PERMISSION_MODE_CYCLE[
            (current_index + 1) % len(PERMISSION_MODE_CYCLE)
        ]

        self.set_permission_mode(next_mode)
        self.get_status_bar().set_idle(f"mode switched to {next_mode}")
        self.get_input_box().focus_input()

    def action_show_commands(self) -> None:
        self.push_screen(
            CommandsDialog(self.get_slash_commands()),
            self.handle_command_dialog_result,
        )

    def handle_command_dialog_result(self, command_text: str | None) -> None:
        if not command_text:
            self.get_input_box().focus_input()
            return

        self.handle_slash_command(command_text)

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "") or "")

        if key.lower() == "ctrl+r":
            self.action_reset_tokens()

            if hasattr(event, "stop"):
                event.stop()

        if key.lower() == "tab":
            self.action_cycle_permission_mode()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

        if key.lower() == "ctrl+p":
            self.action_show_commands()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

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
