from __future__ import annotations

import copy
import importlib.util
import inspect
import platform
import shutil
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import events, on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static, TextArea

from pywork.runtime.controller import RuntimeController
from pywork.runtime.events import RuntimeEvent
from pywork.runtime.permission_gate import PermissionGateResult
from pywork.state.app_state import create_app_state
from pywork.subagents.manager import SubAgentTaskRequest
from pywork.tui.components.agents import AgentActivityPanel
from pywork.tui.components.approval_dialog import (
    ApprovalDialog,
    ApprovalDialogResult,
)
from pywork.tui.components.chat_panel import ChatPanel
from pywork.tui.components.input_box import InputBox, InputSubmitted
from pywork.tui.components.status_bar import StatusBar
from pywork.tui.components.tool_log import ToolLog
from pywork.tui.components.tasks import (
    TaskProgressPanel,
    TaskProgressSnapshot,
    build_task_snapshot,
    build_task_snapshot_from_manager,
    collect_subagent_run_records,
    collect_stats,
)
from pywork.tui.components.teams import (
    TeamViewPanel,
    build_team_snapshot,
)
from pywork.tui.components.friendly_names import (
    friendly_agent_activity,
    friendly_agent_label,
    friendly_task_title,
    friendly_team_member_label,
    role_label,
)


PERMISSION_MODE_CYCLE: tuple[str, ...] = (
    "default",
    "accept_edits",
    "plan",
    "readonly",
    "bypass_permissions",
)


SIDE_PANEL_TOOL_LOG = "tool_log"
SIDE_PANEL_TASKS = "tasks"
SIDE_PANEL_AGENTS = "agents"
SIDE_PANEL_TEAM = "team"

SIDE_PANEL_VIEWS: tuple[str, ...] = (
    SIDE_PANEL_TOOL_LOG,
    SIDE_PANEL_TASKS,
    SIDE_PANEL_AGENTS,
    SIDE_PANEL_TEAM,
)

SIDE_PANEL_LABELS: dict[str, str] = {
    SIDE_PANEL_TOOL_LOG: "Tool Log",
    SIDE_PANEL_TASKS: "Tasks",
    SIDE_PANEL_AGENTS: "Agents",
    SIDE_PANEL_TEAM: "Team",
}

SIDE_PANEL_ALIASES: dict[str, str] = {
    "1": SIDE_PANEL_TOOL_LOG,
    "log": SIDE_PANEL_TOOL_LOG,
    "tool": SIDE_PANEL_TOOL_LOG,
    "tool-log": SIDE_PANEL_TOOL_LOG,
    "tools": SIDE_PANEL_TOOL_LOG,
    "2": SIDE_PANEL_TASKS,
    "task": SIDE_PANEL_TASKS,
    "tasks": SIDE_PANEL_TASKS,
    "jobs": SIDE_PANEL_TASKS,
    "3": SIDE_PANEL_AGENTS,
    "agent": SIDE_PANEL_AGENTS,
    "agents": SIDE_PANEL_AGENTS,
    "runs": SIDE_PANEL_AGENTS,
    "4": SIDE_PANEL_TEAM,
    "team": SIDE_PANEL_TEAM,
    "mailbox": SIDE_PANEL_TEAM,
}


def normalize_side_panel_view(value: str | None) -> str:
    text = str(value or SIDE_PANEL_TOOL_LOG).strip().lower()

    return SIDE_PANEL_ALIASES.get(
        text,
        SIDE_PANEL_TOOL_LOG,
    )


TASK_PANEL_FALLBACK_POLL_SECONDS = 5.0

TASK_RUNTIME_EVENT_STATUSES: set[str] = {
    "task_created",
    "task_queued",
    "task_started",
    "task_retrying",
    "task_finished",
    "task_failed",
    "task_cancelled",
    "task_aborted",
    "task_updated",
}

TUI_PERMISSION_MODE_ALIASES: dict[str, str] = {
    "": "default",
    "default": "default",
    "normal": "default",
    "accept": "accept_edits",
    "accept-edits": "accept_edits",
    "accept_edits": "accept_edits",
    "plan": "plan",
    "planning": "plan",
    "readonly": "readonly",
    "read_only": "readonly",
    "read-only": "readonly",
    "safe": "readonly",
    "bypass": "bypass_permissions",
    "bypass_permissions": "bypass_permissions",
    "dangerous": "bypass_permissions",
}


def normalize_tui_permission_mode(mode: str | None) -> str:
    text = str(mode or "default").strip().lower()

    normalized = TUI_PERMISSION_MODE_ALIASES.get(text, "default")

    if normalized in PERMISSION_MODE_CYCLE:
        return normalized

    return "default"


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


RUNTIME_BUSY_ALLOWED_SLASH_COMMANDS: set[str] = {
    "/help",
    "/?",
    "/h",
    "/status",
    "/info",
    "/doctor",
    "/diag",
    "/tokens",
    "/tasks",
    "/task-list",
    "/jobs",
    "/agents",
    "/agent",
    "/runs",
    "/team",
    "/mailbox",
    "/tool-log",
    "/log",
    "/tools",
    "/copy-log",
    "/copy-tool-log",
    "/panel",
    "/side",
    "/task",
    "/task-output",
    "/output",
    "/stop",
    "/stop-task",
    "/cancel",
    "/cancel-task",
    "/abort",
    "/abort-runtime",
}


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


def get_default_tui_llm_config() -> dict[str, Any]:
    """
    TUI 默认 LLM 配置。

    只有用户没有提供 config["llm"] 时才使用。
    不要在 get_runtime_config() 里强行覆盖用户自己的 llm 配置。
    """
    return {
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


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def enum_or_text(value: Any) -> str:
    raw = getattr(value, "value", value)

    if raw is None:
        return ""

    return str(raw)


def safe_get_attr(
    value: Any,
    name: str,
    default: Any = None,
) -> Any:
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(name, default)

    return getattr(value, name, default)


def object_to_display_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    to_dict = getattr(value, "to_dict", None)

    if callable(to_dict):
        try:
            data = to_dict()

            if isinstance(data, dict):
                return data
        except Exception:
            pass

    result: dict[str, Any] = {}

    for name in (
        "id",
        "task_id",
        "name",
        "title",
        "description",
        "agent_id",
        "agent_name",
        "status",
        "result",
        "error",
        "metadata",
        "created_at",
        "started_at",
        "finished_at",
        "updated_at",
    ):
        item = getattr(value, name, None)

        if item is not None:
            result[name] = item

    return result


def compact_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, dict):
        parts = []

        for key, item in value.items():
            parts.append(f"{key}: {compact_value(item)}")

        return "\n".join(parts)

    if isinstance(value, list):
        return "\n".join(
            f"- {compact_value(item)}"
            for item in value
        )

    raw = getattr(value, "value", value)

    return str(raw)


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
            name="/tool-log",
            description="Show Tool Log side panel.",
            usage="/tool-log",
            aliases=("/log", "/tools"),
        ),
        SlashCommand(
            name="/copy-log",
            description="Copy Tool Log content to clipboard.",
            usage="/copy-log",
            aliases=("/copy-tool-log",),
        ),
        SlashCommand(
            name="/tasks",
            description="Show background tasks.",
            usage="/tasks",
            aliases=("/task-list", "/jobs"),
        ),
        SlashCommand(
            name="/agents",
            description="Show SubAgent runs.",
            usage="/agents",
            aliases=("/agent", "/runs"),
        ),
        SlashCommand(
            name="/team",
            description="Show runtime team and mailbox.",
            usage="/team",
            aliases=("/mailbox",),
        ),
        SlashCommand(
            name="/panel",
            description="Switch side panel: log, tasks, agents, team.",
            usage="/panel <log|tasks|agents|team>",
            aliases=("/side",),
        ),
        SlashCommand(
            name="/exit",
            description="Exit PyWork TUI.",
            usage="/exit",
            aliases=("/quit",),
        ),
        SlashCommand(
            name="/task",
            description="Show task detail.",
            usage="/task",
            aliases=(),
        ),
        SlashCommand(
            name="/task-output",
            description="Show task output.",
            usage="/task-output",
            aliases=("/output",),
        ),
        SlashCommand(
            name="/stop",
            description="Stop a background task.",
            usage="/stop <id>",
            aliases=("/stop-task", "/cancel", "/cancel-task"),
        ),
        SlashCommand(
            name="/abort",
            description="Abort the current runtime run.",
            usage="/abort",
            aliases=("/abort-runtime",),
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
    ]


def normalize_slash_command(text: str) -> str:
    return " ".join(text.strip().lower().split())


class CommandsDialog(ModalScreen[str | None]):
    BINDINGS = [
        Binding("escape", "close", "Close", priority=True),
        Binding("ctrl+c", "copy_text", "Copy", priority=True),
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
        self.commands = self.sort_commands(commands)
        self.filtered_commands = list(self.commands)

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
        normalized_query = query.strip().lower().lstrip("/")

        if not normalized_query:
            return True

        haystack = " ".join(
            [
                self.display_command_name(command),
                command.name,
                command.usage,
                command.description,
                " ".join(command.aliases),
                " ".join(alias.lstrip("/") for alias in command.aliases),
            ]
        ).lower()

        return normalized_query in haystack

    def render_command_row(self, command: SlashCommand) -> str:
        aliases = ""

        if command.aliases:
            aliases = "  " + ", ".join(
                alias.lstrip("/")
                for alias in command.aliases
            )

        return (
            f"{self.display_command_name(command):<16} "
            f"{command.description}{aliases}"
        )

    @staticmethod
    def display_command_name(command: SlashCommand) -> str:
        return command.name.lstrip("/")

    @classmethod
    def sort_commands(
        cls,
        commands: list[SlashCommand],
    ) -> list[SlashCommand]:
        return sorted(
            commands,
            key=lambda command: cls.display_command_name(command).lower(),
        )

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
            list_view.index = (current + 1) % len(self.filtered_commands)
            return

        if key in {"up", "ctrl+p"} and self.filtered_commands:
            event.prevent_default()
            event.stop()
            current = list_view.index or 0
            list_view.index = (current - 1) % len(self.filtered_commands)
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
        layers: base overlay;
    }

    #main-area {
        height: 1fr;
    }

    #chat-panel {
        width: 2fr;
        height: 100%;
        border: round $accent;
    }

    #side-panel {
        width: 1fr;
        height: 100%;
    }

    #side-tabs {
        height: 1;
        padding: 0 1;
        background: #1f1f1f;
        color: #aaaaaa;
    }

    .side-panel-view {
        width: 100%;
        height: 1fr;
    }

    .side-panel-view.hidden {
        display: none;
    }

    #tool-log {
        width: 100%;
        height: 1fr;
        border: round $surface;
    }

    #task-progress-panel {
        width: 100%;
        height: 1fr;
        min-height: 8;
    }

    #agent-activity-panel {
        width: 100%;
        height: 1fr;
        min-height: 8;
    }

    #team-view-panel {
        width: 100%;
        height: 1fr;
        min-height: 8;
    }

    #input-box {
        height: 12;
    }

    #slash-suggestions {
        position: absolute;
        layer: overlay;
        width: 100%;
        height: 1;
        max-height: 10;
        display: none;
        background: #1f1f1f;
        border-left: solid #5aa7ff;
        padding: 0 1;
        margin: 0 1;
    }

    #slash-suggestions.visible {
        display: block;
    }

    #status-bar {
        dock: bottom;
        height: 2;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "copy_selected_text", "Copy", priority=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", priority=True),
        Binding("ctrl+r", "reset_tokens", "Reset Tokens", priority=True),
        Binding("ctrl+s", "show_status", "Show Status", priority=True),
        Binding("ctrl+p", "show_commands", "Commands", priority=True),
        Binding("ctrl+1", "show_tool_log_panel", "Tool Log", priority=True),
        Binding("ctrl+2", "cycle_side_panel", "Next Panel", priority=True),
        Binding("ctrl+3", "show_agents_panel", "Agents", priority=True),
        Binding("ctrl+4", "cycle_side_panel_previous", "Previous Panel", priority=True),

        # Windows Terminal / 部分终端可能不传 Ctrl+数字，所以给 F1~F4 做兜底。
        Binding("f1", "show_tool_log_panel", "Tool Log", priority=True),
        Binding("f2", "show_tasks_panel", "Tasks", priority=True),
        Binding("f3", "show_agents_panel", "Agents", priority=True),
        Binding("f4", "show_team_panel", "Team", priority=True),

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
        self.active_side_panel_view = SIDE_PANEL_TOOL_LOG

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
        self.task_progress_panel: TaskProgressPanel | None = None
        self.agent_activity_panel: AgentActivityPanel | None = None
        self.team_view_panel: TeamViewPanel | None = None
        self.task_panel_refresh_busy = False
        self.agent_panel_refresh_busy = False
        self.team_panel_refresh_busy = False
        self.slash_suggestion_index = 0
        self.runtime_event_unsubscribe: Any | None = None
        self.seen_task_runtime_event_ids: set[str] = set()
        self.visible_task_creation_notice_keys: set[str] = set()

    def compose(self) -> ComposeResult:
        with Vertical(id="main-layout"):
            with Horizontal(id="main-area"):
                yield ChatPanel(id="chat-panel")
                with Vertical(id="side-panel"):
                    yield Static("", id="side-tabs")
                    yield ToolLog(
                        id="tool-log",
                        classes="side-panel-view",
                    )
                    yield TaskProgressPanel(
                        id="task-progress-panel",
                        title="Background Tasks",
                        limit=8,
                        classes="side-panel-view hidden",
                    )
                    yield AgentActivityPanel(
                        id="agent-activity-panel",
                        title="Agents",
                        active_only=False,
                        classes="side-panel-view hidden",
                    )
                    yield TeamViewPanel(
                        id="team-view-panel",
                        title="Team",
                        show_members=True,
                        show_tasks=False,
                        show_mailbox=True,
                        classes="side-panel-view hidden",
                    )

            yield Static("", id="slash-suggestions")
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
        self.task_progress_panel = self.query_one(
            "#task-progress-panel",
            TaskProgressPanel,
        )
        self.agent_activity_panel = self.query_one(
            "#agent-activity-panel",
            AgentActivityPanel,
        )
        self.team_view_panel = self.query_one("#team-view-panel", TeamViewPanel)

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

        self.apply_side_panel_view()

        self.schedule_agent_panel_refresh()

        self.set_interval(
            1.0,
            self.schedule_agent_panel_refresh,
        )

        self.set_interval(
            TASK_PANEL_FALLBACK_POLL_SECONDS,
            self.schedule_task_panel_refresh,
        )

        self.subscribe_task_runtime_events()

        with suppress(Exception):
            self.input_box.focus_input()

    def get_runtime_config(self) -> dict[str, Any]:
        runtime_config = copy.deepcopy(self.config)

        runtime_config["permissions"] = {
            **runtime_config.get("permissions", {}),
            "mode": self.permission_mode,
        }

        runtime_config["agent"] = {
            **runtime_config.get("agent", {}),
            "max_iterations": 1000,
            "max_context_messages": 5000,
        }

        user_llm_config = runtime_config.get("llm")

        if isinstance(user_llm_config, dict) and user_llm_config:
            runtime_config["llm"] = user_llm_config
        else:
            runtime_config["llm"] = get_default_tui_llm_config()

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

    def get_side_tabs_widget(self) -> Static:
        return self.query_one("#side-tabs", Static)

    def render_side_tabs(self) -> str:
        parts: list[str] = []

        for index, view in enumerate(SIDE_PANEL_VIEWS, start=1):
            label = SIDE_PANEL_LABELS[view]

            if view == self.active_side_panel_view:
                parts.append(f"[{index} {label}]")
            else:
                parts.append(f"{index} {label}")

        return "  ".join(parts)

    def apply_side_panel_view(self) -> None:
        if not self.is_mounted:
            return

        widget_by_view = {
            SIDE_PANEL_TOOL_LOG: self.get_tool_log(),
            SIDE_PANEL_TASKS: self.get_task_progress_panel(),
            SIDE_PANEL_AGENTS: self.get_agent_activity_panel(),
            SIDE_PANEL_TEAM: self.get_team_view_panel(),
        }

        for view, widget in widget_by_view.items():
            if view == self.active_side_panel_view:
                widget.remove_class("hidden")
            else:
                widget.add_class("hidden")

        self.get_side_tabs_widget().update(self.render_side_tabs())

    def set_side_panel_view(self, view: str) -> None:
        self.active_side_panel_view = normalize_side_panel_view(view)
        self.apply_side_panel_view()

        if self.status_bar is not None:
            label = SIDE_PANEL_LABELS[self.active_side_panel_view]
            self.status_bar.set_idle(f"side panel: {label}")

        if self.active_side_panel_view == SIDE_PANEL_TASKS:
            self.schedule_task_panel_refresh()
        elif self.active_side_panel_view == SIDE_PANEL_AGENTS:
            self.schedule_agent_panel_refresh()
        elif self.active_side_panel_view == SIDE_PANEL_TEAM:
            self.schedule_team_panel_refresh()

    def action_show_tool_log_panel(self) -> None:
        self.set_side_panel_view(SIDE_PANEL_TOOL_LOG)

    def action_show_tasks_panel(self) -> None:
        self.set_side_panel_view(SIDE_PANEL_TASKS)

    def action_show_agents_panel(self) -> None:
        self.set_side_panel_view(SIDE_PANEL_AGENTS)

    def action_show_team_panel(self) -> None:
        self.set_side_panel_view(SIDE_PANEL_TEAM)

    def action_cycle_side_panel(self) -> None:
        try:
            current_index = SIDE_PANEL_VIEWS.index(self.active_side_panel_view)
        except ValueError:
            current_index = 0

        next_view = SIDE_PANEL_VIEWS[
            (current_index + 1) % len(SIDE_PANEL_VIEWS)
        ]
        self.set_side_panel_view(next_view)

    def get_task_progress_panel(self) -> TaskProgressPanel:
        if self.task_progress_panel is not None:
            return self.task_progress_panel

        with suppress(Exception):
            self.task_progress_panel = self.query_one(
                "#task-progress-panel",
                TaskProgressPanel,
            )
            return self.task_progress_panel

        self.task_progress_panel = TaskProgressPanel(
            title="Background Tasks",
            limit=8,
        )
        return self.task_progress_panel

    def get_agent_activity_panel(self) -> AgentActivityPanel:
        if self.agent_activity_panel is not None:
            return self.agent_activity_panel

        with suppress(Exception):
            self.agent_activity_panel = self.query_one(
                "#agent-activity-panel",
                AgentActivityPanel,
            )
            return self.agent_activity_panel

        self.agent_activity_panel = AgentActivityPanel(
            title="Agents",
            active_only=False,
        )
        return self.agent_activity_panel

    def get_team_view_panel(self) -> TeamViewPanel:
        if self.team_view_panel is not None:
            return self.team_view_panel

        with suppress(Exception):
            self.team_view_panel = self.query_one("#team-view-panel", TeamViewPanel)
            return self.team_view_panel

        self.team_view_panel = TeamViewPanel()
        return self.team_view_panel

    def get_runtime_metadata_sources(self) -> list[dict[str, Any]]:
        controller = self.runtime_controller

        if controller is None:
            return []

        sources: list[dict[str, Any]] = []
        app_state = getattr(controller, "app_state", None)
        app_metadata = getattr(app_state, "metadata", None)

        if isinstance(app_metadata, dict):
            sources.append(app_metadata)

        engine = getattr(controller, "engine", None)
        engine_metadata = getattr(engine, "runtime_metadata", None)

        if isinstance(engine_metadata, dict):
            sources.append(engine_metadata)

        return sources

    @staticmethod
    def first_runtime_metadata_value(
        sources: list[dict[str, Any]],
        *keys: str,
    ) -> Any | None:
        for metadata in sources:
            for key in keys:
                value = metadata.get(key)

                if value is not None:
                    return value

        return None

    def resolve_runtime_task_manager(self) -> Any | None:
        """
        从当前 RuntimeController 里尽量解析共享 TaskManager。

        注意：
        这里不主动创建新的 TaskManager。
        TUI 只读取已经存在的运行时对象，避免出现“面板看的是另一个任务池”的问题。
        """
        controller = self.runtime_controller

        if controller is None:
            return None

        metadata_sources = self.get_runtime_metadata_sources()
        task_manager = self.first_runtime_metadata_value(
            metadata_sources,
            "task_manager",
        )

        if task_manager is not None:
            return task_manager

        subagent_manager = self.first_runtime_metadata_value(
            metadata_sources,
            "subagent_manager",
            "manager",
        )

        if subagent_manager is not None:
            task_manager = getattr(subagent_manager, "task_manager", None)

            if task_manager is not None:
                return task_manager

        direct = getattr(controller, "task_manager", None)

        if direct is not None:
            return direct

        app_state = getattr(controller, "app_state", None)
        app_metadata = getattr(app_state, "metadata", None)

        if isinstance(app_metadata, dict):
            task_manager = app_metadata.get("task_manager")

            if task_manager is not None:
                return task_manager

            subagent_manager = app_metadata.get("subagent_manager")

            if subagent_manager is not None:
                task_manager = getattr(subagent_manager, "task_manager", None)

                if task_manager is not None:
                    return task_manager

        engine = getattr(controller, "engine", None)

        if engine is None:
            return None

        direct = getattr(engine, "task_manager", None)

        if direct is not None:
            return direct

        graph_runner = getattr(engine, "graph_runner", None)

        if graph_runner is not None:
            direct = getattr(graph_runner, "task_manager", None)

            if direct is not None:
                return direct

        registry = getattr(engine, "registry", None)

        if registry is None:
            return None

        get_tool = getattr(registry, "get", None)

        if not callable(get_tool):
            return None

        for tool_name in (
            "agent",
            "task_create",
            "task_update",
            "task_list",
            "task_output",
            "task_stop",
        ):
            tool = get_tool(tool_name)

            if tool is None:
                continue

            manager = getattr(tool, "manager", None)

            if manager is not None:
                task_manager = getattr(manager, "task_manager", None)

                if task_manager is not None:
                    return task_manager

            fallback_runtime = getattr(tool, "_fallback_runtime", None)

            if fallback_runtime is not None:
                fallback_manager = getattr(fallback_runtime, "manager", None)

                if fallback_manager is not None:
                    task_manager = getattr(fallback_manager, "task_manager", None)

                    if task_manager is not None:
                        return task_manager

        return None

    def resolve_runtime_subagent_manager(self) -> Any | None:
        controller = self.runtime_controller

        if controller is None:
            return None

        direct = getattr(controller, "subagent_manager", None)

        if direct is not None:
            return direct

        manager = self.first_runtime_metadata_value(
            self.get_runtime_metadata_sources(),
            "subagent_manager",
            "manager",
        )

        if manager is not None:
            return manager

        engine = getattr(controller, "engine", None)

        if engine is not None:
            direct = getattr(engine, "subagent_manager", None)

            if direct is not None:
                return direct

            manager = self.first_runtime_metadata_value(
                [
                    metadata
                    for metadata in [getattr(engine, "runtime_metadata", None)]
                    if isinstance(metadata, dict)
                ],
                "subagent_manager",
                "manager",
            )

            if manager is not None:
                return manager

            registry = getattr(engine, "registry", None)
        else:
            registry = None

        if registry is None:
            return None

        get_tool = getattr(registry, "get", None)

        if not callable(get_tool):
            return None

        tool = get_tool("agent")

        if tool is None:
            return None

        manager = getattr(tool, "manager", None)

        if manager is not None:
            return manager

        fallback_runtime = getattr(tool, "_fallback_runtime", None)

        if fallback_runtime is not None:
            return getattr(fallback_runtime, "manager", None)

        return None

    def resolve_runtime_team(self) -> Any | None:
        controller = self.runtime_controller

        if controller is None:
            return None

        direct = getattr(controller, "team", None)

        if direct is not None:
            return direct

        for metadata in self.get_runtime_metadata_sources():
            team = metadata.get("team")

            if team is not None:
                return team

            for key in ("team_registry", "teams"):
                registry = metadata.get(key)

                if isinstance(registry, dict):
                    for value in registry.values():
                        if value is not None:
                            return value

        engine = getattr(controller, "engine", None)

        if engine is not None:
            team = getattr(engine, "team", None)

            if team is not None:
                return team

            metadata = getattr(engine, "runtime_metadata", None)

            if isinstance(metadata, dict):
                team = metadata.get("team")

                if team is not None:
                    return team

        return None

    def schedule_task_panel_refresh(self) -> None:
        if self.task_panel_refresh_busy:
            return

        self.run_worker(
            self.refresh_task_panel(),
            name="task-panel-refresh",
            group="ui",
            exclusive=True,
        )

    async def refresh_task_panel(self) -> None:
        self.task_panel_refresh_busy = True

        try:
            panel = self.get_task_progress_panel()
            task_manager = self.resolve_runtime_task_manager()
            subagent_manager = self.resolve_runtime_subagent_manager()

            snapshot = await build_task_snapshot_from_manager(
                task_manager,
                limit=panel.limit,
            ) if task_manager is not None else build_task_snapshot([])

            if not snapshot.rows and subagent_manager is not None:
                subagent_runs = await collect_subagent_run_records(
                    subagent_manager,
                    limit=panel.limit,
                )
                snapshot = build_task_snapshot(subagent_runs)
            elif subagent_manager is not None:
                subagent_runs = await collect_subagent_run_records(
                    subagent_manager,
                    limit=panel.limit,
                )
                subagent_snapshot = build_task_snapshot(subagent_runs)

                if subagent_snapshot.rows:
                    rows_by_id = {
                        row.task_id: row
                        for row in snapshot.rows
                    }

                    for row in subagent_snapshot.rows:
                        rows_by_id.setdefault(row.task_id, row)

                    rows = list(rows_by_id.values())
                    rows.sort(
                        key=lambda row: (
                            not row.is_active,
                            row.updated_at or row.started_at or row.created_at or 0,
                        ),
                        reverse=False,
                    )

                    if panel.limit is not None:
                        rows = rows[: panel.limit]

                    snapshot = TaskProgressSnapshot(
                        rows=rows,
                        stats=collect_stats(rows),
                    )

            panel.set_snapshot(snapshot)

        except Exception as exc:
            if self.tool_log is not None:
                self.tool_log.append_error(f"Task panel refresh failed: {exc}")

        finally:
            self.task_panel_refresh_busy = False

    def get_runtime_event_bus(self) -> Any | None:
        controller = self.runtime_controller

        if controller is None:
            return None

        event_bus = getattr(controller, "event_bus", None)

        if event_bus is not None:
            return event_bus

        engine = getattr(controller, "engine", None)

        if engine is not None:
            return getattr(engine, "event_bus", None)

        return None

    def subscribe_task_runtime_events(self) -> None:
        if self.runtime_event_unsubscribe is not None:
            with suppress(Exception):
                self.runtime_event_unsubscribe()

            self.runtime_event_unsubscribe = None

        event_bus = self.get_runtime_event_bus()
        subscribe = getattr(event_bus, "subscribe", None)

        if not callable(subscribe):
            return

        self.runtime_event_unsubscribe = subscribe(
            self.handle_task_runtime_event,
        )

    def on_unmount(self) -> None:
        if self.runtime_event_unsubscribe is not None:
            with suppress(Exception):
                self.runtime_event_unsubscribe()

            self.runtime_event_unsubscribe = None

    def is_task_runtime_event(self, event: RuntimeEvent) -> bool:
        status = str(getattr(event, "status", "") or "")
        metadata = getattr(event, "metadata", {}) or {}

        if status in TASK_RUNTIME_EVENT_STATUSES:
            return True

        if metadata.get("task_event") is True:
            return True

        if metadata.get("category") == "task":
            return True

        return False

    def handle_task_runtime_event(self, event: RuntimeEvent) -> None:
        if not self.is_task_runtime_event(event):
            return

        event_id = str(getattr(event, "event_id", "") or "")

        if event_id and event_id in self.seen_task_runtime_event_ids:
            return

        if event_id:
            self.seen_task_runtime_event_ids.add(event_id)

            if len(self.seen_task_runtime_event_ids) > 500:
                self.seen_task_runtime_event_ids = set(
                    list(self.seen_task_runtime_event_ids)[-250:]
                )

        self.schedule_task_panel_refresh()

        if self.active_side_panel_view == SIDE_PANEL_TASKS and self.status_bar is not None:
            task_status = str(getattr(event, "status", "") or "task event")
            self.status_bar.set_idle(task_status)

    def schedule_agent_panel_refresh(self) -> None:
        if self.agent_panel_refresh_busy:
            return

        self.run_worker(
            self.refresh_agent_panel(),
            name="agent-panel-refresh",
            group="agent-panel",
            exclusive=True,
        )

    async def refresh_agent_panel(self) -> None:
        self.agent_panel_refresh_busy = True

        try:
            panel = self.get_agent_activity_panel()
            manager = self.resolve_runtime_subagent_manager()
            team = self.resolve_runtime_team()

            await panel.refresh_from_sources(
                manager=manager,
                team=team,
            )

        except Exception as exc:
            if self.tool_log is not None:
                self.tool_log.append_error(f"Agent panel refresh failed: {exc}")

        finally:
            self.agent_panel_refresh_busy = False

    def schedule_team_panel_refresh(self) -> None:
        if self.team_panel_refresh_busy:
            return

        self.run_worker(
            self.refresh_team_panel(),
            name="team-panel-refresh",
            group="team-panel",
            exclusive=True,
        )

    async def refresh_team_panel(self) -> None:
        self.team_panel_refresh_busy = True

        try:
            panel = self.get_team_view_panel()
            team = self.resolve_runtime_team()

            await panel.refresh_from_team(team)

        except Exception as exc:
            if self.tool_log is not None:
                self.tool_log.append_error(f"Team panel refresh failed: {exc}")

        finally:
            self.team_panel_refresh_busy = False

    def get_slash_suggestions_widget(self) -> Static:
        return self.query_one("#slash-suggestions", Static)

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

        return normalize_tui_permission_mode(mode)

    def set_permission_mode(self, mode: str) -> None:
        mode = normalize_tui_permission_mode(mode)

        self.permission_mode = mode
        self.config.setdefault("permissions", {})["mode"] = mode

        if self.status_bar is not None:
            self.status_bar.set_permission_mode(mode)

        if not self.runtime_busy:
            self.runtime_controller = self.create_runtime_controller()

    def get_slash_commands(self) -> list[SlashCommand]:
        return get_builtin_slash_commands()

    def get_matching_slash_commands(
        self,
        text: str,
        *,
        limit: int = 9,
    ) -> list[SlashCommand]:
        raw = text.strip()

        if not raw.startswith("/"):
            return []

        query = raw[1:].lower()

        if " " in query:
            query = query.split(maxsplit=1)[0]

        matches: list[SlashCommand] = []

        for command in self.get_slash_commands():
            names = [
                command.name.lstrip("/"),
                command.usage.lstrip("/"),
                *[alias.lstrip("/") for alias in command.aliases],
            ]
            haystack = " ".join([*names, command.description]).lower()

            if not query or query in haystack:
                matches.append(command)

            if len(matches) >= limit:
                break

        return matches

    def render_slash_suggestions(self, matches: list[SlashCommand]) -> Text:
        rendered = Text()

        for index, command in enumerate(matches):
            style = "black on #f4b183" if index == self.slash_suggestion_index else ""
            rendered.append(f"{command.name:<14} {command.description}", style=style)

            if index < len(matches) - 1:
                rendered.append("\n")

        return rendered

    def hide_slash_suggestions(self) -> None:
        if not self.is_mounted:
            return

        with suppress(Exception):
            suggestions = self.get_slash_suggestions_widget()
            suggestions.update("")
            suggestions.remove_class("visible")

        self.slash_suggestion_index = 0

    def is_slash_suggestions_visible(self) -> bool:
        if not self.is_mounted:
            return False

        with suppress(Exception):
            return self.get_slash_suggestions_widget().has_class("visible")

        return False

    def position_slash_suggestions(self, match_count: int) -> None:
        suggestions = self.get_slash_suggestions_widget()
        input_box = self.get_input_box()
        height = max(1, min(match_count, 9))
        y = max(0, input_box.region.y - height)

        suggestions.styles.height = height
        suggestions.styles.offset = (0, y)

    def update_slash_suggestions(self, text: str) -> None:
        if not self.is_mounted:
            return

        matches = self.get_matching_slash_commands(text)

        if not matches:
            self.hide_slash_suggestions()
            return

        self.slash_suggestion_index = min(
            self.slash_suggestion_index,
            len(matches) - 1,
        )

        with suppress(Exception):
            suggestions = self.get_slash_suggestions_widget()
            self.position_slash_suggestions(len(matches))
            suggestions.update(self.render_slash_suggestions(matches))
            suggestions.add_class("visible")

    @on(TextArea.Changed, "#prompt-input")
    def on_prompt_input_changed(self, event: TextArea.Changed) -> None:
        self.update_slash_suggestions(str(event.text_area.text))

    def execute_selected_slash_suggestion(self, text: str) -> bool:
        if not self.is_mounted:
            return False

        matches = self.get_matching_slash_commands(text)

        if not matches:
            return False

        index = min(self.slash_suggestion_index, len(matches) - 1)
        command = matches[index]

        self.hide_slash_suggestions()
        self.handle_slash_command(command.usage)
        return True

    def is_known_slash_command_text(self, text: str) -> bool:
        command_text = normalize_slash_command(text)

        if not command_text.startswith("/"):
            return False

        command_name = command_text.split(maxsplit=1)[0]

        for command in self.get_slash_commands():
            values = {
                normalize_slash_command(command.name),
                normalize_slash_command(command.usage),
                *[
                    normalize_slash_command(alias)
                    for alias in command.aliases
                ],
            }

            if command_text in values or command_name in values:
                return True

        return False

    def get_slash_command_name(self, text: str) -> str:
        command_text = normalize_slash_command(text)

        if not command_text.startswith("/"):
            return ""

        return command_text.split(maxsplit=1)[0]

    def is_slash_command_allowed_while_busy(self, text: str) -> bool:
        command_name = self.get_slash_command_name(text)

        if not command_name:
            return False

        return command_name in RUNTIME_BUSY_ALLOWED_SLASH_COMMANDS

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
                "- `Ctrl+2`: next side panel",
                "- `Ctrl+4`: previous side panel",
                "- `Tab`: switch permission mode",
                "- `Esc`: clear input",
                "- `Ctrl+L`: clear chat and tool log",
                "- `Ctrl+R`: reset tokens",
                "- `Ctrl+S`: show status",
            ]
        )

        return "\n".join(lines)

    def render_task_detail_text(
        self,
        task: Any,
        *,
        output_only: bool = False,
    ) -> str:
        data = object_to_display_dict(task)

        task_id = (
            data.get("id")
            or data.get("task_id")
            or data.get("run_id")
            or safe_get_attr(task, "id", "")
            or safe_get_attr(task, "task_id", "")
            or safe_get_attr(task, "run_id", "")
        )
        name = (
            data.get("name")
            or data.get("title")
            or data.get("task")
            or data.get("description")
            or safe_get_attr(task, "name", "")
            or safe_get_attr(task, "title", "")
            or safe_get_attr(task, "task", "")
        )
        status = enum_or_text(
            data.get("status")
            or safe_get_attr(task, "status", "")
        )
        agent = (
            data.get("agent_id")
            or data.get("agent_name")
            or data.get("agent")
            or safe_get_attr(task, "agent_id", "")
            or safe_get_attr(task, "agent_name", "")
            or safe_get_attr(task, "agent", "")
        )
        result = data.get("result") or safe_get_attr(task, "result", None)
        error = data.get("error") or safe_get_attr(task, "error", None)

        friendly_title = friendly_task_title(
            {
                "name": name,
                "agent_name": agent,
                "agent_id": agent,
                "status": status,
            }
        )
        friendly_agent = friendly_agent_label(
            {
                "agent_name": agent,
                "agent_id": agent,
            }
        )

        if output_only:
            return (
                f"Task output `{task_id}`:\n\n"
                f"- Name: `{name or '-'}`\n"
                f"- Status: `{status or '-'}`\n"
                f"- Agent: `{agent or '-'}`\n\n"
                f"Result:\n{compact_value(result) or '-'}\n\n"
                f"Error:\n{compact_value(error) or '-'}"
            )

        lines = [
            friendly_title or f"Task `{task_id}`",
            "",
            f"- Status: `{status or '-'}`",
            f"- Agent: `{friendly_agent or '-'}`",
            f"- Name: `{name or '-'}`",
            f"- Task ID: `{task_id or '-'}`",
        ]

        for key in (
            "created_at",
            "started_at",
            "updated_at",
            "finished_at",
        ):
            value = data.get(key) or safe_get_attr(task, key, None)

            if value is not None:
                lines.append(f"- {key}: `{value}`")

        if result is not None:
            lines.extend(
                [
                    "",
                    "Result:",
                    compact_value(result) or "-",
                ]
            )

        if error:
            lines.extend(
                [
                    "",
                    "Error:",
                    compact_value(error),
                ]
            )

        metadata = data.get("metadata") or safe_get_attr(task, "metadata", None)

        if metadata:
            lines.extend(
                [
                    "",
                    "Metadata:",
                    compact_value(metadata),
                ]
            )

        return "\n".join(lines)

    async def find_task_by_id(self, task_id: str) -> Any | None:
        task_manager = self.resolve_runtime_task_manager()

        if task_manager is not None:
            get_task = getattr(task_manager, "get_task", None)

            if callable(get_task):
                try:
                    return await maybe_await(get_task(task_id))
                except Exception:
                    pass

            list_tasks = getattr(task_manager, "list_tasks", None)

            if callable(list_tasks):
                try:
                    tasks = await maybe_await(list_tasks())
                except TypeError:
                    tasks = await maybe_await(list_tasks(limit=None))
                except Exception:
                    tasks = []

                for task in tasks or []:
                    current_id = (
                        safe_get_attr(task, "id", None)
                        or safe_get_attr(task, "task_id", None)
                        or safe_get_attr(task, "run_id", None)
                    )

                    if str(current_id) == task_id:
                        return task

        subagent_manager = self.resolve_runtime_subagent_manager()

        if subagent_manager is not None:
            records = await collect_subagent_run_records(
                subagent_manager,
                limit=None,
            )

            for record in records:
                current_id = (
                    safe_get_attr(record, "id", None)
                    or safe_get_attr(record, "task_id", None)
                    or safe_get_attr(record, "run_id", None)
                )

                if str(current_id) == task_id:
                    return record

        return None

    async def run_tasks_command(self) -> None:
        task_manager = self.resolve_runtime_task_manager()
        subagent_manager = self.resolve_runtime_subagent_manager()

        if task_manager is None and subagent_manager is None:
            self.get_chat_panel().append_system_message(
                "No shared TaskManager is available yet."
            )
            self.get_status_bar().set_idle("tasks unavailable")
            self.get_input_box().focus_input()
            return

        snapshot = (
            await build_task_snapshot_from_manager(
                task_manager,
                limit=20,
            )
            if task_manager is not None
            else build_task_snapshot([])
        )

        if subagent_manager is not None:
            subagent_snapshot = build_task_snapshot(
                await collect_subagent_run_records(
                    subagent_manager,
                    limit=20,
                )
            )

            if subagent_snapshot.rows:
                rows_by_id = {
                    row.task_id: row
                    for row in snapshot.rows
                }

                for row in subagent_snapshot.rows:
                    rows_by_id.setdefault(row.task_id, row)

                rows = list(rows_by_id.values())
                rows.sort(
                    key=lambda row: (
                        not row.is_active,
                        row.updated_at or row.started_at or row.created_at or 0,
                    ),
                    reverse=False,
                )
                rows = rows[:20]

                snapshot = TaskProgressSnapshot(
                    rows=rows,
                    stats=collect_stats(rows),
                )

        self.get_task_progress_panel().set_snapshot(snapshot)

        stats = snapshot.stats

        lines = [
            "Background tasks:",
            "",
            f"- Total: `{stats.total}`",
            f"- Active: `{stats.active}`",
            f"- Running: `{stats.running}`",
            f"- Succeeded: `{stats.succeeded}`",
            f"- Failed: `{stats.failed}`",
            f"- Cancelled: `{stats.cancelled}`",
            "",
        ]

        if not snapshot.rows:
            lines.append("No background tasks.")
        else:
            for row in snapshot.rows:
                lines.append(
                    f"- `{row.task_id}` · `{row.status}` · "
                    f"{row.name or '-'} · agent `{row.agent or '-'}`"
                )

        self.get_chat_panel().append_system_message("\n".join(lines))
        self.get_status_bar().set_idle("tasks shown")
        self.get_input_box().focus_input()

    async def run_agents_command(self) -> None:
        manager = self.resolve_runtime_subagent_manager()

        if manager is None:
            self.get_chat_panel().append_system_message(
                "No shared SubAgentManager is available yet."
            )
            self.get_status_bar().set_idle("agents unavailable")
            self.get_input_box().focus_input()
            return

        active_runs = []

        get_active_runs = getattr(manager, "get_active_runs", None)

        if callable(get_active_runs):
            active_runs = list(get_active_runs() or [])

        history = []

        get_history = getattr(manager, "get_history", None)

        if callable(get_history):
            try:
                history = list(get_history(limit=5) or [])
            except TypeError:
                history = list(get_history() or [])[-5:]

        lines = [
            "SubAgents:",
            "",
            f"- Active runs: `{len(active_runs)}`",
            f"- Recent history: `{len(history)}`",
            "",
        ]

        if active_runs:
            lines.append("Active:")
            for run in active_runs:
                data = object_to_display_dict(run)
                run_id = data.get("run_id") or safe_get_attr(run, "run_id", "-")
                name = data.get("name") or data.get("agent_name") or safe_get_attr(run, "name", "-")
                status = enum_or_text(data.get("status") or safe_get_attr(run, "status", ""))
                task = data.get("task") or safe_get_attr(run, "task", "")

                lines.append(
                    f"- `{run_id}` · `{name}` · `{status or '-'}` · {task or '-'}"
                )
        else:
            lines.append("No active SubAgent runs.")

        if history:
            lines.extend(
                [
                    "",
                    "Recent:",
                ]
            )

            for run in history:
                data = object_to_display_dict(run)
                run_id = data.get("run_id") or safe_get_attr(run, "run_id", "-")
                name = data.get("name") or data.get("agent_name") or safe_get_attr(run, "name", "-")
                status = enum_or_text(data.get("status") or safe_get_attr(run, "status", ""))
                lines.append(f"- `{run_id}` · `{name}` · `{status or '-'}`")

        self.get_chat_panel().append_system_message("\n".join(lines))
        self.get_status_bar().set_idle("agents shown")
        self.get_input_box().focus_input()

    async def run_team_command(self) -> None:
        team = self.resolve_runtime_team()

        if team is None:
            self.get_chat_panel().append_system_message(
                "No runtime Team is available yet."
            )
            self.get_status_bar().set_idle("team unavailable")
            self.get_input_box().focus_input()
            return

        snapshot = await build_team_snapshot(team)

        await self.get_team_view_panel().refresh_from_team(team)

        stats = snapshot.stats
        mailbox = stats.mailbox

        lines = [
            f"Team `{snapshot.name or snapshot.team_id}`",
            "",
            f"- Team ID: `{snapshot.team_id or '-'}`",
            f"- Members: `{stats.members_total}`",
            f"- Active members: `{stats.members_active}`",
            f"- Shared tasks: `{stats.tasks_total}`",
            f"- Active tasks: `{stats.tasks_active}`",
            "",
            "Mailbox:",
            f"- Total: `{mailbox.total}`",
            f"- Unread: `{mailbox.unread}`",
            f"- Read: `{mailbox.read}`",
            f"- Acked: `{mailbox.acked}`",
            f"- Task messages: `{mailbox.task_messages}`",
            f"- Result messages: `{mailbox.result_messages}`",
            f"- Error messages: `{mailbox.error_messages}`",
        ]

        self.get_chat_panel().append_system_message("\n".join(lines))
        self.get_status_bar().set_idle("team shown")
        self.get_input_box().focus_input()

    async def run_task_detail_command(
        self,
        task_id: str,
        *,
        output_only: bool = False,
    ) -> None:
        if not task_id:
            command = "/task-output" if output_only else "/task"
            self.get_chat_panel().append_error_message(
                f"Usage: `{command} <id>`"
            )
            self.get_status_bar().set_error("missing task id")
            self.get_input_box().focus_input()
            return

        task = await self.find_task_by_id(task_id)

        if task is None:
            self.get_chat_panel().append_error_message(
                f"Task not found: `{task_id}`"
            )
            self.get_status_bar().set_error("task not found")
            self.get_input_box().focus_input()
            return

        self.get_chat_panel().append_system_message(
            self.render_task_detail_text(
                task,
                output_only=output_only,
            )
        )
        self.get_status_bar().set_idle("task shown")
        self.get_input_box().focus_input()

    async def run_stop_task_command(self, task_id: str) -> None:
        if not task_id:
            self.get_chat_panel().append_error_message("Usage: `/stop <id>`")
            self.get_status_bar().set_error("missing task id")
            self.get_input_box().focus_input()
            return

        task_manager = self.resolve_runtime_task_manager()
        subagent_manager = self.resolve_runtime_subagent_manager()

        cancelled = None

        if subagent_manager is not None:
            cancel_agent_task = getattr(subagent_manager, "cancel_agent_task", None)

            if callable(cancel_agent_task):
                try:
                    cancelled = await maybe_await(
                        cancel_agent_task(
                            task_id,
                            reason="cancelled from TUI /stop",
                            wait=False,
                        )
                    )
                except Exception:
                    cancelled = None

        if cancelled is None and task_manager is not None:
            cancel_task = getattr(task_manager, "cancel_task", None)

            if callable(cancel_task):
                try:
                    cancelled = await maybe_await(
                        cancel_task(
                            task_id,
                            reason="cancelled from TUI /stop",
                            wait=False,
                        )
                    )
                except TypeError:
                    cancelled = await maybe_await(cancel_task(task_id))

        if cancelled is None:
            self.get_chat_panel().append_error_message(
                f"Could not stop task `{task_id}`. No compatible cancel API was found."
            )
            self.get_status_bar().set_error("stop failed")
            self.get_input_box().focus_input()
            return

        await self.get_task_progress_panel().refresh_from_task_manager(
            self.resolve_runtime_task_manager()
        )

        self.get_chat_panel().append_system_message(
            f"Stop requested for task `{task_id}`."
        )
        self.get_status_bar().set_idle("stop requested")
        self.get_input_box().focus_input()

    async def run_retry_task_command(self, task_id: str) -> None:
        if not task_id:
            self.get_chat_panel().append_error_message("Usage: retry requires a task id.")
            self.get_status_bar().set_error("missing task id")
            return

        task = await self.find_task_by_id(task_id)

        if task is None:
            self.get_chat_panel().append_error_message(f"Task not found: `{task_id}`")
            self.get_status_bar().set_error("task not found")
            return

        task_manager = self.resolve_runtime_task_manager()
        subagent_manager = self.resolve_runtime_subagent_manager()

        for owner in (subagent_manager, task_manager):
            if owner is None:
                continue

            for method_name in (
                "retry_agent_task",
                "retry_task",
                "restart_task",
                "rerun_task",
            ):
                method = getattr(owner, method_name, None)

                if callable(method):
                    try:
                        await maybe_await(method(task_id))
                        self.get_chat_panel().append_system_message(
                            f"Retry requested for task `{task_id}`."
                        )
                        self.get_status_bar().set_idle("retry requested")
                        await self.get_task_progress_panel().refresh_from_task_manager(
                            self.resolve_runtime_task_manager()
                        )
                        return
                    except Exception:
                        pass

        if subagent_manager is not None:
            agent_name = (
                safe_get_attr(task, "agent_id", None)
                or safe_get_attr(task, "agent_name", None)
                or safe_get_attr(task, "agent", None)
                or "general"
            )
            task_text = (
                safe_get_attr(task, "task", None)
                or safe_get_attr(task, "name", None)
                or safe_get_attr(task, "title", None)
                or safe_get_attr(task, "description", None)
            )

            if task_text:
                request = SubAgentTaskRequest(
                    agent_name=str(agent_name),
                    task=str(task_text),
                    metadata={
                        "source": "tui_retry",
                        "retry_of": task_id,
                    },
                )

                await subagent_manager.run_agent_task(
                    request,
                    wait=False,
                )

                self.get_chat_panel().append_system_message(
                    f"Created retry task for `{task_id}`."
                )
                self.get_status_bar().set_idle("retry created")
                await self.get_task_progress_panel().refresh_from_task_manager(
                    self.resolve_runtime_task_manager()
                )
                return

        self.get_chat_panel().append_error_message(
            f"Could not retry task `{task_id}`. No compatible retry API or task payload found."
        )
        self.get_status_bar().set_error("retry failed")

    async def run_abort_agent_command(self, agent_id: str) -> None:
        manager = self.resolve_runtime_subagent_manager()

        if manager is None:
            self.get_chat_panel().append_error_message("No SubAgentManager is available.")
            self.get_status_bar().set_error("agent abort unavailable")
            return

        aborted = False

        for method_name in (
            "abort_agent",
            "abort_run",
            "stop_agent",
            "cancel_agent",
        ):
            method = getattr(manager, method_name, None)

            if callable(method):
                try:
                    await maybe_await(
                        method(
                            agent_id,
                            reason="aborted from TUI Agent panel",
                        )
                    )
                    aborted = True
                    break
                except TypeError:
                    await maybe_await(method(agent_id))
                    aborted = True
                    break
                except Exception:
                    pass

        if not aborted:
            abort_all = getattr(manager, "abort_all", None)

            if callable(abort_all):
                try:
                    await maybe_await(
                        abort_all(
                            reason=f"agent {agent_id} aborted from TUI",
                        )
                    )
                    aborted = True
                except Exception:
                    aborted = False

        if aborted:
            self.get_chat_panel().append_system_message(
                f"Abort requested for agent `{agent_id}`."
            )
            self.get_status_bar().set_idle("agent abort requested")
        else:
            self.get_chat_panel().append_error_message(
                f"Could not abort agent `{agent_id}`. No compatible abort API found."
            )
            self.get_status_bar().set_error("agent abort failed")

    async def run_agent_history_command(self, agent_id: str) -> None:
        manager = self.resolve_runtime_subagent_manager()

        if manager is None:
            self.get_chat_panel().append_error_message("No SubAgentManager is available.")
            self.get_status_bar().set_error("agent history unavailable")
            return

        history = []
        get_history = getattr(manager, "get_history", None)

        if callable(get_history):
            try:
                history = list(get_history(limit=20) or [])
            except TypeError:
                history = list(get_history() or [])

        rows = []

        for item in history:
            data = object_to_display_dict(item)
            current_agent = (
                data.get("agent_id")
                or data.get("agent_name")
                or safe_get_attr(item, "agent_id", None)
                or safe_get_attr(item, "agent_name", None)
                or safe_get_attr(item, "name", None)
            )

            if str(current_agent) != str(agent_id):
                continue

            rows.append(item)

        lines = [
            f"History for agent `{agent_id}`:",
            "",
        ]

        if not rows:
            lines.append("No history found.")
        else:
            for item in rows[:20]:
                data = object_to_display_dict(item)
                run_id = data.get("run_id") or safe_get_attr(item, "run_id", "-")
                status = data.get("status") or safe_get_attr(item, "status", "-")
                task = data.get("task") or safe_get_attr(item, "task", "")
                lines.append(f"- `{run_id}` · `{status}` · {task or '-'}")

        self.get_chat_panel().append_system_message("\n".join(lines))
        self.get_status_bar().set_idle("agent history shown")

    async def run_team_mailbox_command(self) -> None:
        team = self.resolve_runtime_team()

        if team is None:
            self.get_chat_panel().append_error_message("No runtime Team is available.")
            self.get_status_bar().set_error("team unavailable")
            return

        mailbox = getattr(team, "mailbox", None)

        if mailbox is None:
            self.get_chat_panel().append_error_message("Current team has no mailbox.")
            self.get_status_bar().set_error("mailbox unavailable")
            return

        list_messages = getattr(mailbox, "list_messages", None)

        if not callable(list_messages):
            self.get_chat_panel().append_error_message("Mailbox does not support list_messages().")
            self.get_status_bar().set_error("mailbox unsupported")
            return

        try:
            messages = await maybe_await(list_messages(include_deleted=False))
        except TypeError:
            messages = await maybe_await(list_messages())

        lines = [
            f"Mailbox for team `{getattr(team, 'team_id', '-')}`:",
            "",
        ]

        if not messages:
            lines.append("No messages.")
        else:
            for message in list(messages)[-20:]:
                data = object_to_display_dict(message)
                message_id = data.get("message_id") or safe_get_attr(message, "message_id", "-")
                sender = data.get("sender_id") or safe_get_attr(message, "sender_id", "-")
                recipient = data.get("recipient_id") or safe_get_attr(message, "recipient_id", "-")
                status = data.get("status") or safe_get_attr(message, "status", "-")
                content = data.get("content") or safe_get_attr(message, "content", "")
                lines.append(
                    f"- `{message_id}` · {sender} → {recipient} · `{status}` · {str(content)[:80]}"
                )

        self.get_chat_panel().append_system_message("\n".join(lines))
        self.get_status_bar().set_idle("mailbox shown")

    async def run_team_dispatch_command(self) -> None:
        team = self.resolve_runtime_team()

        if team is None:
            self.get_chat_panel().append_error_message("No runtime Team is available.")
            self.get_status_bar().set_error("team unavailable")
            return

        dispatch_next_task = getattr(team, "dispatch_next_task", None)

        if not callable(dispatch_next_task):
            self.get_chat_panel().append_error_message(
                "Current team does not support dispatch_next_task()."
            )
            self.get_status_bar().set_error("dispatch unsupported")
            return

        message = await maybe_await(dispatch_next_task())

        if message is None:
            self.get_chat_panel().append_system_message("No pending team task to dispatch.")
            self.get_status_bar().set_idle("nothing to dispatch")
            return

        self.get_chat_panel().append_system_message(
            "Team task dispatched.\n\n"
            f"- Message: `{getattr(message, 'message_id', '-')}`\n"
            f"- Recipient: `{getattr(message, 'recipient_id', '-')}`"
        )
        self.get_status_bar().set_idle("team task dispatched")
        await self.get_team_view_panel().refresh_from_team(team)

    def run_slash_command_worker(
        self,
        awaitable: Any,
        *,
        name: str,
    ) -> None:
        self.run_worker(
            awaitable,
            name=name,
            group="slash-command",
            exclusive=False,
        )

    def handle_slash_command(self, user_text: str) -> bool:
        raw_text = user_text.strip()

        if not raw_text:
            return False

        parts = raw_text.split(maxsplit=1)
        command_name = parts[0].lower()
        command_args = parts[1].strip() if len(parts) > 1 else ""

        if command_name == "/tool":
            return False

        command_text = normalize_slash_command(raw_text)

        if not command_text.startswith("/"):
            return False

        if command_name in {"/help", "/?", "/h"}:
            self.get_chat_panel().append_system_message(self.render_help_text())
            self.get_status_bar().set_idle("help shown")
            self.get_input_box().focus_input()
            return True

        if command_name in {"/clear", "/cls"}:
            self.action_clear_chat()
            return True

        if command_name in {"/status", "/info"}:
            self.action_show_status()
            return True

        if command_name in {"/tool-log", "/log", "/tools"}:
            self.set_side_panel_view(SIDE_PANEL_TOOL_LOG)
            self.get_input_box().focus_input()
            return True

        if command_name in {"/copy-log", "/copy-tool-log"}:
            self.action_copy_tool_log()
            return True

        if command_name in {"/tasks", "/task-list", "/jobs"}:
            self.set_side_panel_view(SIDE_PANEL_TASKS)
            self.run_slash_command_worker(
                self.run_tasks_command(),
                name="slash-tasks",
            )
            return True

        if command_name in {"/agents", "/agent", "/runs"}:
            self.set_side_panel_view(SIDE_PANEL_AGENTS)
            self.run_slash_command_worker(
                self.run_agents_command(),
                name="slash-agents",
            )
            return True

        if command_name in {"/team", "/mailbox"}:
            self.set_side_panel_view(SIDE_PANEL_TEAM)
            self.run_slash_command_worker(
                self.run_team_command(),
                name="slash-team",
            )
            return True

        if command_name in {"/panel", "/side"}:
            if not command_args:
                self.get_chat_panel().append_system_message(
                    "Usage: `/panel <log|tasks|agents|team>`"
                )
                self.get_status_bar().set_idle("panel help")
                self.get_input_box().focus_input()
                return True

            self.set_side_panel_view(command_args)
            self.get_input_box().focus_input()
            return True

        if command_name == "/task":
            self.run_slash_command_worker(
                self.run_task_detail_command(command_args),
                name="slash-task-detail",
            )
            return True

        if command_name in {"/task-output", "/output"}:
            self.run_slash_command_worker(
                self.run_task_detail_command(
                    command_args,
                    output_only=True,
                ),
                name="slash-task-output",
            )
            return True

        if command_name in {"/stop", "/stop-task", "/cancel", "/cancel-task"}:
            self.run_slash_command_worker(
                self.run_stop_task_command(command_args),
                name="slash-stop-task",
            )
            return True

        if command_name in {"/abort", "/abort-runtime"}:
            self.action_abort_runtime()
            return True

        if command_name in {"/doctor", "/diag"}:
            self.action_show_doctor()
            return True

        if command_text == "/tokens":
            self.action_show_tokens()
            return True

        if command_text in {"/reset-token", "/reset-tokens", "/tokens reset"}:
            self.action_reset_tokens()
            return True

        if command_name in {"/exit", "/quit"}:
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

        if user_text.startswith("/"):
            if self.runtime_busy and not self.is_slash_command_allowed_while_busy(user_text):
                command_name = self.get_slash_command_name(user_text)

                if self.chat_panel is not None:
                    self.chat_panel.append_system_message(
                        "A runtime task is still running.\n\n"
                        f"Command `{command_name}` is not available while the runtime is busy.\n\n"
                        "Allowed while running:\n"
                        "- `/status`\n"
                        "- `/tasks`\n"
                        "- `/agents`\n"
                        "- `/team`\n"
                        "- `/task <id>`\n"
                        "- `/task-output <id>`\n"
                        "- `/stop <id>`\n"
                        "- `/abort`"
                    )

                if self.status_bar is not None:
                    self.status_bar.set_idle("runtime busy")

                return

            if (
                not self.is_known_slash_command_text(user_text)
                and self.execute_selected_slash_suggestion(user_text)
            ):
                return

        self.hide_slash_suggestions()

        if user_text.startswith("/"):
            handled = self.handle_slash_command(user_text)

            if handled:
                return

        if self.runtime_busy:
            if self.chat_panel is not None:
                self.chat_panel.append_system_message(
                    "A runtime task is still running. Please wait for it to finish."
                )
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
                        self.chat_panel.append_assistant_message(
                            result.output,
                            metadata={
                                "model": self.get_configured_model_label(),
                            },
                        )

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

    def extract_task_creation_notice_from_event(
        self,
        event: RuntimeEvent,
    ) -> dict[str, Any] | None:
        tool_result = getattr(event, "tool_result", None)

        if tool_result is not None:
            tool_name = str(getattr(tool_result, "tool_name", "") or "")
            success = bool(getattr(tool_result, "success", False))
            data = getattr(tool_result, "data", {}) or {}

            if (
                tool_name == "task_create"
                and success
                and isinstance(data, dict)
            ):
                notice = data.get("ui_notice")

                if isinstance(notice, dict):
                    return dict(notice)

        metadata = getattr(event, "metadata", {}) or {}
        status = str(getattr(event, "status", "") or "")

        if (
            status == "task_created"
            or metadata.get("task_event_type") == "created"
        ) and metadata.get("task_event") is True:
            task_id = (
                metadata.get("task_id")
                or metadata.get("id")
                or "-"
            )
            event_data = metadata.get("task_event_data")

            if not isinstance(event_data, dict):
                event_data = {}

            return {
                "kind": "background_task_created",
                "target": "task_manager",
                "task_id": str(task_id),
                "agent": str(
                    event_data.get("agent_id")
                    or event_data.get("agent_name")
                    or "-"
                ),
                "name": str(
                    event_data.get("name")
                    or event_data.get("title")
                    or event_data.get("message")
                    or "Task"
                ),
                "status": str(
                    metadata.get("task_status")
                    or event_data.get("status")
                    or "created"
                ),
                "started": False,
                "message": "已创建后台任务，可在右侧 Tasks 面板查看进度。",
            }

        return None

    def render_task_creation_notice_for_chat(
        self,
        notice: dict[str, Any],
    ) -> str:
        headline = (
            "Started background task:"
            if notice.get("started")
            else "Created background task:"
        )

        return (
            f"{headline}\n"
            f"- id: `{notice.get('task_id', '-')}`\n"
            f"- agent: `{notice.get('agent', '-')}`\n"
            f"- name: {notice.get('name', '-')}\n"
            f"- status: `{notice.get('status', '-')}`\n\n"
            f"{notice.get('message', '已创建后台任务，可在右侧 Tasks 面板查看进度。')}"
        )

    def handle_task_creation_feedback(
        self,
        event: RuntimeEvent,
    ) -> None:
        notice = self.extract_task_creation_notice_from_event(event)

        if notice is None:
            return

        task_id = str(notice.get("task_id") or "")
        notice_key = f"task-created:{task_id or getattr(event, 'event_id', '')}"

        if notice_key in self.visible_task_creation_notice_keys:
            return

        self.visible_task_creation_notice_keys.add(notice_key)

        if len(self.visible_task_creation_notice_keys) > 500:
            self.visible_task_creation_notice_keys = set(
                list(self.visible_task_creation_notice_keys)[-250:]
            )

        if self.chat_panel is not None:
            self.chat_panel.append_system_message(
                self.render_task_creation_notice_for_chat(notice)
            )

        self.schedule_task_panel_refresh()

        if self.active_side_panel_view != SIDE_PANEL_TASKS:
            self.set_side_panel_view(SIDE_PANEL_TASKS)

    def handle_runtime_event(self, event: RuntimeEvent) -> None:
        if self.tool_log is not None:
            self.tool_log.append_runtime_event(event)

        self.handle_task_creation_feedback(event)

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

    def get_selected_text_for_copy(self) -> str:
        screen = self.screen

        if hasattr(screen, "get_selected_text"):
            selected_text = screen.get_selected_text()

            if selected_text:
                return selected_text

        focused = self.focused
        selected_text = getattr(focused, "selected_text", "")

        if isinstance(selected_text, str) and selected_text:
            return selected_text

        return ""

    def action_copy_selected_text(self) -> None:
        selected_text = self.get_selected_text_for_copy()

        if not selected_text:
            self.get_status_bar().set_idle("nothing selected")
            return

        self.copy_to_clipboard(selected_text)
        self.get_status_bar().set_idle("copied selection")

    def action_copy_tool_log(self) -> None:
        tool_log = self.get_tool_log()
        text = tool_log.to_text().strip()

        if not text:
            self.get_status_bar().set_idle("tool log empty")
            self.get_input_box().focus_input()
            return

        self.copy_to_clipboard(text)
        self.get_status_bar().set_idle("tool log copied")
        self.get_input_box().focus_input()

    def on_key(self, event: Any) -> None:
        key = str(getattr(event, "key", "") or "")
        key_lower = key.lower()

        side_panel_keys = {
            "ctrl+1": SIDE_PANEL_TOOL_LOG,
            "ctrl+2": SIDE_PANEL_TASKS,
            "ctrl+3": SIDE_PANEL_AGENTS,
            "ctrl+4": SIDE_PANEL_TEAM,
            "ctrl+space": SIDE_PANEL_TASKS,
            "ctrl+@": SIDE_PANEL_TASKS,
            "ctrl+[": SIDE_PANEL_AGENTS,
            "ctrl+left_square_bracket": SIDE_PANEL_AGENTS,
            "ctrl+\\": SIDE_PANEL_TEAM,
            "ctrl+backslash": SIDE_PANEL_TEAM,
            "f1": SIDE_PANEL_TOOL_LOG,
            "f2": SIDE_PANEL_TASKS,
            "f3": SIDE_PANEL_AGENTS,
            "f4": SIDE_PANEL_TEAM,
        }

        if key_lower == "ctrl+2":
            self.action_cycle_side_panel()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

            return

        if key_lower == "ctrl+4":
            self.action_cycle_side_panel_previous()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

            return

        if key_lower in side_panel_keys:
            self.set_side_panel_view(side_panel_keys[key_lower])

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

            return

        if key_lower in {"up", "down"} and self.is_mounted:
            matches = self.get_matching_slash_commands(self.get_input_box().get_text())

            if matches:
                if hasattr(event, "prevent_default"):
                    event.prevent_default()

                if hasattr(event, "stop"):
                    event.stop()

                delta = -1 if key_lower == "up" else 1
                self.slash_suggestion_index = (
                    self.slash_suggestion_index + delta
                ) % len(matches)
                self.get_slash_suggestions_widget().update(
                    self.render_slash_suggestions(matches)
                )
                return

        if key_lower == "ctrl+r":
            self.action_reset_tokens()

            if hasattr(event, "stop"):
                event.stop()

        if key_lower == "tab":
            self.action_cycle_permission_mode()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

        if key_lower == "ctrl+p":
            self.action_show_commands()

            if hasattr(event, "prevent_default"):
                event.prevent_default()

            if hasattr(event, "stop"):
                event.stop()

            return

        if key_lower == "c" and self.active_side_panel_view == SIDE_PANEL_TOOL_LOG:
            self.action_copy_tool_log()

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

    @on(TaskProgressPanel.TaskDetailRequested)
    def on_task_detail_requested(
        self,
        event: TaskProgressPanel.TaskDetailRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_task_detail_command(event.task_id),
            name="panel-task-detail",
        )

    @on(TaskProgressPanel.TaskOutputRequested)
    def on_task_output_requested(
        self,
        event: TaskProgressPanel.TaskOutputRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_task_detail_command(
                event.task_id,
                output_only=True,
            ),
            name="panel-task-output",
        )

    @on(TaskProgressPanel.TaskStopRequested)
    def on_task_stop_requested(
        self,
        event: TaskProgressPanel.TaskStopRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_stop_task_command(event.task_id),
            name="panel-task-stop",
        )

    @on(TaskProgressPanel.TaskRetryRequested)
    def on_task_retry_requested(
        self,
        event: TaskProgressPanel.TaskRetryRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_retry_task_command(event.task_id),
            name="panel-task-retry",
        )

    @on(TaskProgressPanel.TaskCopyRequested)
    def on_task_copy_requested(
        self,
        event: TaskProgressPanel.TaskCopyRequested,
    ) -> None:
        self.copy_to_clipboard(event.task_id)
        self.get_status_bar().set_idle("task id copied")

    @on(AgentActivityPanel.AgentInspectRequested)
    def on_agent_inspect_requested(
        self,
        event: AgentActivityPanel.AgentInspectRequested,
    ) -> None:
        row = event.row

        self.get_chat_panel().append_system_message(
            "Agent detail:\n\n"
            f"- Display: `{friendly_agent_label(row)}`\n"
            f"- Activity: {friendly_agent_activity(row)}\n"
            f"- Role: `{role_label(row.role) or '-'}`\n"
            f"- Status: `{row.status}`\n"
            f"- Agent ID: `{row.agent_id}`\n"
            f"- Run ID: `{row.current_run_id or '-'}`\n"
            f"- Task record ID: `{row.current_task_record_id or '-'}`\n"
            f"- Current task: {row.current_task or '-'}\n"
            f"- Error: {row.error or '-'}"
        )

        self.get_status_bar().set_idle("agent shown")

    @on(AgentActivityPanel.AgentAbortRequested)
    def on_agent_abort_requested(
        self,
        event: AgentActivityPanel.AgentAbortRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_abort_agent_command(event.agent_id),
            name="panel-agent-abort",
        )

    @on(AgentActivityPanel.AgentHistoryRequested)
    def on_agent_history_requested(
        self,
        event: AgentActivityPanel.AgentHistoryRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_agent_history_command(event.agent_id),
            name="panel-agent-history",
        )

    @on(TeamViewPanel.TeamMailboxRequested)
    def on_team_mailbox_requested(
        self,
        event: TeamViewPanel.TeamMailboxRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_team_mailbox_command(),
            name="panel-team-mailbox",
        )

    @on(TeamViewPanel.TeamDispatchRequested)
    def on_team_dispatch_requested(
        self,
        event: TeamViewPanel.TeamDispatchRequested,
    ) -> None:
        self.run_slash_command_worker(
            self.run_team_dispatch_command(),
            name="panel-team-dispatch",
        )

    @on(TeamViewPanel.TeamMessageMemberRequested)
    def on_team_message_member_requested(
        self,
        event: TeamViewPanel.TeamMessageMemberRequested,
    ) -> None:
        template = (
            '/tool send_message '
            '{"action":"send",'
            f'"recipient_id":"{event.teammate_id}",'
            '"content":""}'
        )

        input_box = self.get_input_box()
        set_text = getattr(input_box, "set_text", None)

        if callable(set_text):
            set_text(template)
        else:
            self.get_chat_panel().append_system_message(
                "Message template:\n\n"
                f"`{template}`"
            )

        input_box.focus_input()
        self.get_status_bar().set_idle("message template ready")

    def action_abort_runtime(self) -> None:
        if self.runtime_controller is None:
            self.get_chat_panel().append_error_message(
                "RuntimeController is not initialized."
            )
            self.get_status_bar().set_error("abort unavailable")
            self.get_input_box().focus_input()
            return

        abort = getattr(self.runtime_controller, "abort", None)

        if not callable(abort):
            self.get_chat_panel().append_error_message(
                "Current RuntimeController does not support abort()."
            )
            self.get_status_bar().set_error("abort unsupported")
            self.get_input_box().focus_input()
            return

        abort()

        aborted_subagent_runs = 0
        manager = self.resolve_runtime_subagent_manager()

        if manager is not None:
            abort_all = getattr(manager, "abort_all", None)

            if callable(abort_all):
                try:
                    aborted_subagent_runs = int(
                        abort_all(
                            reason="aborted from TUI /abort",
                        )
                        or 0
                    )
                except Exception:
                    aborted_subagent_runs = 0

        self.get_chat_panel().append_system_message(
            "Abort requested for the current runtime run.\n\n"
            f"- SubAgent runs signalled: `{aborted_subagent_runs}`"
        )
        self.get_status_bar().set_idle("abort requested")
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
