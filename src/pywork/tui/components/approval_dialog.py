from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, ClassVar

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Static

from pywork.permission.audit import (
    PermissionAuditUserAction,
    sanitize_arguments,
)
from pywork.permission.policy import (
    PermissionDecision,
    PermissionDecisionType,
)
from pywork.permission.risk import RiskLevel


DEFAULT_ARGUMENT_MAX_CHARS = 4_000

FILE_CHANGE_TOOL_NAMES: set[str] = {
    "file_write",
    "file_edit",
}


class ApprovalChoice(str, Enum):
    """审批弹窗按钮选择。"""

    ALLOW = "allow"
    DENY = "deny"
    ALWAYS_ALLOW = "always_allow"


@dataclass(slots=True, frozen=True)
class ApprovalDialogResult:
    """审批弹窗返回结果。"""

    choice: ApprovalChoice
    user_action: PermissionAuditUserAction
    allowed: bool
    always_allow: bool
    decision: PermissionDecision

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["choice"] = self.choice.value
        data["user_action"] = self.user_action.value
        data["decision"] = self.decision.to_dict()
        return data


def normalize_approval_choice(choice: ApprovalChoice | str) -> ApprovalChoice:
    if isinstance(choice, ApprovalChoice):
        return choice

    return ApprovalChoice(str(choice))


def choice_to_user_action(choice: ApprovalChoice | str) -> PermissionAuditUserAction:
    normalized = normalize_approval_choice(choice)

    if normalized == ApprovalChoice.ALLOW:
        return PermissionAuditUserAction.ALLOW

    if normalized == ApprovalChoice.DENY:
        return PermissionAuditUserAction.DENY

    if normalized == ApprovalChoice.ALWAYS_ALLOW:
        return PermissionAuditUserAction.ALWAYS_ALLOW

    raise ValueError(f"unknown approval choice: {choice!r}")


def approval_result_from_choice(
    decision: PermissionDecision,
    choice: ApprovalChoice | str,
) -> ApprovalDialogResult:
    normalized = normalize_approval_choice(choice)
    user_action = choice_to_user_action(normalized)

    return ApprovalDialogResult(
        choice=normalized,
        user_action=user_action,
        allowed=normalized in {
            ApprovalChoice.ALLOW,
            ApprovalChoice.ALWAYS_ALLOW,
        },
        always_allow=normalized == ApprovalChoice.ALWAYS_ALLOW,
        decision=decision,
    )


def enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def normalize_tool_name(tool_name: str | Any) -> str:
    return str(tool_name).strip().lower().replace("-", "_")


def get_decision_tool_name(decision: PermissionDecision) -> str:
    request = getattr(decision, "request", None)

    if request is None:
        return ""

    return str(getattr(request, "tool_name", ""))


def get_decision_arguments(decision: PermissionDecision) -> dict[str, Any]:
    request = getattr(decision, "request", None)

    if request is None:
        return {}

    arguments = getattr(request, "arguments", {})

    if isinstance(arguments, dict):
        return dict(arguments)

    try:
        return dict(arguments)
    except Exception:
        return {
            "arguments": str(arguments),
        }


def is_file_change_decision(decision: PermissionDecision) -> bool:
    return normalize_tool_name(get_decision_tool_name(decision)) in FILE_CHANGE_TOOL_NAMES


def risk_style(risk: RiskLevel | str) -> str:
    value = getattr(risk, "value", risk)
    text = str(value).lower()

    if text == "critical":
        return "bold red"

    if text == "high":
        return "bold yellow"

    if text == "medium":
        return "yellow"

    if text == "low":
        return "green"

    return "dim"


def decision_style(decision_type: PermissionDecisionType | str) -> str:
    value = getattr(decision_type, "value", decision_type)
    text = str(value).lower()

    if text == "ask_elevated":
        return "bold red"

    if text == "ask":
        return "bold yellow"

    if text == "deny":
        return "bold red"

    if text == "allow":
        return "green"

    return "white"


def truncate_text(
    text: str,
    *,
    max_chars: int = DEFAULT_ARGUMENT_MAX_CHARS,
) -> str:
    if len(text) <= max_chars:
        return text

    omitted = len(text) - max_chars

    return text[:max_chars] + f"\n... [truncated {omitted} chars]"


def format_arguments_json(
    arguments: dict[str, Any],
    *,
    max_chars: int = DEFAULT_ARGUMENT_MAX_CHARS,
) -> str:
    sanitized = sanitize_arguments(arguments)

    text = json.dumps(
        sanitized,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    return truncate_text(
        text,
        max_chars=max_chars,
    )


def build_decision_table(decision: PermissionDecision) -> Table:
    request = getattr(decision, "request", None)

    table = Table.grid(
        padding=(0, 1),
    )

    table.add_column(
        justify="right",
        style="bold cyan",
        no_wrap=True,
    )
    table.add_column(
        style="white",
        overflow="fold",
    )

    tool_name = getattr(request, "tool_name", "") if request is not None else ""
    action = getattr(request, "action", None) if request is not None else None
    call_id = getattr(request, "call_id", None) if request is not None else None

    table.add_row("Tool", str(tool_name))

    if action:
        table.add_row("Action", str(action))

    table.add_row("Mode", enum_value(decision.mode))
    table.add_row("Risk", Text(enum_value(decision.risk), style=risk_style(decision.risk)))
    table.add_row(
        "Decision",
        Text(enum_value(decision.decision), style=decision_style(decision.decision)),
    )

    if call_id:
        table.add_row("Call ID", str(call_id))

    table.add_row("Reason", str(decision.reason))

    return table


def build_arguments_renderable(
    decision: PermissionDecision,
    *,
    max_chars: int = DEFAULT_ARGUMENT_MAX_CHARS,
    max_argument_chars: int | None = None,
) -> RenderableType:
    arguments = get_decision_arguments(decision)

    if max_argument_chars is not None:
        max_chars = max_argument_chars

    if not arguments:
        return Text("No arguments.", style="dim")

    text = format_arguments_json(
        arguments,
        max_chars=max_chars,
    )

    return Syntax(
        text,
        "json",
        word_wrap=True,
        indent_guides=True,
    )


def build_warning_text(
    decision: PermissionDecision,
    *,
    is_file_preview: bool = False,
) -> Text:
    if decision.decision == PermissionDecisionType.ASK_ELEVATED:
        return Text(
            "High-risk operation. Review carefully before allowing.",
            style="bold red",
        )

    if is_file_preview:
        return Text(
            "Review the diff preview before accepting. Reject will leave the file unchanged.",
            style="bold yellow",
        )

    if decision.decision == PermissionDecisionType.ASK:
        return Text(
            "This operation needs your approval before it can run.",
            style="bold yellow",
        )

    if decision.decision == PermissionDecisionType.DENY:
        return Text(
            "This operation was denied by policy.",
            style="bold red",
        )

    return Text(
        "This operation is allowed by policy.",
        style="green",
    )


def render_diff_preview_text(diff_text: str) -> RenderableType:
    return Syntax(
        diff_text,
        "diff",
        word_wrap=False,
        theme="ansi_dark",
    )


def build_approval_summary(
    decision: PermissionDecision,
    *,
    max_argument_chars: int = DEFAULT_ARGUMENT_MAX_CHARS,
) -> RenderableType:
    """构造审批弹窗主体内容。"""
    warning = build_warning_text(
        decision,
        is_file_preview=is_file_change_decision(decision),
    )

    table = build_decision_table(decision)

    arguments_panel = Panel(
        build_arguments_renderable(
            decision,
            max_chars=max_argument_chars,
        ),
        title="Arguments",
        border_style="dim",
    )

    return Group(
        warning,
        "",
        table,
        "",
        arguments_panel,
    )


class ApprovalDialog(ModalScreen[ApprovalDialogResult]):
    """
    权限审批弹窗。

    返回：
        ApprovalDialogResult

    按钮：
        Allow
        Deny
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("escape", "deny", "Deny"),
        ("a", "allow", "Allow"),
        ("d", "deny", "Deny"),
    ]

    DEFAULT_CSS = """
    ApprovalDialog {
        align: center middle;
    }

    ApprovalDialog > Container {
        width: 80%;
        max-width: 110;
        height: auto;
        max-height: 88%;
        border: round #666666;
        background: #181818;
        padding: 1 2;
    }

    ApprovalDialog .dialog-title {
        text-style: bold;
        color: #eeeeee;
        margin-bottom: 1;
    }

    ApprovalDialog .dialog-body {
        height: 1fr;
        min-height: 12;
        max-height: 28;
        margin-bottom: 1;
    }

    ApprovalDialog .dialog-buttons {
        height: auto;
        align: right middle;
    }

    ApprovalDialog Button {
        width: 18;
        height: 3;
        margin-left: 1;
        background: #303030;
        color: #f2f2f2;
        border: tall #777777;
        text-style: bold;
    }

    ApprovalDialog Button:focus {
        background: #444444;
        color: #ffffff;
        border: tall #d0d0d0;
    }

    ApprovalDialog Button:hover {
        background: #3a3a3a;
        color: #ffffff;
    }
    """

    def __init__(
        self,
        decision: PermissionDecision,
        *,
        title: str | None = "Approve Operation",
        show_always_allow: bool = False,
        max_argument_chars: int = DEFAULT_ARGUMENT_MAX_CHARS,
        diff_text: str | None = None,
        preview_renderable: RenderableType | None = None,
    ) -> None:
        super().__init__()
        self.decision = decision
        self.show_always_allow = False
        self.max_argument_chars = max_argument_chars
        self.diff_text = diff_text
        self.preview_renderable = preview_renderable
        self.is_file_change = is_file_change_decision(decision)
        self.has_diff_preview = bool(diff_text) or preview_renderable is not None

        if title is not None:
            self.title_text = title
        elif self.decision.decision == PermissionDecisionType.ASK_ELEVATED:
            self.title_text = "Approve Elevated Operation"
        elif self.is_file_change:
            self.title_text = "Review File Change"
        else:
            self.title_text = "Approve Operation"

    @property
    def allow_label(self) -> str:
        return "Accept" if self.is_file_change else "Allow"

    @property
    def deny_label(self) -> str:
        return "Reject" if self.is_file_change else "Deny"
    
    def compose(self) -> ComposeResult:
        with Container():
            yield Static(
                self._render_title(),
                classes="dialog-title",
            )

            with VerticalScroll(classes="dialog-body"):
                yield Static(
                    build_approval_summary(
                        self.decision,
                        max_argument_chars=self.max_argument_chars,
                    )
                )

                if self.has_diff_preview:
                    yield Static("Diff preview", classes="dialog-title")

                    if self.preview_renderable is not None:
                        yield Static(self.preview_renderable)
                    elif self.diff_text:
                        yield self._create_diff_panel_or_fallback()

            with Horizontal(classes="dialog-buttons"):
                yield Button(
                    self.deny_label,
                    id="approval-deny",
                )
                yield Button(
                    self.allow_label,
                    id="approval-allow",
                )

    def _render_title(self) -> Text:
        title = Text(self.title_text, style="bold")

        if self.decision.decision == PermissionDecisionType.ASK_ELEVATED:
            title.append("  ")
            title.append("[Elevated]", style="bold red")
        elif self.decision.decision == PermissionDecisionType.ASK:
            title.append("  ")
            title.append("[Approval Required]", style="bold yellow")

        if self.is_file_change:
            title.append("  ")
            title.append("[File Change]", style="cyan")

        return title

    def _create_diff_panel_or_fallback(self) -> Static:
        diff_text = self.diff_text or ""

        try:
            from pywork.tui.components.diff.widgets import DiffPanel

            panel = DiffPanel()
            panel.set_diff(diff_text)
            return panel
        except Exception:
            return Static(render_diff_preview_text(diff_text))

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        button_id = event.button.id

        if button_id == "approval-allow":
            self.dismiss(
                approval_result_from_choice(
                    self.decision,
                    ApprovalChoice.ALLOW,
                )
            )
            return

        if button_id == "approval-deny":
            self.dismiss(
                approval_result_from_choice(
                    self.decision,
                    ApprovalChoice.DENY,
                )
            )
            return

    def action_allow(self) -> None:
        self.dismiss(
            approval_result_from_choice(
                self.decision,
                ApprovalChoice.ALLOW,
            )
        )

    def action_deny(self) -> None:
        self.dismiss(
            approval_result_from_choice(
                self.decision,
                ApprovalChoice.DENY,
            )
        )

def demo_decision(
    *,
    file_change: bool = True,
) -> PermissionDecision:
    from pywork.permission.policy import evaluate_permission

    if file_change:
        return evaluate_permission(
            "file_edit",
            mode="default",
            arguments={
                "path": "src/utils/helper.py",
                "old_string": "old",
                "new_string": "new",
            },
            call_id="demo_file_edit",
        )

    return evaluate_permission(
        "powershell",
        mode="default",
        arguments={
            "command": "Remove-Item demo.txt",
            "cwd": ".",
        },
        call_id="demo_call_1",
    )


DEMO_DIFF = """--- a/src/utils/helper.py
+++ b/src/utils/helper.py
@@ -1,3 +1,4 @@
 def helper():
-    return "old"
+    value = "new"
+    return value
"""


def demo() -> None:
    from textual.app import App

    class ApprovalDialogDemoApp(App[None]):
        CSS = """
        Screen {
            align: center middle;
            background: #111111;
        }

        #open {
            width: 32;
            height: 3;
            background: #303030;
            color: #f2f2f2;
            border: tall #777777;
            text-style: bold;
        }

        #open:focus {
            background: #444444;
            color: #ffffff;
            border: tall #d0d0d0;
        }

        #open:hover {
            background: #3a3a3a;
            color: #ffffff;
        }
        """

        def compose(self) -> ComposeResult:
            yield Button("Open Approval Dialog", id="open")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id != "open":
                return

            self.push_screen(
                ApprovalDialog(
                    demo_decision(file_change=True),
                    title="Demo Approval",
                    diff_text=DEMO_DIFF,
                ),
                self.on_approval_result,
            )

        def on_approval_result(
            self,
            result: ApprovalDialogResult | None,
        ) -> None:
            if result is None:
                self.exit()
                return

            self.notify(
                f"choice={result.choice.value}, allowed={result.allowed}",
            )

    ApprovalDialogDemoApp().run()


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
