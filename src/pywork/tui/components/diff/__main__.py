from __future__ import annotations

from textual.app import App, ComposeResult
from textual.widgets import Static

from pywork.tui.components.diff import DiffPanel


def make_basic_diff() -> str:
    return (
        "--- a/README.md\n"
        "+++ b/README.md\n"
        "@@ -1,7 +1,8 @@\n"
        " # PyWork\n"
        " \n"
        "-A Python TUI agent workspace inspired by mature coding-agent architecture.\n"
        "+A Python TUI coding-agent workspace inspired by mature agent architecture.\n"
        " \n"
        " ## Status\n"
        " \n"
        "-Project skeleton initialized.\n"
        "+Project runtime, tools, and TUI initialized.\n"
        "+Diff subcomponents added.\n"
    )


def make_new_file_diff() -> str:
    return (
        "--- /dev/null\n"
        "+++ b/src/pywork/demo.py\n"
        "@@ -0,0 +1,5 @@\n"
        "+from __future__ import annotations\n"
        "+\n"
        "+def hello() -> str:\n"
        "+    return \"hello PyWork\"\n"
        "+\n"
    )


class DiffComponentsDemoApp(App[None]):
    """Diff 子组件 demo。"""

    CSS = """
    Screen {
        layout: vertical;
    }

    #demo-help {
        height: 1;
        text-style: bold;
    }

    .diff-panel-title {
        height: 1;
        padding: 0 1;
        text-style: bold;
        color: $accent;
    }

    DiffPanel {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("1", "show_basic", "Basic diff"),
        ("2", "show_new_file", "New file"),
        ("3", "show_empty", "Empty"),
        ("q", "quit", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static(
            "Diff components demo: 1 basic, 2 new file, 3 empty, q quit",
            id="demo-help",
        )

        yield DiffPanel(
            make_basic_diff(),
            title="Basic diff",
            id="diff-panel",
        )

    def action_show_basic(self) -> None:
        panel = self.query_one("#diff-panel", DiffPanel)
        panel.set_diff(
            make_basic_diff(),
            title="Basic diff",
        )

    def action_show_new_file(self) -> None:
        panel = self.query_one("#diff-panel", DiffPanel)
        panel.set_diff(
            make_new_file_diff(),
            title="New file diff",
        )

    def action_show_empty(self) -> None:
        panel = self.query_one("#diff-panel", DiffPanel)
        panel.set_diff(
            "",
            title="Empty diff",
        )


def main() -> int:
    DiffComponentsDemoApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())