# PyWork

PyWork is a Python TUI coding-agent workspace. It is built around a Textual
interface, a RuntimeController / RuntimeEngine execution loop, LangGraph-style
agent orchestration, tool calling, and a permission gate for risky operations.

The project is currently in early development. The core skeleton, TUI demo
flow, runtime graph, tool registry, file tools, shell tools, approval dialog,
and test coverage are being built incrementally.

## Features

- Textual-based TUI with chat panel, input box, status bar, and tool log.
- RuntimeController and RuntimeEngine for executing user requests.
- LangGraph-backed AgentGraphRunner for multi-step tool workflows.
- OpenAI-compatible LLM provider routing.
- Built-in tools:
  - `file_read`
  - `glob`
  - `grep`
  - `file_write`
  - `file_edit`
  - `bash`
  - `powershell`
  - `echo`
- PermissionGate for file and shell operations.
- Approval dialog for operations that require user confirmation.
- File change preview and diff-related utilities.
- Audit and session override plumbing for permission decisions.
- Focused tests for runtime, permissions, tools, TUI components, and LLM message
  conversion.

## Requirements

- Python `>=3.12,<3.14`
- `uv`
- Windows PowerShell is supported during current development.

Optional environment variables depend on the LLM provider you use. The current
TUI runtime configuration uses DashScope / Qwen-compatible OpenAI API settings:

```powershell
$env:DASHSCOPE_API_KEY = "your-api-key"
```

## Install

Run commands from the project root:

```powershell
cd E:\MrWang\Desktop\pywork
uv sync
```

If `uv sync` reports a TOML parse error, check that `pyproject.toml` starts with
a valid TOML table such as:

```toml
[project]
```

Do not paste shell commands into `pyproject.toml`.

## Run

Start PyWork for the current workspace:

```powershell
uv run pywork .
```

Equivalent module entry:

```powershell
uv run python -m pywork.entrypoints.cli .
```

Run startup checks without launching the TUI:

```powershell
uv run pywork . --no-tui
```

Print startup information as JSON:

```powershell
uv run pywork . --json
```

Run diagnostics:

```powershell
uv run pywork doctor .
```

Initialize PyWork files for a workspace:

```powershell
uv run pywork init .
```

## TUI Shortcuts

Inside the TUI:

- `q`: quit
- `Ctrl+C`: quit
- `Ctrl+L`: clear chat
- `Ctrl+R`: reset token counters
- `Ctrl+S`: show current status

The input box supports multi-line input. Use the configured submit shortcut in
the input component, such as `Ctrl+Enter` or `Ctrl+J`, to submit.

## Useful Development Commands

Run the import smoke check from the project root:

```powershell
uv run python scripts/check_imports.py
```

If you are already inside the `scripts` directory, run:

```powershell
uv run python check_imports.py
```

Run all tests:

```powershell
uv run pytest
```

Run focused test groups:

```powershell
uv run pytest tests\test_tui_app.py -q
uv run pytest tests\test_runtime_graph_tools.py -q
uv run pytest tests\test_permission_gate.py -q
uv run pytest tests\test_file_write_edit_tools.py -q
```

Compile-check selected modules:

```powershell
uv run python -m compileall src\pywork
```

## Project Layout

```text
src/pywork/
  entrypoints/          CLI startup, init, doctor
  tui/                  Textual app and TUI components
  runtime/              Runtime engine, graph, controller, events
  llm/                  Provider routing and message conversion
  tools/                Built-in tool implementations and registry
  permission/           Permission policy, risk, audit, file/shell checks
  schemas/              Message, tool, and config schemas
  state/                App/session/UI state
  storage/              Persistence and history helpers
  utils/                Shared utilities

tests/                  Unit and integration-style tests
config/                 Default TOML configuration files
scripts/                Developer helper scripts
```

## Runtime Flow

At a high level:

```text
User input
-> TUI InputBox
-> PyWorkApp
-> RuntimeController
-> RuntimeEngine
-> AgentGraphRunner
-> LLM / deterministic route
-> Tool call
-> PermissionGate
-> ApprovalDialog when needed
-> Tool execution
-> Tool result
-> final assistant response or next graph step
```

For risky file and shell operations, the runtime checks permission rules before
execution. Normal file writes and edits can require approval. Sensitive paths,
protected project files, or dangerous shell commands can be denied or require
elevated confirmation.

## Current Status

PyWork is pre-alpha. The architecture is actively evolving, and some modules are
scaffolds for future functionality. The most actively developed areas are:

- TUI component integration
- Runtime graph behavior
- tool result handling
- file read / write / edit flows
- approval and permission gate behavior
- OpenAI-compatible LLM provider support

Use the tests as the source of truth for currently expected behavior.

