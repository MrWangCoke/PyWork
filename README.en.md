# PyWork

Languages: [中文](README.md) | **English**

PyWork is a Python TUI coding-agent workspace. It brings together a terminal UI, tool calling, permission approval, file-change previews, runtime orchestration, and LLM integration for local development workflows.

> This project is still in early development. Many modules already have structure and tests, but behavior is evolving quickly.

## Features

- Textual-based TUI with chat panel, input box, status bar, and tool log.
- RuntimeController / RuntimeEngine coordinate user requests and run results.
- AgentGraphRunner uses a LangGraph-style flow for LLM and tool execution.
- Supports OpenAI-compatible providers; the current TUI config targets Qwen / DashScope.
- Built-in file, search, and shell tools such as `file_read`, `glob`, `grep`, `file_write`, `file_edit`, `bash`, and `powershell`.
- PermissionGate checks risky operations such as file writes, edits, and shell commands before execution.
- ApprovalDialog is used for tool calls that require user confirmation.
- Includes foundations for diffs, file-change previews, permission audit, and session overrides.

## Requirements

- Python `>=3.12,<3.14`
- `uv`
- Current development is mainly verified on Windows PowerShell

If you use the current Qwen / DashScope TUI configuration, set:

```powershell
$env:DASHSCOPE_API_KEY = "your-api-key"
```

## Install

Run commands from the project root:

```powershell
cd E:\MrWang\Desktop\pywork
uv sync
```

If `uv sync` reports a TOML parse error, `pyproject.toml` may have been edited incorrectly. It should start with something like:

```toml
[project]
```

Do not paste shell commands like `uv sync` into `pyproject.toml`.

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

- `q`: quit
- `Ctrl+C`: quit
- `Ctrl+L`: clear chat
- `Ctrl+R`: reset token counters
- `Ctrl+S`: show current status

The input box supports multi-line input. Submission is handled by the `InputBox` component; common shortcuts during development are `Ctrl+Enter` or `Ctrl+J`.

## Development Commands

Import smoke check:

```powershell
uv run python scripts/check_imports.py
```

If you are already inside the `scripts` directory:

```powershell
uv run python check_imports.py
```

Run all tests:

```powershell
uv run pytest
```

Run focused tests:

```powershell
uv run pytest tests\test_tui_app.py -q
uv run pytest tests\test_runtime_graph_tools.py -q
uv run pytest tests\test_permission_gate.py -q
uv run pytest tests\test_file_write_edit_tools.py -q
```

Compile check:

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
  tools/                Built-in tools and tool registry
  permission/           Permission policy, risk checks, audit, file/shell checks
  schemas/              message, tool, and config schemas
  state/                app/session/UI state
  storage/              persistence and history helpers
  utils/                shared utilities

tests/                  tests
config/                 default TOML configuration
scripts/                developer helper scripts
```

## Runtime Flow

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
-> ApprovalDialog
-> Tool execution
-> Tool result
-> final response or next graph step
```

For risky operations such as file writes, edits, and shell commands, the runtime checks permissions before execution. Normal file changes may require approval; sensitive paths, protected files, or dangerous commands may be denied or require elevated confirmation.

## Current Status

PyWork is currently pre-alpha. Active work focuses on the runtime, TUI, tool calls, permission approval, file read/write/edit flows, LLM provider integration, and test coverage. Tests are the best reference for current expected behavior.

