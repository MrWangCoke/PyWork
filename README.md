# PyWork

语言 / Languages: **中文** | [English](README.en.md)

PyWork 是一个 Python TUI 编程代理工作区，目标是把终端聊天界面、工具调用、权限审批、文件修改预览、Runtime 执行流和 LLM 连接整合到一个可扩展的本地开发助手里。

> 当前项目仍处于早期开发阶段。很多模块已经有骨架和测试，但整体行为仍在快速迭代。

## 功能概览

- 基于 Textual 的 TUI，包含聊天区、输入框、状态栏和工具日志。
- RuntimeController / RuntimeEngine 负责调度用户输入和运行结果。
- AgentGraphRunner 使用 LangGraph 风格的流程组织 LLM 与工具调用。
- 支持 OpenAI-compatible LLM provider，目前 TUI 默认配置为 Qwen / DashScope。
- 内置文件、搜索和命令工具，例如 `file_read`、`glob`、`grep`、`file_write`、`file_edit`、`bash`、`powershell`。
- PermissionGate 会在文件写入、文件编辑、shell 命令等高风险操作前进行检查。
- ApprovalDialog 用于需要用户确认的工具调用。
- 包含 diff、文件变更预览、权限审计、session override 等基础能力。

## 环境要求

- Python `>=3.12,<3.14`
- `uv`
- 当前开发环境主要在 Windows PowerShell 下验证

如果使用当前 TUI 里的 Qwen / DashScope 配置，需要设置：

```powershell
$env:DASHSCOPE_API_KEY = "your-api-key"
```

## 安装

请在项目根目录运行命令：

```powershell
cd E:\MrWang\Desktop\pywork
uv sync
```

如果 `uv sync` 报 TOML 解析错误，通常是 `pyproject.toml` 内容被误改了。文件开头应该类似：

```toml
[project]
```

不要把 `uv sync` 这类命令写进 `pyproject.toml`。

## 运行

启动当前工作区：

```powershell
uv run pywork .
```

等价的模块运行方式：

```powershell
uv run python -m pywork.entrypoints.cli .
```

只做启动检查，不打开 TUI：

```powershell
uv run pywork . --no-tui
```

输出 JSON 启动信息：

```powershell
uv run pywork . --json
```

运行环境诊断：

```powershell
uv run pywork doctor .
```

初始化 PyWork 工作区文件：

```powershell
uv run pywork init .
```

## TUI 快捷键

- `q`：退出
- `Ctrl+C`：退出
- `Ctrl+L`：清空聊天
- `Ctrl+R`：重置 token 计数
- `Ctrl+S`：显示当前状态

输入框支持多行输入。提交快捷键由 `InputBox` 组件处理，当前开发中常用 `Ctrl+Enter` 或 `Ctrl+J`。

## 常用开发命令

导入检查：

```powershell
uv run python scripts/check_imports.py
```

如果你已经在 `scripts` 目录里：

```powershell
uv run python check_imports.py
```

运行全部测试：

```powershell
uv run pytest
```

运行部分测试：

```powershell
uv run pytest tests\test_tui_app.py -q
uv run pytest tests\test_runtime_graph_tools.py -q
uv run pytest tests\test_permission_gate.py -q
uv run pytest tests\test_file_write_edit_tools.py -q
```

编译检查：

```powershell
uv run python -m compileall src\pywork
```

## 项目结构

```text
src/pywork/
  entrypoints/          CLI 启动、init、doctor
  tui/                  Textual App 和 TUI 组件
  runtime/              Runtime engine、graph、controller、events
  llm/                  Provider 路由和消息转换
  tools/                内置工具和工具注册表
  permission/           权限策略、风险判断、审计、文件/命令检查
  schemas/              message、tool、config schema
  state/                app/session/UI state
  storage/              持久化和历史记录
  utils/                通用工具函数

tests/                  测试
config/                 默认 TOML 配置
scripts/                开发辅助脚本
```

## Runtime 流程

```text
用户输入
-> TUI InputBox
-> PyWorkApp
-> RuntimeController
-> RuntimeEngine
-> AgentGraphRunner
-> LLM / 确定性路由
-> Tool call
-> PermissionGate
-> ApprovalDialog
-> Tool execution
-> Tool result
-> 最终回复或下一轮 graph step
```

对于文件写入、文件编辑、shell 命令等高风险操作，Runtime 会先执行权限检查。普通文件修改可能需要确认；敏感路径、保护文件或危险命令可能被拒绝或要求 elevated confirmation。

## 当前状态

PyWork 目前是 Pre-Alpha。项目的重点仍在 Runtime、TUI、工具调用、权限审批、文件读写编辑、LLM provider 连接和测试覆盖上。请以测试用例作为当前行为的主要参考。

