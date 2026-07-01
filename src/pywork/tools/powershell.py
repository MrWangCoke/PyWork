from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

from pywork.schemas.tool_schema import ToolCall, ToolResult, ToolRiskLevel
from pywork.tools.file_read import get_context_workspace_path
from pywork.tools.tool import (
    BaseTool,
    ToolExecutionContext,
    ToolValidationError,
)
from pywork.utils.shell import (
    DEFAULT_MAX_OUTPUT_CHARS,
    DEFAULT_TIMEOUT_SECONDS,
    ShellRunOptions,
    build_powershell_command,
    render_shell_result,
    run_shell_command,
    which,
)


DEFAULT_POWERSHELL_TIMEOUT_SECONDS = DEFAULT_TIMEOUT_SECONDS
DEFAULT_POWERSHELL_MAX_OUTPUT_CHARS = DEFAULT_MAX_OUTPUT_CHARS


def coerce_bool_arg(
    value: Any,
    *,
    default: bool,
) -> bool:
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in {"true", "1", "yes", "y", "on"}:
            return True

        if normalized in {"false", "0", "no", "n", "off"}:
            return False

    return bool(value)


def coerce_float_arg(
    value: Any,
    *,
    name: str,
    default: float,
    minimum: float | None = None,
) -> float:
    if value is None:
        result = default
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{name} must be a number") from exc

    if minimum is not None and result < minimum:
        raise ToolValidationError(f"{name} must be >= {minimum}")

    return result


def coerce_int_arg(
    value: Any,
    *,
    name: str,
    default: int,
    minimum: int | None = None,
) -> int:
    if value is None:
        result = default
    else:
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{name} must be an integer") from exc

    if minimum is not None and result < minimum:
        raise ToolValidationError(f"{name} must be >= {minimum}")

    return result


def normalize_env_arg(value: Any) -> dict[str, str | None] | None:
    if value is None:
        return None

    if not isinstance(value, Mapping):
        raise ToolValidationError("env must be an object")

    env: dict[str, str | None] = {}

    for key, item in value.items():
        key_text = str(key).strip()

        if not key_text:
            raise ToolValidationError("env keys cannot be empty")

        if item is None:
            env[key_text] = None
        else:
            env[key_text] = str(item)

    return env


def find_default_powershell_executable() -> str | None:
    """
    查找 PowerShell 可执行文件。

    优先级：
    1. pwsh        PowerShell 7+
    2. powershell  Windows PowerShell 5.1
    """
    return which("pwsh") or which("powershell")


def add_powershell_prelude(
    command: str,
    *,
    force_utf8: bool = True,
    silence_progress: bool = True,
) -> str:
    """
    给用户命令前面加 PowerShell 运行前置设置。

    force_utf8:
        尽量让 stdout/stderr 以 UTF-8 输出，避免中文乱码。

    silence_progress:
        关闭进度条输出，避免 Invoke-WebRequest 等命令刷屏。
    """
    prelude_parts: list[str] = []

    if force_utf8:
        prelude_parts.extend(
            [
                "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
                "$OutputEncoding = [System.Text.Encoding]::UTF8",
            ]
        )

    if silence_progress:
        prelude_parts.append("$ProgressPreference = 'SilentlyContinue'")

    if not prelude_parts:
        return command

    return "; ".join(prelude_parts + [command])


class PowerShellTool(BaseTool):
    """
    在 workspace 内执行 PowerShell 命令。

    这个工具负责：
    - Windows PowerShell / pwsh 适配
    - subprocess 执行
    - 捕获 stdout
    - 捕获 stderr
    - 捕获 exit_code
    - 捕获 timeout
    - 支持 stdin / env / cwd
    - 默认限制 cwd 在 workspace 内

    注意：
    - 这是危险工具。
    - 当前阶段先实现，不注册给 LLM。
    - 后续接 Permission 后再开放。
    """

    name: ClassVar[str] = "powershell"

    description: ClassVar[str] = (
        "Run a PowerShell command inside the current workspace and capture "
        "stdout, stderr, exit_code, timeout, and duration. Use only when "
        "PowerShell execution is explicitly required."
    )

    risk_level: ClassVar[ToolRiskLevel] = ToolRiskLevel.DANGEROUS

    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "PowerShell command string to execute.",
            },
            "cwd": {
                "type": "string",
                "description": (
                    "Working directory relative to workspace. "
                    "Defaults to workspace root."
                ),
            },
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds. Defaults to 60.",
            },
            "stdin": {
                "type": "string",
                "description": "Optional stdin text passed to the command.",
            },
            "env": {
                "type": "object",
                "description": (
                    "Extra environment variables. Use null value to remove a variable."
                ),
                "additionalProperties": {
                    "type": ["string", "null"],
                },
            },
            "inherit_env": {
                "type": "boolean",
                "description": "Whether to inherit current process env. Defaults to true.",
            },
            "executable": {
                "type": "string",
                "description": (
                    "Optional PowerShell executable path. "
                    "Defaults to pwsh or powershell found in PATH."
                ),
            },
            "no_profile": {
                "type": "boolean",
                "description": "Run PowerShell with -NoProfile. Defaults to true.",
            },
            "execution_policy_bypass": {
                "type": "boolean",
                "description": (
                    "Run with -ExecutionPolicy Bypass when supported. Defaults to true."
                ),
            },
            "force_utf8": {
                "type": "boolean",
                "description": (
                    "Prepend UTF-8 output settings before the command. Defaults to true."
                ),
            },
            "silence_progress": {
                "type": "boolean",
                "description": (
                    "Set ProgressPreference to SilentlyContinue. Defaults to true."
                ),
            },
            "max_stdout_chars": {
                "type": "integer",
                "description": "Maximum stdout chars returned. Defaults to 80000.",
            },
            "max_stderr_chars": {
                "type": "integer",
                "description": "Maximum stderr chars returned. Defaults to 80000.",
            },
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    async def execute(
        self,
        call: ToolCall,
        context: ToolExecutionContext,
    ) -> ToolResult:
        args = call.arguments

        command = str(args.get("command", "")).strip()

        if not command:
            raise ToolValidationError("command cannot be empty")

        cwd = args.get("cwd")
        cwd_text = str(cwd).strip() if cwd is not None else None

        timeout = coerce_float_arg(
            args.get("timeout"),
            name="timeout",
            default=DEFAULT_POWERSHELL_TIMEOUT_SECONDS,
            minimum=0.1,
        )

        max_stdout_chars = coerce_int_arg(
            args.get("max_stdout_chars"),
            name="max_stdout_chars",
            default=DEFAULT_POWERSHELL_MAX_OUTPUT_CHARS,
            minimum=1,
        )

        max_stderr_chars = coerce_int_arg(
            args.get("max_stderr_chars"),
            name="max_stderr_chars",
            default=DEFAULT_POWERSHELL_MAX_OUTPUT_CHARS,
            minimum=1,
        )

        inherit_env = coerce_bool_arg(
            args.get("inherit_env"),
            default=True,
        )

        no_profile = coerce_bool_arg(
            args.get("no_profile"),
            default=True,
        )

        execution_policy_bypass = coerce_bool_arg(
            args.get("execution_policy_bypass"),
            default=True,
        )

        force_utf8 = coerce_bool_arg(
            args.get("force_utf8"),
            default=True,
        )

        silence_progress = coerce_bool_arg(
            args.get("silence_progress"),
            default=True,
        )

        env = normalize_env_arg(args.get("env"))

        stdin = args.get("stdin")
        stdin_text = None if stdin is None else str(stdin)

        executable = args.get("executable")
        executable_text = str(executable).strip() if executable is not None else None

        if executable_text == "":
            executable_text = None

        if executable_text is None:
            executable_text = find_default_powershell_executable()

        if executable_text is None:
            raise ToolValidationError("PowerShell executable not found: pwsh or powershell")

        workspace_path = get_context_workspace_path(context)

        final_command = add_powershell_prelude(
            command,
            force_utf8=force_utf8,
            silence_progress=silence_progress,
        )

        argv = build_powershell_command(
            final_command,
            executable=executable_text,
            no_profile=no_profile,
            execution_policy_bypass=execution_policy_bypass,
        )

        shell_result = await run_shell_command(
            argv,
            shell=False,
            options=ShellRunOptions(
                cwd=cwd_text,
                workspace_path=workspace_path,
                allow_outside_workspace=False,
                env=env,
                inherit_env=inherit_env,
                timeout=timeout,
                stdin=stdin_text,
                max_stdout_chars=max_stdout_chars,
                max_stderr_chars=max_stderr_chars,
                kill_process_group=True,
            ),
        )

        content = render_shell_result(shell_result)

        data = {
            "tool": self.name,
            "command": command,
            "final_command": final_command,
            "argv": argv,
            "cwd": shell_result.cwd,
            "exit_code": shell_result.exit_code,
            "stdout": shell_result.stdout,
            "stderr": shell_result.stderr,
            "timed_out": shell_result.timed_out,
            "duration_ms": shell_result.duration_ms,
            "command_success": shell_result.success,
            "stdout_truncated": shell_result.stdout_truncated,
            "stderr_truncated": shell_result.stderr_truncated,
            "shell_result": shell_result.to_dict(),
        }

        return ToolResult.success_result(
            call=call,
            content=content,
            data=data,
        )

    def render_result(self, result: ToolResult) -> str:
        if not result.success:
            return result.error or result.content or "powershell failed"

        return result.content


def create_demo_context() -> ToolExecutionContext:
    signature = inspect.signature(ToolExecutionContext)
    kwargs: dict[str, Any] = {}

    if "workspace_path" in signature.parameters:
        kwargs["workspace_path"] = Path.cwd()

    if "project_root" in signature.parameters:
        kwargs["project_root"] = Path.cwd()

    if "permission_mode" in signature.parameters:
        kwargs["permission_mode"] = "default"

    if "metadata" in signature.parameters:
        kwargs["metadata"] = {}

    return ToolExecutionContext(**kwargs)


async def demo_async() -> None:
    if find_default_powershell_executable() is None:
        print("PowerShell executable not found; demo skipped.")
        return

    tool = PowerShellTool()
    context = create_demo_context()

    call = tool.create_call(
        {
            "command": "Write-Output 'hello from powershell'",
            "timeout": 10,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    print(tool.render_result(result))


def main() -> int:
    import asyncio

    asyncio.run(demo_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())