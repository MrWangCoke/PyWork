from __future__ import annotations

import asyncio
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal


ShellKind = Literal["exec", "shell"]
ShellPlatform = Literal["windows", "posix"]


DEFAULT_ENCODING = "utf-8"
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_OUTPUT_CHARS = 80_000


class ShellError(Exception):
    """Shell 辅助模块基础异常。"""


class ShellValidationError(ShellError):
    """Shell 参数校验失败。"""


class ShellExecutionError(ShellError):
    """Shell 执行失败。"""


class ShellTimeoutError(ShellExecutionError):
    """Shell 命令执行超时。"""


@dataclass(slots=True, frozen=True)
class OutputLimitResult:
    """输出截断结果。"""

    text: str
    truncated: bool
    original_chars: int
    max_chars: int


@dataclass(slots=True, frozen=True)
class ShellCommand:
    """
    规范化后的命令。

    kind:
        exec  表示用 create_subprocess_exec，命令是 argv 列表。
        shell 表示用 create_subprocess_shell，命令是字符串。
    """

    kind: ShellKind
    command: str | tuple[str, ...]
    display: str

    @property
    def is_shell(self) -> bool:
        return self.kind == "shell"

    @property
    def argv(self) -> tuple[str, ...]:
        if self.kind != "exec":
            raise ShellValidationError("shell command does not have argv")

        if not isinstance(self.command, tuple):
            raise ShellValidationError("exec command must be a tuple")

        return self.command

    @property
    def shell_command(self) -> str:
        if self.kind != "shell":
            raise ShellValidationError("exec command does not have shell_command")

        if not isinstance(self.command, str):
            raise ShellValidationError("shell command must be a string")

        return self.command


@dataclass(slots=True, frozen=True)
class ShellRunOptions:
    """Shell 执行选项。"""

    cwd: str | Path | None = None
    workspace_path: str | Path | None = None
    allow_outside_workspace: bool = False

    env: Mapping[str, str | None] | None = None
    inherit_env: bool = True

    timeout: float = DEFAULT_TIMEOUT_SECONDS
    encoding: str = DEFAULT_ENCODING
    errors: str = "replace"

    stdin: str | bytes | None = None
    max_stdout_chars: int = DEFAULT_MAX_OUTPUT_CHARS
    max_stderr_chars: int = DEFAULT_MAX_OUTPUT_CHARS

    kill_process_group: bool = True


@dataclass(slots=True, frozen=True)
class ShellResult:
    """Shell 执行结果。"""

    command: str
    cwd: str
    exit_code: int | None

    stdout: str = ""
    stderr: str = ""

    timed_out: bool = False
    duration_ms: int = 0

    stdout_truncated: bool = False
    stderr_truncated: bool = False

    started_at: str = ""
    finished_at: str = ""

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def failed(self) -> bool:
        return not self.success

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def datetime_to_iso(value: datetime) -> str:
    return value.isoformat()


def current_shell_platform() -> ShellPlatform:
    if os.name == "nt":
        return "windows"

    return "posix"


def is_windows() -> bool:
    return current_shell_platform() == "windows"


def get_default_shell_name() -> str:
    if is_windows():
        return "powershell"

    return "bash"


def which(executable: str) -> str | None:
    """查找可执行文件路径。"""
    executable = executable.strip()

    if not executable:
        return None

    return shutil.which(executable)


def require_executable(executable: str) -> str:
    """查找可执行文件，找不到就抛异常。"""
    found = which(executable)

    if found is None:
        raise ShellValidationError(f"executable not found: {executable}")

    return found


def normalize_shell_command(
    command: str | Sequence[str],
    *,
    shell: bool = False,
) -> ShellCommand:
    """
    把用户传入的命令规范化。

    shell=False:
        推荐传 list/tuple，例如 ["python", "-V"]。
        如果传字符串，会用 shlex.split 做简单拆分。

    shell=True:
        推荐传字符串，例如 "python -V"。
        如果传 list/tuple，会拼成展示字符串后交给 shell。
    """
    if isinstance(command, str):
        text = command.strip()

        if not text:
            raise ShellValidationError("command cannot be empty")

        if shell:
            return ShellCommand(
                kind="shell",
                command=text,
                display=text,
            )

        argv = tuple(split_command_text(text))

        if not argv:
            raise ShellValidationError("command cannot be empty")

        return ShellCommand(
            kind="exec",
            command=argv,
            display=format_command_for_display(argv),
        )

    argv = tuple(str(part) for part in command)

    if not argv:
        raise ShellValidationError("command cannot be empty")

    if any(part == "" for part in argv):
        raise ShellValidationError("command arguments cannot contain empty strings")

    display = format_command_for_display(argv)

    if shell:
        return ShellCommand(
            kind="shell",
            command=display,
            display=display,
        )

    return ShellCommand(
        kind="exec",
        command=argv,
        display=display,
    )


def split_command_text(command: str) -> list[str]:
    """把命令字符串拆成 argv。"""
    try:
        return shlex.split(
            command,
            posix=not is_windows(),
        )
    except ValueError as exc:
        raise ShellValidationError(f"failed to split command: {exc}") from exc


def format_command_for_display(command: str | Sequence[str]) -> str:
    """把命令格式化成适合日志展示的字符串。"""
    if isinstance(command, str):
        return command

    argv = [str(part) for part in command]

    if is_windows():
        return subprocess.list2cmdline(argv)

    return shlex.join(argv)


def quote_shell_arg(value: str) -> str:
    """Shell 参数转义。"""
    if is_windows():
        return subprocess.list2cmdline([value])

    return shlex.quote(value)


def resolve_cwd(
    cwd: str | Path | None,
    *,
    workspace_path: str | Path | None = None,
    allow_outside_workspace: bool = False,
) -> Path:
    """
    解析命令运行目录。

    默认限制在 workspace 内，避免命令跑到项目外。
    """
    workspace = (
        Path(workspace_path).expanduser().resolve()
        if workspace_path is not None
        else Path.cwd().resolve()
    )

    if cwd is None:
        resolved = workspace
    else:
        raw_cwd = Path(cwd).expanduser()

        if not raw_cwd.is_absolute():
            raw_cwd = workspace / raw_cwd

        resolved = raw_cwd.resolve()

    if not resolved.exists():
        raise ShellValidationError(f"cwd does not exist: {resolved}")

    if not resolved.is_dir():
        raise ShellValidationError(f"cwd is not a directory: {resolved}")

    if not allow_outside_workspace:
        try:
            resolved.relative_to(workspace)
        except ValueError as exc:
            raise ShellValidationError(f"cwd is outside workspace: {resolved}") from exc

    return resolved


def merge_env(
    extra_env: Mapping[str, str | None] | None = None,
    *,
    inherit_env: bool = True,
) -> dict[str, str]:
    """
    合并环境变量。

    extra_env 中 value=None 表示删除该环境变量。
    """
    env = dict(os.environ) if inherit_env else {}

    if not extra_env:
        return env

    for key, value in extra_env.items():
        key_text = str(key)

        if not key_text:
            raise ShellValidationError("environment variable name cannot be empty")

        if value is None:
            env.pop(key_text, None)
        else:
            env[key_text] = str(value)

    return env


def limit_output(
    text: str,
    *,
    max_chars: int,
) -> OutputLimitResult:
    """限制 stdout / stderr 的最大字符数。"""
    if max_chars <= 0:
        raise ShellValidationError("max_chars must be > 0")

    original_chars = len(text)

    if original_chars <= max_chars:
        return OutputLimitResult(
            text=text,
            truncated=False,
            original_chars=original_chars,
            max_chars=max_chars,
        )

    suffix = (
        "\n"
        f"... output truncated "
        f"(original_chars={original_chars}, max_chars={max_chars})\n"
    )

    allowed = max(0, max_chars - len(suffix))

    return OutputLimitResult(
        text=text[:allowed] + suffix,
        truncated=True,
        original_chars=original_chars,
        max_chars=max_chars,
    )


def decode_process_output(
    content: bytes | None,
    *,
    encoding: str = DEFAULT_ENCODING,
    errors: str = "replace",
) -> str:
    if not content:
        return ""

    return content.decode(
        encoding,
        errors=errors,
    )


def encode_process_input(
    content: str | bytes | None,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> bytes | None:
    if content is None:
        return None

    if isinstance(content, bytes):
        return content

    return content.encode(encoding)


def get_process_creation_kwargs(
    *,
    kill_process_group: bool = True,
) -> dict[str, Any]:
    """
    创建子进程时的跨平台参数。

    Windows:
        CREATE_NEW_PROCESS_GROUP 方便后续中断整个进程组。

    POSIX:
        start_new_session=True 方便 killpg。
    """
    if not kill_process_group:
        return {}

    if is_windows():
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }

    return {
        "start_new_session": True,
    }


def terminate_process(
    process: asyncio.subprocess.Process,
    *,
    kill_process_group: bool = True,
) -> None:
    """尽量终止子进程。"""
    if process.returncode is not None:
        return

    if not kill_process_group:
        process.kill()
        return

    if is_windows():
        process.kill()
        return

    try:
        os.killpg(
            process.pid,
            signal.SIGKILL,
        )
    except ProcessLookupError:
        return
    except OSError:
        process.kill()


async def run_shell_command(
    command: str | Sequence[str],
    *,
    shell: bool = False,
    options: ShellRunOptions | None = None,
) -> ShellResult:
    """
    执行命令，捕获 stdout / stderr / exit_code。

    这个函数是后面 BashTool / PowerShellTool 的底层执行入口。
    """
    options = options or ShellRunOptions()

    normalized = normalize_shell_command(
        command,
        shell=shell,
    )

    cwd = resolve_cwd(
        options.cwd,
        workspace_path=options.workspace_path,
        allow_outside_workspace=options.allow_outside_workspace,
    )

    env = merge_env(
        options.env,
        inherit_env=options.inherit_env,
    )

    stdin_bytes = encode_process_input(
        options.stdin,
        encoding=options.encoding,
    )

    started_at = utc_now()
    process: asyncio.subprocess.Process | None = None

    try:
        process_kwargs = get_process_creation_kwargs(
            kill_process_group=options.kill_process_group,
        )

        if normalized.is_shell:
            process = await asyncio.create_subprocess_shell(
                normalized.shell_command,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_kwargs,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *normalized.argv,
                cwd=str(cwd),
                env=env,
                stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **process_kwargs,
            )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(stdin_bytes),
                timeout=options.timeout,
            )
            timed_out = False

        except TimeoutError:
            timed_out = True
            terminate_process(
                process,
                kill_process_group=options.kill_process_group,
            )
            stdout_bytes, stderr_bytes = await process.communicate()

        finished_at = utc_now()

        stdout = decode_process_output(
            stdout_bytes,
            encoding=options.encoding,
            errors=options.errors,
        )
        stderr = decode_process_output(
            stderr_bytes,
            encoding=options.encoding,
            errors=options.errors,
        )

        limited_stdout = limit_output(
            stdout,
            max_chars=options.max_stdout_chars,
        )
        limited_stderr = limit_output(
            stderr,
            max_chars=options.max_stderr_chars,
        )

        duration_ms = int((finished_at - started_at).total_seconds() * 1000)

        return ShellResult(
            command=normalized.display,
            cwd=str(cwd),
            exit_code=process.returncode,
            stdout=limited_stdout.text,
            stderr=limited_stderr.text,
            timed_out=timed_out,
            duration_ms=duration_ms,
            stdout_truncated=limited_stdout.truncated,
            stderr_truncated=limited_stderr.truncated,
            started_at=datetime_to_iso(started_at),
            finished_at=datetime_to_iso(finished_at),
            metadata={
                "shell": shell,
                "platform": current_shell_platform(),
                "python": sys.version.split()[0],
                "system": platform.system(),
                "stdout_original_chars": limited_stdout.original_chars,
                "stderr_original_chars": limited_stderr.original_chars,
            },
        )

    except FileNotFoundError as exc:
        raise ShellExecutionError(f"executable not found: {exc}") from exc

    except PermissionError as exc:
        raise ShellExecutionError(f"permission denied: {exc}") from exc

    except OSError as exc:
        raise ShellExecutionError(f"failed to execute command: {exc}") from exc


def build_bash_command(
    command: str,
    *,
    bash_executable: str | None = None,
    login: bool = False,
) -> list[str]:
    """
    构造 bash 命令。

    后面 BashTool 可以这样用：
        build_bash_command("ls -la")
    """
    executable = bash_executable or which("bash") or "bash"

    args = [executable]

    if login:
        args.append("-l")

    args.extend(
        [
            "-c",
            command,
        ]
    )

    return args


def build_powershell_command(
    command: str,
    *,
    executable: str | None = None,
    no_profile: bool = True,
    execution_policy_bypass: bool = True,
) -> list[str]:
    """
    构造 PowerShell 命令。

    优先使用传入 executable。
    后面 PowerShellTool 可以自己决定用 pwsh 还是 powershell。
    """
    resolved_executable = (
        executable
        or which("pwsh")
        or which("powershell")
        or "powershell"
    )

    args = [resolved_executable]

    if no_profile:
        args.append("-NoProfile")

    if execution_policy_bypass:
        args.extend(
            [
                "-ExecutionPolicy",
                "Bypass",
            ]
        )

    args.extend(
        [
            "-Command",
            command,
        ]
    )

    return args


def render_shell_result(result: ShellResult) -> str:
    """把 ShellResult 渲染成适合 ToolResult.content 的文本。"""
    status = "success" if result.success else "failed"

    parts = [
        f"# shell: {status}",
        f"# command: {result.command}",
        f"# cwd: {result.cwd}",
        f"# exit_code: {result.exit_code}",
        f"# duration_ms: {result.duration_ms}",
    ]

    if result.timed_out:
        parts.append("# timed_out: true")

    if result.stdout_truncated:
        parts.append("# stdout_truncated: true")

    if result.stderr_truncated:
        parts.append("# stderr_truncated: true")

    parts.append("")

    if result.stdout:
        parts.append("## stdout")
        parts.append(result.stdout.rstrip())
        parts.append("")

    if result.stderr:
        parts.append("## stderr")
        parts.append(result.stderr.rstrip())
        parts.append("")

    if not result.stdout and not result.stderr:
        parts.append("## output")
        parts.append("(no output)")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


async def demo_async() -> None:
    result = await run_shell_command(
        [
            sys.executable,
            "-c",
            "print('hello from shell utils')",
        ],
        options=ShellRunOptions(
            cwd=Path.cwd(),
            workspace_path=Path.cwd(),
        ),
    )

    print(render_shell_result(result))


def main() -> int:
    asyncio.run(demo_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())