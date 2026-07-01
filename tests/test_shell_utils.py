from __future__ import annotations

import sys
from pathlib import Path

import pytest

from pywork.utils.shell import (
    ShellRunOptions,
    ShellValidationError,
    build_bash_command,
    build_powershell_command,
    format_command_for_display,
    limit_output,
    merge_env,
    normalize_shell_command,
    render_shell_result,
    resolve_cwd,
    run_shell_command,
)


def test_normalize_exec_command_from_list() -> None:
    command = normalize_shell_command(
        [
            "python",
            "-V",
        ]
    )

    assert command.kind == "exec"
    assert command.argv == ("python", "-V")
    assert "python" in command.display


def test_normalize_shell_command_from_string() -> None:
    command = normalize_shell_command(
        "echo hello",
        shell=True,
    )

    assert command.kind == "shell"
    assert command.shell_command == "echo hello"
    assert command.display == "echo hello"


def test_format_command_for_display() -> None:
    display = format_command_for_display(
        [
            "python",
            "-c",
            "print('hello')",
        ]
    )

    assert "python" in display
    assert "hello" in display


def test_limit_output_truncates() -> None:
    result = limit_output(
        "abcdef",
        max_chars=5,
    )

    assert result.truncated
    assert result.original_chars == 6
    assert "output truncated" in result.text


def test_merge_env_add_and_remove() -> None:
    env = merge_env(
        {
            "PYWORK_TEST_ENV": "hello",
            "PATH": None,
        },
        inherit_env=True,
    )

    assert env["PYWORK_TEST_ENV"] == "hello"
    assert "PATH" not in env


def test_resolve_cwd_inside_workspace(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()

    resolved = resolve_cwd(
        "child",
        workspace_path=tmp_path,
    )

    assert resolved == child.resolve()


def test_resolve_cwd_rejects_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent

    with pytest.raises(ShellValidationError):
        resolve_cwd(
            outside,
            workspace_path=tmp_path,
        )


@pytest.mark.asyncio
async def test_run_shell_command_success(tmp_path: Path) -> None:
    result = await run_shell_command(
        [
            sys.executable,
            "-c",
            "print('hello shell')",
        ],
        options=ShellRunOptions(
            cwd=tmp_path,
            workspace_path=tmp_path,
        ),
    )

    assert result.success
    assert result.exit_code == 0
    assert "hello shell" in result.stdout
    assert result.stderr == ""


@pytest.mark.asyncio
async def test_run_shell_command_stderr_and_exit_code(tmp_path: Path) -> None:
    result = await run_shell_command(
        [
            sys.executable,
            "-c",
            "import sys; print('bad', file=sys.stderr); raise SystemExit(3)",
        ],
        options=ShellRunOptions(
            cwd=tmp_path,
            workspace_path=tmp_path,
        ),
    )

    assert not result.success
    assert result.exit_code == 3
    assert "bad" in result.stderr


@pytest.mark.asyncio
async def test_run_shell_command_timeout(tmp_path: Path) -> None:
    result = await run_shell_command(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(2)",
        ],
        options=ShellRunOptions(
            cwd=tmp_path,
            workspace_path=tmp_path,
            timeout=0.2,
        ),
    )

    assert result.timed_out
    assert not result.success


@pytest.mark.asyncio
async def test_run_shell_command_with_stdin(tmp_path: Path) -> None:
    result = await run_shell_command(
        [
            sys.executable,
            "-c",
            "import sys; print(sys.stdin.read().upper())",
        ],
        options=ShellRunOptions(
            cwd=tmp_path,
            workspace_path=tmp_path,
            stdin="hello",
        ),
    )

    assert result.success
    assert "HELLO" in result.stdout


def test_build_bash_command() -> None:
    command = build_bash_command("echo hello", bash_executable="bash")

    assert command == ["bash", "-c", "echo hello"]


def test_build_powershell_command() -> None:
    command = build_powershell_command(
        "Write-Output hello",
        executable="powershell",
    )

    assert command[0] == "powershell"
    assert "-NoProfile" in command
    assert "-Command" in command
    assert "Write-Output hello" in command


def test_render_shell_result(tmp_path: Path) -> None:
    # 这里用异步测试太重，直接复用一个真实执行结果更稳放在上面。
    # 这个测试只验证函数可导入和基本字符串结构，具体渲染在集成测试里看。
    from pywork.utils.shell import ShellResult

    rendered = render_shell_result(
        ShellResult(
            command="echo hello",
            cwd=str(tmp_path),
            exit_code=0,
            stdout="hello\n",
            stderr="",
            duration_ms=1,
        )
    )

    assert "# shell: success" in rendered
    assert "# command: echo hello" in rendered
    assert "## stdout" in rendered
    assert "hello" in rendered