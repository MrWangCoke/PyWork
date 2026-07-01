from __future__ import annotations

from pathlib import Path

import pytest

from pathlib import Path

from pywork.utils.shell import which


def find_git_bash() -> str | None:
    candidates = [
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        r"C:\Program Files (x86)\Git\bin\bash.exe",
        r"C:\Program Files (x86)\Git\usr\bin\bash.exe",
    ]

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    return which("bash")

from pywork.tools.bash import BashTool
from pywork.tools.tool import ToolExecutionContext
from pywork.utils.shell import which


pytestmark = pytest.mark.skipif(
    which("bash") is None,
    reason="bash executable not available",
)


def make_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
    )


@pytest.mark.asyncio
async def test_bash_tool_stdout(tmp_path: Path) -> None:
    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "printf 'hello bash\\n'",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["exit_code"] == 0
    assert result.data["command_success"]
    assert "hello bash" in result.data["stdout"]
    assert "## stdout" in result.content


@pytest.mark.asyncio
async def test_bash_tool_stderr_and_nonzero_exit(tmp_path: Path) -> None:
    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "printf 'bad\\n' >&2; exit 7",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["exit_code"] == 7
    assert not result.data["command_success"]
    assert "bad" in result.data["stderr"]
    assert "## stderr" in result.content


@pytest.mark.asyncio
async def test_bash_tool_timeout(tmp_path: Path) -> None:
    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "sleep 2",
            "timeout": 0.2,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["timed_out"]
    assert not result.data["command_success"]


@pytest.mark.asyncio
async def test_bash_tool_stdin(tmp_path: Path) -> None:
    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "cat",
            "stdin": "hello stdin",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert "hello stdin" in result.data["stdout"]


@pytest.mark.asyncio
async def test_bash_tool_env(tmp_path: Path) -> None:
    bash_executable = find_git_bash()

    if bash_executable is None:
        pytest.skip("bash executable not available")

    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "printf \"$PYWORK_BASH_TEST\"",
            "env": {
                "PYWORK_BASH_TEST": "env-ok",
            },
            "bash_executable": bash_executable,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["exit_code"] == 0
    assert "env-ok" in result.data["stdout"]

@pytest.mark.asyncio
async def test_bash_tool_cwd_inside_workspace(tmp_path: Path) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()

    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "printf 'created' > out.txt",
            "cwd": "sub",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["exit_code"] == 0
    assert (subdir / "out.txt").read_text(encoding="utf-8") == "created"


@pytest.mark.asyncio
async def test_bash_tool_rejects_outside_cwd(tmp_path: Path) -> None:
    outside = tmp_path.parent

    tool = BashTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "printf 'bad'",
            "cwd": str(outside),
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "outside workspace" in result.content