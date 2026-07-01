from __future__ import annotations

from pathlib import Path

import pytest

from pywork.tools.powershell import PowerShellTool, find_default_powershell_executable
from pywork.tools.tool import ToolExecutionContext


pytestmark = pytest.mark.skipif(
    find_default_powershell_executable() is None,
    reason="PowerShell executable not available",
)


def make_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
    )


@pytest.mark.asyncio
async def test_powershell_tool_stdout(tmp_path: Path) -> None:
    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Write-Output 'hello powershell'",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert result.data["exit_code"] == 0
    assert result.data["command_success"]
    assert "hello powershell" in result.data["stdout"]
    assert "## stdout" in result.content


@pytest.mark.asyncio
async def test_powershell_tool_stderr_and_nonzero_exit(tmp_path: Path) -> None:
    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Write-Error 'bad'; exit 7",
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
async def test_powershell_tool_timeout(tmp_path: Path) -> None:
    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Start-Sleep -Seconds 2",
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
async def test_powershell_tool_stdin(tmp_path: Path) -> None:
    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "[Console]::In.ReadToEnd().ToUpperInvariant()",
            "stdin": "hello stdin",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert "HELLO STDIN" in result.data["stdout"]


@pytest.mark.asyncio
async def test_powershell_tool_env(tmp_path: Path) -> None:
    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Write-Output $env:PYWORK_PS_TEST",
            "env": {
                "PYWORK_PS_TEST": "env-ok",
            },
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
async def test_powershell_tool_cwd_inside_workspace(tmp_path: Path) -> None:
    subdir = tmp_path / "sub"
    subdir.mkdir()

    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Set-Content -Path out.txt -Value 'created' -NoNewline",
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
async def test_powershell_tool_rejects_outside_cwd(tmp_path: Path) -> None:
    outside = tmp_path.parent

    tool = PowerShellTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "command": "Write-Output 'bad'",
            "cwd": str(outside),
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "outside workspace" in result.content