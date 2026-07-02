from pathlib import Path

import pytest

from pywork.tools.file_edit import FileEditTool
from pywork.tools.file_write import FileWriteTool
from pywork.tools.tool import ToolExecutionContext


def make_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
    )


@pytest.mark.asyncio
async def test_file_write_creates_new_file(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "hello.txt",
            "content": "hello\nworld\n",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert (tmp_path / "hello.txt").read_text(encoding="utf-8") == "hello\nworld\n"
    assert "+++ b/hello.txt" in result.content
    assert "+hello" in result.content


@pytest.mark.asyncio
async def test_file_write_overwrites_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_text(
        "old\n",
        encoding="utf-8",
    )

    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "hello.txt",
            "content": "new\n",
            "overwrite": True,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert path.read_text(encoding="utf-8") == "new\n"
    assert "-old" in result.content
    assert "+new" in result.content


@pytest.mark.asyncio
async def test_file_write_rejects_overwrite_false(tmp_path: Path) -> None:
    path = tmp_path / "hello.txt"
    path.write_text(
        "old\n",
        encoding="utf-8",
    )

    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "hello.txt",
            "content": "new\n",
            "overwrite": False,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "overwrite=true" in result.content


@pytest.mark.asyncio
async def test_file_write_rejects_outside_workspace(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "../outside.txt",
            "content": "bad\n",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "outside workspace" in result.content


@pytest.mark.asyncio
async def test_file_edit_replaces_exact_string(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text(
        "hello old world\n",
        encoding="utf-8",
    )

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "demo.txt",
            "old_string": "old",
            "new_string": "new",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert path.read_text(encoding="utf-8") == "hello new world\n"
    assert "-hello old world" in result.content
    assert "+hello new world" in result.content


@pytest.mark.asyncio
async def test_file_edit_rejects_missing_old_string(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text(
        "hello world\n",
        encoding="utf-8",
    )

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "demo.txt",
            "old_string": "missing",
            "new_string": "new",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "old_string was not found" in result.content


@pytest.mark.asyncio
async def test_file_edit_rejects_ambiguous_multiple_matches(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text(
        "old\nold\n",
        encoding="utf-8",
    )

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "demo.txt",
            "old_string": "old",
            "new_string": "new",
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert not result.success
    assert "old_string appears 2 times" in result.content


@pytest.mark.asyncio
async def test_file_edit_replace_all(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text(
        "old\nold\n",
        encoding="utf-8",
    )

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "demo.txt",
            "old_string": "old",
            "new_string": "new",
            "replace_all": True,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert path.read_text(encoding="utf-8") == "new\nnew\n"


@pytest.mark.asyncio
async def test_file_edit_specific_occurrence(tmp_path: Path) -> None:
    path = tmp_path / "demo.txt"
    path.write_text(
        "old\nold\nold\n",
        encoding="utf-8",
    )

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = tool.create_call(
        {
            "path": "demo.txt",
            "old_string": "old",
            "new_string": "new",
            "occurrence": 2,
        }
    )

    result = await tool.run(
        call,
        context,
    )

    assert result.success
    assert path.read_text(encoding="utf-8") == "old\nnew\nold\n"