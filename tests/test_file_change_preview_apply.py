from __future__ import annotations

from pathlib import Path

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.tools.file_edit import FileEditTool
from pywork.tools.file_write import FileWriteTool
from pywork.tools.tool import ToolExecutionContext


def make_context(tmp_path: Path) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
    )


def test_file_write_preview_does_not_create_file(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
    )

    preview = tool.preview(call, context)

    assert preview.operation == "write"
    assert preview.has_changes
    assert "src/utils/helper.py" in preview.path
    assert "+print('hello')" in preview.diff_text
    assert not (tmp_path / "src" / "utils" / "helper.py").exists()


def test_file_write_apply_creates_file_after_preview(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "src/utils/helper.py",
            "content": "print('hello')\n",
        },
    )

    preview = tool.preview(call, context)
    result = tool.apply(preview)

    assert result.applied
    assert result.changed
    assert (tmp_path / "src" / "utils" / "helper.py").read_text(
        encoding="utf-8"
    ) == "print('hello')\n"


@pytest.mark.asyncio
async def test_file_write_preview_only_run_does_not_write(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "helper.py",
            "content": "x = 1\n",
            "preview_only": True,
        },
    )

    result = await tool.run(call, context)

    assert result.success
    assert result.data["preview_only"] is True
    assert not (tmp_path / "helper.py").exists()


@pytest.mark.asyncio
async def test_file_write_execute_writes_after_preview_apply(tmp_path: Path) -> None:
    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "helper.py",
            "content": "x = 1\n",
        },
    )

    result = await tool.run(call, context)

    assert result.success
    assert result.data["applied"] is True
    assert (tmp_path / "helper.py").read_text(encoding="utf-8") == "x = 1\n"


def test_file_write_refuses_existing_file_without_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "helper.py"
    target.write_text("old\n", encoding="utf-8")

    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "helper.py",
            "content": "new\n",
        },
    )

    with pytest.raises(Exception):
        tool.preview(call, context)

    assert target.read_text(encoding="utf-8") == "old\n"


def test_file_write_overwrite_preview_does_not_modify_until_apply(tmp_path: Path) -> None:
    target = tmp_path / "helper.py"
    target.write_text("old\n", encoding="utf-8")

    tool = FileWriteTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_write",
        arguments={
            "path": "helper.py",
            "content": "new\n",
            "overwrite": True,
        },
    )

    preview = tool.preview(call, context)

    assert "-old" in preview.diff_text
    assert "+new" in preview.diff_text
    assert target.read_text(encoding="utf-8") == "old\n"

    tool.apply(preview)

    assert target.read_text(encoding="utf-8") == "new\n"


def test_file_edit_preview_does_not_modify_file(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("name = 'old'\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "old",
            "new_string": "new",
        },
    )

    preview = tool.preview(call, context)

    assert preview.operation == "edit"
    assert "-name = 'old'" in preview.diff_text
    assert "+name = 'new'" in preview.diff_text
    assert target.read_text(encoding="utf-8") == "name = 'old'\n"


def test_file_edit_apply_modifies_after_preview(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("name = 'old'\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "old",
            "new_string": "new",
        },
    )

    preview = tool.preview(call, context)
    result = tool.apply(preview)

    assert result.applied
    assert target.read_text(encoding="utf-8") == "name = 'new'\n"


@pytest.mark.asyncio
async def test_file_edit_preview_only_run_does_not_write(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("x = 1\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "1",
            "new_string": "2",
            "preview_only": True,
        },
    )

    result = await tool.run(call, context)

    assert result.success
    assert result.data["preview_only"] is True
    assert target.read_text(encoding="utf-8") == "x = 1\n"


@pytest.mark.asyncio
async def test_file_edit_execute_applies_change(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("x = 1\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "1",
            "new_string": "2",
        },
    )

    result = await tool.run(call, context)

    assert result.success
    assert result.data["applied"] is True
    assert target.read_text(encoding="utf-8") == "x = 2\n"


def test_file_edit_requires_unique_match_by_default(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "1",
            "new_string": "2",
        },
    )

    with pytest.raises(Exception):
        tool.preview(call, context)

    assert target.read_text(encoding="utf-8") == "x = 1\nx = 1\n"


def test_file_edit_occurrence_replaces_only_selected_match(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("x = 1\ny = 1\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "1",
            "new_string": "2",
            "occurrence": 2,
        },
    )

    preview = tool.preview(call, context)
    tool.apply(preview)

    assert target.read_text(encoding="utf-8") == "x = 1\ny = 2\n"


def test_file_edit_replace_all(tmp_path: Path) -> None:
    target = tmp_path / "demo.py"
    target.write_text("x = 1\ny = 1\n", encoding="utf-8")

    tool = FileEditTool()
    context = make_context(tmp_path)

    call = create_tool_call(
        tool_name="file_edit",
        arguments={
            "path": "demo.py",
            "old_string": "1",
            "new_string": "2",
            "replace_all": True,
        },
    )

    preview = tool.preview(call, context)
    tool.apply(preview)

    assert target.read_text(encoding="utf-8") == "x = 2\ny = 2\n"