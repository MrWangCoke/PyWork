from __future__ import annotations

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.teams.team import Team
from pywork.tools.team_create import TeamCreateTool
from pywork.tools.team_delete import TeamDeleteTool
from pywork.tools.tool import ToolExecutionContext


def make_context(**metadata):
    return ToolExecutionContext(
        workspace_path=".",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_team_create_registers_team_in_registry(tmp_path) -> None:
    registry = {}
    tool = TeamCreateTool()

    call = create_tool_call(
        "team_create",
        {
            "team_id": "team_1",
            "name": "Team One",
            "description": "测试团队",
            "workspace_path": str(tmp_path),
            "metadata": {
                "source": "test",
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(team_registry=registry),
    )

    assert result.success is True
    assert "team_1" in registry
    assert isinstance(registry["team_1"], Team)
    assert registry["team_1"].name == "Team One"
    assert registry["team_1"].metadata["source"] == "test"
    assert result.data["registered"] is True


@pytest.mark.asyncio
async def test_team_create_auto_creates_registry_in_context(tmp_path) -> None:
    metadata = {}
    tool = TeamCreateTool()

    call = create_tool_call(
        "team_create",
        {
            "team_id": "team_auto",
            "workspace_path": str(tmp_path),
        },
    )

    result = await tool.execute(
        call,
        ToolExecutionContext(
            workspace_path=tmp_path,
            metadata=metadata,
        ),
    )

    assert result.success is True
    assert "team_registry" in metadata
    assert "team_auto" in metadata["team_registry"]


@pytest.mark.asyncio
async def test_team_create_creates_members(tmp_path) -> None:
    registry = {}
    tool = TeamCreateTool()

    call = create_tool_call(
        "team_create",
        {
            "team_id": "team_members",
            "workspace_path": str(tmp_path),
            "members": [
                {
                    "teammate_id": "planner_1",
                    "role": "planner",
                    "name": "Planner One",
                },
                {
                    "teammate_id": "reviewer_1",
                    "role": "reviewer",
                    "name": "Reviewer One",
                },
            ],
        },
    )

    result = await tool.execute(
        call,
        make_context(team_registry=registry),
    )

    assert result.success is True

    team = registry["team_members"]

    assert len(team.list_members()) == 2
    assert team.require_teammate("planner_1").role == "planner"
    assert team.require_teammate("reviewer_1").role == "reviewer"
    assert len(result.data["created_members"]) == 2


@pytest.mark.asyncio
async def test_team_create_default_members(tmp_path) -> None:
    registry = {}
    tool = TeamCreateTool()

    call = create_tool_call(
        "team_create",
        {
            "team_id": "team_default",
            "workspace_path": str(tmp_path),
            "create_default_members": True,
        },
    )

    result = await tool.execute(
        call,
        make_context(team_registry=registry),
    )

    assert result.success is True

    team = registry["team_default"]
    roles = {
        member.role
        for member in team.list_members()
    }

    assert {
        "planner",
        "reviewer",
        "verifier",
        "general",
    }.issubset(roles)


@pytest.mark.asyncio
async def test_team_create_rejects_duplicate_without_replace(tmp_path) -> None:
    registry = {}
    tool = TeamCreateTool()

    first = create_tool_call(
        "team_create",
        {
            "team_id": "team_dup",
            "workspace_path": str(tmp_path),
        },
    )

    second = create_tool_call(
        "team_create",
        {
            "team_id": "team_dup",
            "workspace_path": str(tmp_path),
        },
    )

    first_result = await tool.execute(
        first,
        make_context(team_registry=registry),
    )
    second_result = await tool.execute(
        second,
        make_context(team_registry=registry),
    )

    assert first_result.success is True
    assert second_result.success is False
    assert "already exists" in second_result.error


@pytest.mark.asyncio
async def test_team_create_replace_duplicate(tmp_path) -> None:
    registry = {}
    tool = TeamCreateTool()

    first = create_tool_call(
        "team_create",
        {
            "team_id": "team_replace",
            "name": "Old",
            "workspace_path": str(tmp_path),
        },
    )

    second = create_tool_call(
        "team_create",
        {
            "team_id": "team_replace",
            "name": "New",
            "workspace_path": str(tmp_path),
            "replace": True,
        },
    )

    await tool.execute(
        first,
        make_context(team_registry=registry),
    )
    result = await tool.execute(
        second,
        make_context(team_registry=registry),
    )

    assert result.success is True
    assert registry["team_replace"].name == "New"


@pytest.mark.asyncio
async def test_team_create_set_current(tmp_path) -> None:
    metadata = {
        "team_registry": {},
    }
    tool = TeamCreateTool()

    call = create_tool_call(
        "team_create",
        {
            "team_id": "team_current",
            "workspace_path": str(tmp_path),
            "set_current": True,
        },
    )

    result = await tool.execute(
        call,
        ToolExecutionContext(
            workspace_path=tmp_path,
            metadata=metadata,
        ),
    )

    assert result.success is True
    assert metadata["team"].team_id == "team_current"


@pytest.mark.asyncio
async def test_team_delete_removes_team_from_registry(tmp_path) -> None:
    registry = {}
    create_tool = TeamCreateTool()
    delete_tool = TeamDeleteTool()

    await create_tool.execute(
        create_tool_call(
            "team_create",
            {
                "team_id": "team_delete",
                "workspace_path": str(tmp_path),
            },
        ),
        make_context(team_registry=registry),
    )

    assert "team_delete" in registry

    result = await delete_tool.execute(
        create_tool_call(
            "team_delete",
            {
                "team_id": "team_delete",
                "stop_members": False,
                "cancel_current": False,
            },
        ),
        make_context(team_registry=registry),
    )

    assert result.success is True
    assert "team_delete" not in registry
    assert result.data["removed_from_registry"] is True


@pytest.mark.asyncio
async def test_team_delete_stops_members(tmp_path) -> None:
    registry = {}
    create_tool = TeamCreateTool()
    delete_tool = TeamDeleteTool()

    await create_tool.execute(
        create_tool_call(
            "team_create",
            {
                "team_id": "team_stop",
                "workspace_path": str(tmp_path),
                "members": [
                    {
                        "teammate_id": "planner_1",
                        "role": "planner",
                    },
                    {
                        "teammate_id": "reviewer_1",
                        "role": "reviewer",
                    },
                ],
            },
        ),
        make_context(team_registry=registry),
    )

    team = registry["team_stop"]

    result = await delete_tool.execute(
        create_tool_call(
            "team_delete",
            {
                "team_id": "team_stop",
                "stop_members": True,
                "cancel_current": False,
            },
        ),
        make_context(team_registry=registry),
    )

    assert result.success is True
    assert result.data["stopped_member_count"] == 2
    assert team.require_teammate("planner_1").is_stopped
    assert team.require_teammate("reviewer_1").is_stopped


@pytest.mark.asyncio
async def test_team_delete_clears_current_team(tmp_path) -> None:
    metadata = {
        "team_registry": {},
    }

    create_tool = TeamCreateTool()
    delete_tool = TeamDeleteTool()

    await create_tool.execute(
        create_tool_call(
            "team_create",
            {
                "team_id": "team_current",
                "workspace_path": str(tmp_path),
                "set_current": True,
            },
        ),
        ToolExecutionContext(
            workspace_path=tmp_path,
            metadata=metadata,
        ),
    )

    assert metadata["team"].team_id == "team_current"

    result = await delete_tool.execute(
        create_tool_call(
            "team_delete",
            {
                "team_id": "team_current",
                "clear_current": True,
                "stop_members": False,
                "cancel_current": False,
            },
        ),
        ToolExecutionContext(
            workspace_path=tmp_path,
            metadata=metadata,
        ),
    )

    assert result.success is True
    assert "team" not in metadata


@pytest.mark.asyncio
async def test_team_delete_missing_team_returns_error() -> None:
    registry = {}
    tool = TeamDeleteTool()

    result = await tool.execute(
        create_tool_call(
            "team_delete",
            {
                "team_id": "missing",
            },
        ),
        make_context(team_registry=registry),
    )

    assert result.success is False
    assert "team not found" in result.error