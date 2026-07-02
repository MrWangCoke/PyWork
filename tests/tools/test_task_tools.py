from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.teams.team import TeamTaskStatus, create_team
from pywork.tools.task_tools import (
    TaskCreateTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
)
from pywork.tools.tool import ToolExecutionContext


@dataclass
class FakeTaskRecord:
    id: str
    name: str = "fake"
    status: str = "pending"
    payload: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    parent_id: str | None = None
    agent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_cancelled(self, reason: str = "") -> None:
        self.status = "cancelled"
        self.error = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "payload": self.payload,
            "result": self.result,
            "error": self.error,
            "parent_id": self.parent_id,
            "agent_id": self.agent_id,
            "metadata": self.metadata,
        }


class FakeTaskManager:
    def __init__(self) -> None:
        self._records: dict[str, FakeTaskRecord] = {}
        self.updated: list[str] = []
        self.cancelled: list[str] = []

    async def create_task(
        self,
        *,
        name,
        task_type=None,
        payload=None,
        parent_id=None,
        agent_id=None,
        metadata=None,
        max_retries=0,
        timeout_seconds=None,
        created_by=None,
    ):
        task_id = f"task_{len(self._records) + 1}"
        record = FakeTaskRecord(
            id=task_id,
            name=name,
            payload=payload or {},
            parent_id=parent_id,
            agent_id=agent_id,
            metadata=metadata or {},
        )
        self._records[task_id] = record
        return record

    async def register_task(self, task) -> None:
        self._records[task.id] = task

    async def update_task(self, task) -> None:
        self._records[task.id] = task
        self.updated.append(task.id)

    async def get_task(self, task_id: str):
        return self._records.get(task_id)

    async def list_tasks(
        self,
        status=None,
        parent_id=None,
        agent_id=None,
        limit=None,
    ):
        records = list(self._records.values())

        if status is not None:
            status_value = getattr(status, "value", status)
            records = [
                record
                for record in records
                if record.status == status_value
            ]

        if parent_id is not None:
            records = [
                record
                for record in records
                if record.parent_id == parent_id
            ]

        if agent_id is not None:
            records = [
                record
                for record in records
                if record.agent_id == agent_id
            ]

        if limit is not None:
            records = records[:limit]

        return records

    async def cancel_task(self, task_id: str, reason: str = "cancelled"):
        record = self._records.get(task_id)

        if record is None:
            return None

        record.mark_cancelled(reason)
        self.cancelled.append(task_id)

        return record


class FakeSubAgentManager:
    def __init__(self) -> None:
        self.started = []
        self.cancelled = []

    async def run_agent_task(self, request, wait=False):
        self.started.append((request, wait))

        return {
            "task_id": "subagent_task_1",
            "agent_name": request.agent_name,
            "task": request.task,
            "wait": wait,
        }

    async def cancel_agent_task(self, task_id: str, reason: str = "cancelled"):
        self.cancelled.append((task_id, reason))
        return True


def make_context(**metadata):
    return ToolExecutionContext(
        workspace_path=".",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_task_create_creates_task_manager_record() -> None:
    manager = FakeTaskManager()
    tool = TaskCreateTool()

    call = create_tool_call(
        "task_create",
        {
            "target": "task_manager",
            "name": "实现功能",
            "task": "实现 task_tools.py",
            "payload": {
                "file": "task_tools.py",
            },
            "metadata": {
                "source": "test",
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["target"] == "task_manager"

    task = result.data["task"]

    assert task["name"] == "实现功能"
    assert task["payload"]["task"] == "实现 task_tools.py"
    assert task["payload"]["file"] == "task_tools.py"
    assert task["metadata"]["source"] == "test"
    assert "task_1" in manager._records


@pytest.mark.asyncio
async def test_task_create_creates_team_shared_task(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    tool = TaskCreateTool()

    call = create_tool_call(
        "task_create",
        {
            "target": "team",
            "task_id": "team_task_1",
            "title": "团队任务",
            "description": "实现 Team shared task",
            "role": "planner",
            "priority": "high",
            "payload": {
                "x": 1,
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["target"] == "team"

    task = team.require_shared_task("team_task_1")

    assert task.title == "团队任务"
    assert task.role == "planner"
    assert task.payload["x"] == 1


@pytest.mark.asyncio
async def test_task_create_starts_subagent_task() -> None:
    manager = FakeSubAgentManager()
    tool = TaskCreateTool()

    call = create_tool_call(
        "task_create",
        {
            "target": "subagent",
            "agent_name": "planner",
            "task": "规划实现",
            "wait": False,
        },
    )

    result = await tool.execute(
        call,
        make_context(subagent_manager=manager),
    )

    assert result.success is True
    assert result.data["target"] == "subagent"
    assert len(manager.started) == 1

    request, wait = manager.started[0]

    assert request.agent_name == "planner"
    assert request.task == "规划实现"
    assert wait is False


@pytest.mark.asyncio
async def test_task_list_lists_task_manager_records() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(
        id="task_1",
        name="one",
        status="pending",
    )
    manager._records["task_2"] = FakeTaskRecord(
        id="task_2",
        name="two",
        status="running",
    )

    tool = TaskListTool()

    call = create_tool_call(
        "task_list",
        {
            "target": "task_manager",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["count"] == 2
    assert {
        task["id"]
        for task in result.data["task_records"]
    } == {
        "task_1",
        "task_2",
    }


@pytest.mark.asyncio
async def test_task_list_filters_task_manager_status() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(
        id="task_1",
        name="one",
        status="pending",
    )
    manager._records["task_2"] = FakeTaskRecord(
        id="task_2",
        name="two",
        status="running",
    )

    tool = TaskListTool()

    call = create_tool_call(
        "task_list",
        {
            "target": "task_manager",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["count"] == 1
    assert result.data["task_records"][0]["id"] == "task_2"


@pytest.mark.asyncio
async def test_task_list_lists_team_tasks(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    team.create_shared_task(
        "任务一",
        task_id="team_task_1",
        role="planner",
    )
    team.create_shared_task(
        "任务二",
        task_id="team_task_2",
        role="reviewer",
    )

    tool = TaskListTool()

    call = create_tool_call(
        "task_list",
        {
            "target": "team",
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["count"] == 2
    assert {
        task["task_id"]
        for task in result.data["team_tasks"]
    } == {
        "team_task_1",
        "team_task_2",
    }


@pytest.mark.asyncio
async def test_task_output_gets_task_manager_record() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(
        id="task_1",
        name="one",
        status="succeeded",
        result={
            "ok": True,
        },
    )

    tool = TaskOutputTool()

    call = create_tool_call(
        "task_output",
        {
            "target": "task_manager",
            "task_id": "task_1",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["sources"] == ["task_manager"]
    assert result.data["task_record"]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_task_output_gets_team_task(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    task = team.create_shared_task(
        "任务一",
        task_id="team_task_1",
    )
    task.mark_succeeded(
        {
            "ok": True,
        }
    )

    tool = TaskOutputTool()

    call = create_tool_call(
        "task_output",
        {
            "target": "team",
            "task_id": "team_task_1",
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["sources"] == ["team"]
    assert result.data["team_task"]["result"]["ok"] is True


@pytest.mark.asyncio
async def test_task_stop_cancels_task_manager_record() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(
        id="task_1",
        name="one",
        status="running",
    )

    tool = TaskStopTool()

    call = create_tool_call(
        "task_stop",
        {
            "target": "task_manager",
            "task_id": "task_1",
            "reason": "user stop",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["cancelled_targets"] == ["task_manager"]
    assert manager._records["task_1"].status == "cancelled"
    assert manager._records["task_1"].error == "user stop"


@pytest.mark.asyncio
async def test_task_stop_cancels_team_task(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    task = team.create_shared_task(
        "任务一",
        task_id="team_task_1",
    )
    task.mark_running()

    tool = TaskStopTool()

    call = create_tool_call(
        "task_stop",
        {
            "target": "team",
            "task_id": "team_task_1",
            "reason": "stop team task",
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["cancelled_targets"] == ["team"]
    assert task.status == TeamTaskStatus.CANCELLED
    assert task.error == "stop team task"


@pytest.mark.asyncio
async def test_task_stop_cancels_subagent_task() -> None:
    manager = FakeSubAgentManager()
    tool = TaskStopTool()

    call = create_tool_call(
        "task_stop",
        {
            "target": "subagent",
            "task_id": "subagent_task_1",
            "reason": "stop subagent",
        },
    )

    result = await tool.execute(
        call,
        make_context(subagent_manager=manager),
    )

    assert result.success is True
    assert result.data["cancelled_targets"] == ["subagent"]
    assert manager.cancelled == [
        ("subagent_task_1", "stop subagent"),
    ]


@pytest.mark.asyncio
async def test_task_output_missing_task_returns_error() -> None:
    manager = FakeTaskManager()
    tool = TaskOutputTool()

    call = create_tool_call(
        "task_output",
        {
            "target": "task_manager",
            "task_id": "missing",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is False
    assert "task not found" in result.error


@pytest.mark.asyncio
async def test_task_create_missing_runtime_returns_error() -> None:
    tool = TaskCreateTool()

    call = create_tool_call(
        "task_create",
        {
            "name": "无运行时任务",
        },
    )

    result = await tool.execute(
        call,
        make_context(),
    )

    assert result.success is False
    assert "requires context.metadata" in result.error