from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.teams.team import TeamTaskStatus, create_team
from pywork.tools.task_update import TaskUpdateTool
from pywork.tools.tool import ToolExecutionContext


@dataclass
class FakeTaskRecord:
    id: str
    name: str = "fake"
    status: str = "pending"
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def mark_queued(self) -> None:
        self.status = "queued"

    def mark_running(self) -> None:
        self.status = "running"

    def mark_retrying(self) -> None:
        self.status = "retrying"

    def mark_succeeded(self, result=None) -> None:
        self.status = "succeeded"
        self.result = result
        self.error = None

    def mark_failed(self, error: str, result=None) -> None:
        self.status = "failed"
        self.error = error
        self.result = result

    def mark_cancelled(self, reason: str = "") -> None:
        self.status = "cancelled"
        self.error = reason

    def mark_aborted(self, reason: str = "") -> None:
        self.status = "aborted"
        self.error = reason

    def touch(self) -> None:
        self.metadata["touched"] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "result": (
                self.result.to_dict()
                if hasattr(self.result, "to_dict")
                else self.result
            ),
            "error": self.error,
            "metadata": self.metadata,
        }


class FakeTaskManager:
    def __init__(self) -> None:
        self._records: dict[str, FakeTaskRecord] = {}
        self.updated: list[str] = []

    async def get_task(self, task_id: str):
        return self._records.get(task_id)

    async def update_task(self, task) -> None:
        self._records[task.id] = task
        self.updated.append(task.id)


def make_context(**metadata):
    return ToolExecutionContext(
        workspace_path=".",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_task_update_marks_task_record_running() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(id="task_1")

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is True
    assert result.data["updated_targets"] == ["task_manager"]
    assert manager._records["task_1"].status == "running"
    assert manager.updated == ["task_1"]


@pytest.mark.asyncio
async def test_task_update_marks_task_record_succeeded_with_result() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(id="task_1")

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "status": "succeeded",
            "result": {
                "answer": 42,
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

    task = manager._records["task_1"]

    assert result.success is True
    assert task.status == "succeeded"
    assert task.error is None
    assert task.metadata["source"] == "test"
    assert task.metadata["touched"] is True
    assert task.result.success is True
    assert task.result.value == {
        "answer": 42,
    }


@pytest.mark.asyncio
async def test_task_update_marks_task_record_failed() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(id="task_1")

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "action": "fail",
            "error": "boom",
            "error_type": "RuntimeError",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    task = manager._records["task_1"]

    assert result.success is True
    assert task.status == "failed"
    assert task.error == "boom"
    assert task.result.success is False
    assert task.result.error == "boom"


@pytest.mark.asyncio
async def test_task_update_marks_task_record_cancelled() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(id="task_1")

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "action": "cancel",
            "reason": "user cancelled",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    task = manager._records["task_1"]

    assert result.success is True
    assert task.status == "cancelled"
    assert task.error == "user cancelled"


@pytest.mark.asyncio
async def test_task_update_updates_metadata_only() -> None:
    manager = FakeTaskManager()
    manager._records["task_1"] = FakeTaskRecord(
        id="task_1",
        metadata={
            "old": True,
        },
    )

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "metadata": {
                "new": True,
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    task = manager._records["task_1"]

    assert result.success is True
    assert task.status == "pending"
    assert task.metadata["old"] is True
    assert task.metadata["new"] is True


@pytest.mark.asyncio
async def test_task_update_updates_team_shared_task_running(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )

    shared_task = team.create_shared_task(
        "实现 task_update.py",
        task_id="team_task_1",
    )

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "team_task_1",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert result.data["updated_targets"] == ["team"]
    assert shared_task.status == TeamTaskStatus.RUNNING


@pytest.mark.asyncio
async def test_task_update_updates_team_shared_task_succeeded(tmp_path) -> None:
    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )

    shared_task = team.create_shared_task(
        "完成任务",
        task_id="team_task_1",
    )

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "team_task_1",
            "action": "result",
            "result": {
                "ok": True,
            },
        },
    )

    result = await tool.execute(
        call,
        make_context(team=team),
    )

    assert result.success is True
    assert shared_task.status == TeamTaskStatus.SUCCEEDED
    assert shared_task.result == {
        "ok": True,
    }


@pytest.mark.asyncio
async def test_task_update_updates_both_task_manager_and_team_when_ids_match(tmp_path) -> None:
    manager = FakeTaskManager()
    manager._records["same_id"] = FakeTaskRecord(id="same_id")

    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    shared_task = team.create_shared_task(
        "同 ID 任务",
        task_id="same_id",
    )

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "same_id",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(
            task_manager=manager,
            team=team,
        ),
    )

    assert result.success is True
    assert result.data["updated_targets"] == [
        "task_manager",
        "team",
    ]
    assert manager._records["same_id"].status == "running"
    assert shared_task.status == TeamTaskStatus.RUNNING


@pytest.mark.asyncio
async def test_task_update_target_task_manager_only(tmp_path) -> None:
    manager = FakeTaskManager()
    manager._records["same_id"] = FakeTaskRecord(id="same_id")

    team = create_team(
        team_id="team_1",
        workspace_path=tmp_path,
    )
    shared_task = team.create_shared_task(
        "同 ID 任务",
        task_id="same_id",
    )

    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "same_id",
            "status": "running",
            "target": "task_manager",
        },
    )

    result = await tool.execute(
        call,
        make_context(
            task_manager=manager,
            team=team,
        ),
    )

    assert result.success is True
    assert result.data["updated_targets"] == ["task_manager"]
    assert manager._records["same_id"].status == "running"
    assert shared_task.status == TeamTaskStatus.PENDING


@pytest.mark.asyncio
async def test_task_update_missing_runtime_returns_error() -> None:
    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "task_1",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(),
    )

    assert result.success is False
    assert "requires context.metadata" in result.error


@pytest.mark.asyncio
async def test_task_update_missing_task_returns_error() -> None:
    manager = FakeTaskManager()
    tool = TaskUpdateTool()

    call = create_tool_call(
        "task_update",
        {
            "task_id": "missing",
            "status": "running",
        },
    )

    result = await tool.execute(
        call,
        make_context(task_manager=manager),
    )

    assert result.success is False
    assert "task not found" in result.error