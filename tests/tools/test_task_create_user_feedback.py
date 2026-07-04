from __future__ import annotations

import pytest

from pywork.schemas.tool_schema import create_tool_call
from pywork.tools.task_tools import TaskCreateTool
from pywork.tools.tool import ToolExecutionContext


class FakeSubAgentManager:
    async def run_agent_task(self, *args, **kwargs):
        return {
            "task_id": "task_reviewer_1",
            "done": False,
            "cancelled": False,
            "record": {
                "id": "task_reviewer_1",
                "name": "Review diff.py",
                "status": "running",
                "agent_id": "reviewer",
            },
        }


@pytest.mark.asyncio
async def test_task_create_subagent_result_contains_user_visible_notice(tmp_path) -> None:
    tool = TaskCreateTool()

    result = await tool.execute(
        create_tool_call(
            "task_create",
            {
                "target": "subagent",
                "agent_name": "reviewer",
                "task": "Review diff.py",
                "wait": False,
            },
        ),
        ToolExecutionContext(
            workspace_path=str(tmp_path),
            project_root=str(tmp_path),
            permission_mode="default",
            metadata={
                "subagent_manager": FakeSubAgentManager(),
            },
        ),
    )

    assert result.success is True
    assert "Started background task:" in result.content
    assert "task_reviewer_1" in result.content
    assert "reviewer" in result.content
    assert "已创建后台任务" in result.content

    assert result.data["ui_notice"]["task_id"] == "task_reviewer_1"
    assert result.data["ui_notice"]["agent"] == "reviewer"
    assert result.data["ui_notice"]["name"] == "Review diff.py"
    assert result.data["ui_notice"]["status"] == "running"