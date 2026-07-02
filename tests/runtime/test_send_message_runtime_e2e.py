from __future__ import annotations

import pytest

from pywork.runtime.engine import RuntimeEngine
from pywork.schemas.tool_schema import create_tool_call
from pywork.tools.send_message import SendMessageTool
from pywork.tools.tool import ToolExecutionContext


def make_context(
    engine: RuntimeEngine,
    *,
    agent_id: str,
    tmp_path,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_path=str(tmp_path),
        project_root=str(tmp_path),
        permission_mode="default",
        metadata={
            **engine.runtime_metadata,
            "agent_id": agent_id,
            "current_agent_id": agent_id,
        },
    )


@pytest.mark.asyncio
async def test_runtime_injects_shared_mailbox_and_team(tmp_path) -> None:
    engine = RuntimeEngine(
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        }
    )

    assert engine.mailbox is engine.runtime_metadata["mailbox"]
    assert engine.team is engine.runtime_metadata["team"]
    assert engine.team.mailbox is engine.mailbox
    assert engine.runtime_metadata["team_registry"][engine.team.team_id] is engine.team


@pytest.mark.asyncio
async def test_agent_a_sends_message_to_agent_b_through_runtime_mailbox(tmp_path) -> None:
    engine = RuntimeEngine(
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        }
    )

    tool = SendMessageTool()

    send_result = await tool.execute(
        create_tool_call(
            "send_message",
            {
                "action": "send",
                "recipient_id": "agent_b",
                "content": "hello from agent_a",
                "subject": "hello",
            },
        ),
        make_context(
            engine,
            agent_id="agent_a",
            tmp_path=tmp_path,
        ),
    )

    assert send_result.success is True

    inbox_result = await tool.execute(
        create_tool_call(
            "send_message",
            {
                "action": "inbox",
                "agent_id": "agent_b",
            },
        ),
        make_context(
            engine,
            agent_id="agent_b",
            tmp_path=tmp_path,
        ),
    )

    assert inbox_result.success is True
    assert inbox_result.data["count"] == 1

    message = inbox_result.data["messages"][0]

    assert message["sender_id"] == "agent_a"
    assert message["recipient_id"] == "agent_b"
    assert message["content"] == "hello from agent_a"


@pytest.mark.asyncio
async def test_team_and_teammates_share_runtime_mailbox(tmp_path) -> None:
    engine = RuntimeEngine(
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        }
    )

    planner = engine.team.create_teammate(
        teammate_id="planner_1",
        name="Planner",
        role="planner",
        agent_name="planner",
    )
    reviewer = engine.team.create_teammate(
        teammate_id="reviewer_1",
        name="Reviewer",
        role="reviewer",
        agent_name="reviewer",
    )

    assert planner.mailbox is engine.mailbox
    assert reviewer.mailbox is engine.mailbox
    assert engine.team.mailbox is engine.mailbox

    tool = SendMessageTool()

    send_result = await tool.execute(
        create_tool_call(
            "send_message",
            {
                "action": "send",
                "recipient_id": "reviewer_1",
                "content": "please review this",
                "subject": "review request",
            },
        ),
        ToolExecutionContext(
            workspace_path=str(tmp_path),
            project_root=str(tmp_path),
            permission_mode="default",
            metadata={
                **engine.runtime_metadata,
                "teammate": planner,
            },
        ),
    )

    assert send_result.success is True

    inbox_result = await tool.execute(
        create_tool_call(
            "send_message",
            {
                "action": "inbox",
                "agent_id": "reviewer_1",
            },
        ),
        ToolExecutionContext(
            workspace_path=str(tmp_path),
            project_root=str(tmp_path),
            permission_mode="default",
            metadata={
                **engine.runtime_metadata,
                "teammate": reviewer,
            },
        ),
    )

    assert inbox_result.success is True
    assert inbox_result.data["count"] == 1
    assert inbox_result.data["messages"][0]["sender_id"] == "planner_1"