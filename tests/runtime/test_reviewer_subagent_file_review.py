from __future__ import annotations

import pytest

from pywork.runtime.graph import (
    AgentGraphRunner,
    create_default_agent_graph_state,
    detect_reviewer_subagent_tool_call,
)
from pywork.subagents.base import SubAgentContext
from pywork.subagents.manager import create_default_subagent_manager
from pywork.subagents.reviewer import ReviewerSubAgent
from pywork.tools.registry import create_default_registry


def write_diff_file(tmp_path):
    path = tmp_path / "src" / "pywork" / "utils" / "diff.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "DEFAULT_CONTEXT_LINES = 3",
                "",
                "def build_patch(old: str, new: str) -> str:",
                "    return '\\n'.join([old, new])",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_reviewer_route_rewrites_legacy_diff_path(tmp_path) -> None:
    write_diff_file(tmp_path)

    registry = create_default_registry()

    data = create_default_agent_graph_state(
        user_input="用 SubAgent 审查 src/utils/diff.py",
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            }
        },
        metadata={},
    )

    call = detect_reviewer_subagent_tool_call(data)

    assert call is not None
    assert call.tool_name == "agent"
    assert call.arguments["action"] == "run"
    assert call.arguments["agent_name"] == "reviewer"
    assert call.arguments["metadata"]["review_target_path"] == "src/pywork/utils/diff.py"
    assert "src/pywork/utils/diff.py" in call.arguments["task"]


@pytest.mark.asyncio
async def test_reviewer_agent_reviews_diff_file(tmp_path) -> None:
    write_diff_file(tmp_path)

    captured = {}

    async def fake_review_llm(messages, *, tools=None, metadata=None):
        captured["messages"] = messages
        captured["tools"] = tools
        captured["metadata"] = metadata

        joined = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
        )

        assert "DEFAULT_CONTEXT_LINES = 3" in joined
        assert "def build_patch" in joined
        assert metadata["review_target_path"] == "src/pywork/utils/diff.py"
        assert metadata["review_file_loaded"] is True

        return """
1. Summary
Reviewer read src/pywork/utils/diff.py and reviewed the diff helper.

2. Issues found
No blocking issue in this test fixture.

3. Safety and permission concerns
Readonly review only.

4. Test coverage gaps
Add tests for empty old/new input.

5. Suggested fixes
Add edge-case tests.

6. Recommended next action
Create unit tests for build_patch.
""".strip()

    agent = ReviewerSubAgent(
        llm=fake_review_llm,
        tool_definitions=[
            {
                "name": "file_read",
                "description": "Read file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                        }
                    },
                },
            }
        ],
    )

    result = await agent.run(
        "Review file src/utils/diff.py",
        context=SubAgentContext(
            task="Review file src/utils/diff.py",
            workspace_path=tmp_path,
        ),
        metadata={
            "review_target_path": "src/utils/diff.py",
        },
    )

    assert result.success is True
    assert result.name == "reviewer"
    assert "1. Summary" in result.content
    assert result.metadata["reviewer"]["review_file_loaded"] is True
    assert result.metadata["reviewer"]["review_target_path"] == "src/pywork/utils/diff.py"


@pytest.mark.asyncio
async def test_runtime_routes_natural_language_to_reviewer_agent(tmp_path) -> None:
    write_diff_file(tmp_path)

    async def fake_review_llm(messages, *, tools=None, metadata=None):
        joined = "\n\n".join(
            str(message.get("content", ""))
            for message in messages
        )

        assert "DEFAULT_CONTEXT_LINES = 3" in joined
        assert "src/pywork/utils/diff.py" in joined

        return """
1. Summary
Reviewed src/pywork/utils/diff.py.

2. Issues found
No critical issue in test fixture.

3. Safety and permission concerns
Readonly review.

4. Test coverage gaps
Add tests for patch generation.

5. Suggested fixes
Add unit tests.

6. Recommended next action
Run pytest for diff utilities.
""".strip()

    registry = create_default_registry()
    manager = create_default_subagent_manager(
        llm=fake_review_llm,
        tool_definitions=registry.list_definitions(),
        workspace_path=tmp_path,
    )

    runner = AgentGraphRunner(
        registry=registry,
        config={
            "workspace": {
                "path": str(tmp_path),
                "project_root": str(tmp_path),
            },
            "permissions": {
                "mode": "default",
            },
            "agent": {
                "max_iterations": 5,
                "max_context_messages": 20,
            },
        },
        runtime_objects={
            "subagent_manager": manager,
            "task_manager": manager.task_manager,
        },
    )

    state = await runner.arun(
        "用 SubAgent 审查 src/utils/diff.py",
        metadata={
            "subagent_manager": manager,
            "task_manager": manager.task_manager,
        },
    )

    last_message = state.get_last_message()

    assert last_message is not None
    assert "Tool `agent` result" in last_message.content
    assert "Reviewed src/pywork/utils/diff.py" in last_message.content