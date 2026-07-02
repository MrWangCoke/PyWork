from __future__ import annotations

import json
from pathlib import Path

import pytest

from pywork.subagents.manager import create_default_subagent_manager
from pywork.subagents.router import (
    LLMSubAgentRouter,
    SubAgentRouteKind,
    SubAgentRouterLLMError,
)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read file",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run bash",
        },
    },
    {
        "type": "function",
        "function": {
            "name": "powershell",
            "description": "Run PowerShell",
        },
    },
]


async def agent_llm(messages, *, tools=None, metadata=None):
    return {
        "content": (
            f"agent={metadata['agent_name']} "
            f"task={messages[-1]['content']}"
        ),
        "metadata": {},
    }


def make_manager(tmp_path: Path, router_llm=None):
    return create_default_subagent_manager(
        llm=router_llm or agent_llm,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )


@pytest.mark.asyncio
async def test_llm_router_routes_to_planner(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return json.dumps(
            {
                "agent_name": "planner",
                "task": "Plan the router implementation.",
                "reason": "The user asks for implementation planning.",
                "confidence": 0.92,
                "kind": "single",
                "pipeline": [],
                "missing_information": [],
                "metadata": {
                    "source": "test",
                },
            },
            ensure_ascii=False,
        )

    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    result = await router.route(
        "帮我规划 router.py 的实现",
    )

    assert result.agent_name == "planner"
    assert result.task == "Plan the router implementation."
    assert result.confidence == 0.92
    assert result.kind == SubAgentRouteKind.SINGLE
    assert result.confidence_label.value == "high"
    assert result.metadata["source"] == "test"


@pytest.mark.asyncio
async def test_llm_router_extracts_json_from_markdown(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return """
```json
{
  "agent_name": "debugger",
  "task": "Analyze the stuck runtime after tool_result_observed.",
  "reason": "The user reports a stuck runtime.",
  "confidence": 0.88,
  "kind": "single",
  "pipeline": [],
  "missing_information": [],
  "metadata": {}
}

"""
    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    result = await router.route(
        "文件创建成功后卡住不结束",
    )

    assert result.agent_name == "debugger"
    assert result.task == "Analyze the stuck runtime after tool_result_observed."
    assert result.confidence == 0.88


@pytest.mark.asyncio
async def test_llm_router_builds_task_request(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return json.dumps(
            {
                "agent_name": "reviewer",
                "task": "Review the permission gate changes.",
                "reason": "The task asks for code review.",
                "confidence": 0.81,
                "kind": "single",
                "pipeline": [],
                "missing_information": [],
                "metadata": {},
            },
            ensure_ascii=False,
        )

    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    request = await router.route_to_request(
        "审查一下权限系统改动",
        workspace_path=tmp_path,
    )

    assert request.agent_name == "reviewer"
    assert request.task == "Review the permission gate changes."
    assert request.workspace_path == tmp_path
    assert request.metadata["route"]["agent_name"] == "reviewer"


@pytest.mark.asyncio
async def test_llm_router_pipeline_to_requests(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return json.dumps(
            {
                "agent_name": "planner",
                "task": "Plan and verify the subagent router.",
                "reason": "This needs planning and verification.",
                "confidence": 0.9,
                "kind": "pipeline",
                "pipeline": [
                    {
                        "agent_name": "planner",
                        "task": "Plan the router implementation.",
                        "reason": "Planning comes first.",
                    },
                    {
                        "agent_name": "verifier",
                        "task": "Verify router tests.",
                        "reason": "Tests should confirm behavior.",
                    },
                ],
                "missing_information": [],
                "metadata": {},
            },
            ensure_ascii=False,
        )

    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    requests = await router.route_to_requests(
        "实现 router 并验证",
        workspace_path=tmp_path,
    )

    assert [
        request.agent_name
        for request in requests
    ] == [
        "planner",
        "verifier",
    ]

    assert requests[0].task == "Plan the router implementation."
    assert requests[1].task == "Verify router tests."


@pytest.mark.asyncio
async def test_llm_router_invalid_agent_falls_back_to_general(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return json.dumps(
            {
                "agent_name": "unknown_agent",
                "task": "Handle unknown route.",
                "reason": "Bad model output.",
                "confidence": 0.95,
                "kind": "single",
                "pipeline": [],
                "missing_information": [],
                "metadata": {},
            },
            ensure_ascii=False,
        )

    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    result = await router.route(
        "普通问题",
    )

    assert result.agent_name == "general"
    assert result.confidence <= 0.25
    assert result.metadata["used_fallback_agent"] is True


@pytest.mark.asyncio
async def test_llm_router_bad_json_falls_back_to_general(tmp_path: Path) -> None:
    async def router_llm(messages, *, tools=None, metadata=None):
        return "not json"

    manager = make_manager(tmp_path, router_llm=router_llm)
    router = LLMSubAgentRouter(manager=manager)

    result = await router.route(
        "随便问一个问题",
    )

    assert result.agent_name == "general"
    assert result.metadata["fallback"] is True
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_llm_router_missing_llm_raises(tmp_path: Path) -> None:
    manager = create_default_subagent_manager(
        llm=None,
        tool_definitions=TOOL_DEFINITIONS,
        workspace_path=tmp_path,
    )

    router = LLMSubAgentRouter(manager=manager)

    with pytest.raises(SubAgentRouterLLMError):
        await router.route("plan something")