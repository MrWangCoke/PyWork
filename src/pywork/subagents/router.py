from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    SubAgentContext,
    SubAgentLLMCallable,
    maybe_await,
    normalize_llm_response,
)
from pywork.subagents.manager import (
    SubAgentManager,
    SubAgentTaskRequest,
)


class SubAgentRouterError(Exception):
    """SubAgentRouter 基础异常。"""


class SubAgentRouterLLMError(SubAgentRouterError):
    """LLM Router 调用失败。"""


class SubAgentRouterParseError(SubAgentRouterError):
    """LLM Router 返回内容解析失败。"""


class SubAgentRouteKind(str, Enum):
    SINGLE = "single"
    PIPELINE = "pipeline"


class SubAgentRouteConfidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(slots=True, frozen=True)
class SubAgentRouterConfig:
    """
    LLM Router 配置。

    fallback_agent:
        LLM 返回非法 agent 或 JSON 解析失败时的兜底 Agent。

    allow_fallback_on_invalid_response:
        True 时，LLM 返回坏 JSON 不会直接抛异常，而是回退到 general。

    allow_pipeline:
        True 时允许 LLM 返回多 Agent 协作流程。

    max_parent_messages:
        给 Router LLM 的父上下文最多保留多少条。

    max_parent_message_chars:
        每条父上下文消息最多保留多少字符。
    """

    fallback_agent: str = "general"
    allow_fallback_on_invalid_response: bool = True
    allow_pipeline: bool = True
    max_parent_messages: int = 8
    max_parent_message_chars: int = 2000


@dataclass(slots=True)
class SubAgentRouteStep:
    agent_name: str
    task: str
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_task_request(
        self,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentTaskRequest:
        return SubAgentTaskRequest(
            agent_name=self.agent_name,
            task=self.task,
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata={
                "route_reason": self.reason,
                **dict(self.metadata),
                **dict(metadata or {}),
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class SubAgentRouteResult:
    agent_name: str
    task: str
    reason: str
    confidence: float
    kind: SubAgentRouteKind = SubAgentRouteKind.SINGLE
    pipeline: list[SubAgentRouteStep] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def confidence_label(self) -> SubAgentRouteConfidence:
        if self.confidence >= 0.8:
            return SubAgentRouteConfidence.HIGH

        if self.confidence >= 0.45:
            return SubAgentRouteConfidence.MEDIUM

        return SubAgentRouteConfidence.LOW

    @property
    def needs_pipeline(self) -> bool:
        return self.kind == SubAgentRouteKind.PIPELINE and bool(self.pipeline)

    def primary_step(self) -> SubAgentRouteStep:
        return SubAgentRouteStep(
            agent_name=self.agent_name,
            task=self.task,
            reason=self.reason,
            metadata={
                "confidence": self.confidence,
                "confidence_label": self.confidence_label.value,
                **dict(self.metadata),
            },
        )

    def to_task_request(
        self,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentTaskRequest:
        return self.primary_step().to_task_request(
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata=metadata,
        )

    def to_task_requests(
        self,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SubAgentTaskRequest]:
        steps = self.pipeline if self.needs_pipeline else [self.primary_step()]

        return [
            step.to_task_request(
                context=context,
                workspace_path=workspace_path,
                parent_messages=parent_messages,
                metadata=metadata,
            )
            for step in steps
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task": self.task,
            "reason": self.reason,
            "confidence": self.confidence,
            "confidence_label": self.confidence_label.value,
            "kind": self.kind.value,
            "pipeline": [
                step.to_dict()
                for step in self.pipeline
            ],
            "missing_information": list(self.missing_information),
            "metadata": dict(self.metadata),
        }


ROUTER_SYSTEM_PROMPT = """
You are PyWork's real LLM subagent router.

Your job is to choose the best subagent for the user's task.

Available subagent roles:
- general: general development analysis, reading/searching code, answering general project questions.
- planner: task decomposition, implementation planning, architecture planning, step-by-step roadmap.
- reviewer: code review, safety review, permission review, maintainability review, test coverage review.
- debugger: analyze errors, tracebacks, logs, failing tests, stuck runtime, unexpected behavior.
- verifier: decide and run verification checks, summarize test results, stdout, stderr, exit_code.

Important routing rules:
- If the user asks how to implement something or wants a plan, choose planner.
- If the user asks to inspect code quality, risks, bugs, permission bypasses, or test coverage, choose reviewer.
- If the user provides errors, failing tests, logs, traceback, stuck behavior, or asks why something failed, choose debugger.
- If the user asks to run tests, verify a change, or confirm whether implementation works, choose verifier.
- If the task is broad, ordinary, or unclear, choose general.
- If a multi-step workflow is clearly useful, return kind="pipeline" and include a pipeline.
- Do not invent unavailable agents.
- Do not execute anything. You only route.

Return JSON only. Do not include markdown.

JSON schema:
{
  "agent_name": "general | planner | reviewer | debugger | verifier",
  "task": "rewritten task for the selected subagent",
  "reason": "short reason for the routing decision",
  "confidence": 0.0,
  "kind": "single | pipeline",
  "pipeline": [
    {
      "agent_name": "planner",
      "task": "task for this subagent",
      "reason": "why this step is needed"
    }
  ],
  "missing_information": [],
  "metadata": {}
}
""".strip()


def truncate_text(
    text: str,
    max_chars: int,
) -> str:
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 20] + "\n...[truncated]..."


def message_to_router_dict(
    message: Any,
    *,
    max_chars: int,
) -> dict[str, Any]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        name = message.get("name")
    else:
        role = str(getattr(message, "role", "user"))
        content = str(getattr(message, "content", ""))
        name = getattr(message, "name", None)

    data: dict[str, Any] = {
        "role": role,
        "content": truncate_text(content, max_chars),
    }

    if name:
        data["name"] = str(name)

    return data


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()

    if not stripped:
        raise SubAgentRouterParseError("empty router response")

    fenced = re.search(
        r"```(?:json)?\s*(.*?)```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced:
        stripped = fenced.group(1).strip()

    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    decoder = json.JSONDecoder()

    for index, char in enumerate(stripped):
        if char != "{":
            continue

        try:
            _, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue

        return stripped[index : index + end]

    raise SubAgentRouterParseError(
        "router response does not contain a JSON object"
    )


def parse_router_json(text: str) -> dict[str, Any]:
    json_text = extract_json_object_text(text)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise SubAgentRouterParseError(str(exc)) from exc

    if not isinstance(data, dict):
        raise SubAgentRouterParseError("router JSON must be an object")

    return data


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        return 0.0

    if confidence < 0:
        return 0.0

    if confidence > 1:
        if confidence <= 100:
            return confidence / 100

        return 1.0

    return confidence


def normalize_route_kind(value: Any) -> SubAgentRouteKind:
    normalized = str(value or "").strip().lower()

    if normalized == "pipeline":
        return SubAgentRouteKind.PIPELINE

    return SubAgentRouteKind.SINGLE


class LLMSubAgentRouter:
    """
    真实 LLM Router。

    它不靠关键词规则决定 Agent，而是调用传入的 llm callable。
    推荐使用 manager.llm，也就是和主 Runtime 相同的真实 LLM 客户端。
    """

    def __init__(
        self,
        *,
        manager: SubAgentManager,
        llm: SubAgentLLMCallable | None = None,
        config: SubAgentRouterConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.manager = manager
        self.llm = llm
        self.config = config or SubAgentRouterConfig()
        self.metadata = metadata or {}

    def get_llm(self) -> SubAgentLLMCallable:
        llm = self.llm or self.manager.llm

        if llm is None:
            raise SubAgentRouterLLMError(
                "LLMSubAgentRouter requires a real llm callable"
            )

        return llm

    def available_agents_payload(self) -> list[dict[str, Any]]:
        return self.manager.list_agents()

    def build_router_messages(
        self,
        *,
        task: str,
        context: SubAgentContext | None = None,
        parent_messages: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        parent = list(parent_messages or [])

        if context is not None and context.parent_messages:
            parent = list(context.parent_messages) + parent

        if self.config.max_parent_messages > 0:
            parent = parent[-self.config.max_parent_messages :]

        parent_payload = [
            message_to_router_dict(
                message,
                max_chars=self.config.max_parent_message_chars,
            )
            for message in parent
        ]

        user_payload = {
            "task": task,
            "available_agents": self.available_agents_payload(),
            "parent_context": parent_payload,
            "workspace_path": str(
                context.resolved_workspace_path()
                if context is not None
                else self.manager.workspace_path
            ),
            "metadata": {
                **self.metadata,
                **dict(metadata or {}),
            },
        }

        return [
            {
                "role": "system",
                "content": ROUTER_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    def resolve_agent_name_or_fallback(
        self,
        value: Any,
    ) -> tuple[str, bool]:
        requested = str(value or "").strip()

        try:
            return self.manager.resolve_agent_name(requested), False
        except Exception:
            return self.manager.resolve_agent_name(self.config.fallback_agent), True

    def normalize_pipeline(
        self,
        raw_pipeline: Any,
        *,
        fallback_task: str,
    ) -> tuple[list[SubAgentRouteStep], list[str]]:
        if not isinstance(raw_pipeline, list):
            return [], []

        steps: list[SubAgentRouteStep] = []
        warnings: list[str] = []

        for index, raw_step in enumerate(raw_pipeline):
            if not isinstance(raw_step, Mapping):
                warnings.append(f"pipeline[{index}] is not an object")
                continue

            agent_name, used_fallback = self.resolve_agent_name_or_fallback(
                raw_step.get("agent_name")
            )

            if used_fallback:
                warnings.append(
                    f"pipeline[{index}] used fallback agent {agent_name}"
                )

            step_task = str(
                raw_step.get("task")
                or fallback_task
            ).strip()

            reason = str(raw_step.get("reason") or "").strip()

            metadata = raw_step.get("metadata")
            if not isinstance(metadata, dict):
                metadata = {}

            steps.append(
                SubAgentRouteStep(
                    agent_name=agent_name,
                    task=step_task,
                    reason=reason,
                    metadata=dict(metadata),
                )
            )

        return steps, warnings

    def build_fallback_result(
        self,
        *,
        task: str,
        raw_response: str,
        error: str,
    ) -> SubAgentRouteResult:
        agent_name = self.manager.resolve_agent_name(self.config.fallback_agent)

        return SubAgentRouteResult(
            agent_name=agent_name,
            task=task,
            reason=f"router fallback because LLM response was invalid: {error}",
            confidence=0.0,
            kind=SubAgentRouteKind.SINGLE,
            pipeline=[],
            raw_response=raw_response,
            metadata={
                "fallback": True,
                "error": error,
            },
        )

    def build_result_from_payload(
        self,
        *,
        task: str,
        payload: Mapping[str, Any],
        raw_response: str,
    ) -> SubAgentRouteResult:
        agent_name, used_fallback = self.resolve_agent_name_or_fallback(
            payload.get("agent_name")
        )

        routed_task = str(
            payload.get("task")
            or task
        ).strip()

        reason = str(
            payload.get("reason")
            or "LLM router selected this subagent"
        ).strip()

        confidence = normalize_confidence(
            payload.get("confidence", 0.0)
        )

        kind = normalize_route_kind(
            payload.get("kind")
        )

        pipeline, pipeline_warnings = self.normalize_pipeline(
            payload.get("pipeline"),
            fallback_task=routed_task,
        )

        if kind == SubAgentRouteKind.PIPELINE and not self.config.allow_pipeline:
            kind = SubAgentRouteKind.SINGLE
            pipeline = []
            pipeline_warnings.append("pipeline disabled by router config")

        if kind == SubAgentRouteKind.PIPELINE and not pipeline:
            kind = SubAgentRouteKind.SINGLE
            pipeline_warnings.append("pipeline requested but no valid steps found")

        missing_information = payload.get("missing_information")
        if not isinstance(missing_information, list):
            missing_information = []

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        metadata = {
            **dict(metadata),
            "router": "LLMSubAgentRouter",
            "used_fallback_agent": used_fallback,
            "pipeline_warnings": pipeline_warnings,
        }

        if used_fallback:
            confidence = min(confidence, 0.25)

        return SubAgentRouteResult(
            agent_name=agent_name,
            task=routed_task,
            reason=reason,
            confidence=confidence,
            kind=kind,
            pipeline=pipeline,
            missing_information=[
                str(item)
                for item in missing_information
            ],
            raw_response=raw_response,
            metadata=metadata,
        )

    async def route(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        parent_messages: Sequence[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentRouteResult:
        llm = self.get_llm()

        messages = self.build_router_messages(
            task=task,
            context=context,
            parent_messages=parent_messages,
            metadata=metadata,
        )

        try:
            response = await maybe_await(
                llm(
                    messages,
                    tools=None,
                    metadata={
                        "component": "subagent_router",
                        "router": "LLMSubAgentRouter",
                        "available_agents": self.available_agents_payload(),
                        **self.metadata,
                        **dict(metadata or {}),
                    },
                )
            )
        except Exception as exc:
            raise SubAgentRouterLLMError(str(exc)) from exc

        llm_response = normalize_llm_response(response)
        raw_response = llm_response.content

        try:
            payload = parse_router_json(raw_response)
        except SubAgentRouterParseError as exc:
            if not self.config.allow_fallback_on_invalid_response:
                raise

            return self.build_fallback_result(
                task=task,
                raw_response=raw_response,
                error=str(exc),
            )

        return self.build_result_from_payload(
            task=task,
            payload=payload,
            raw_response=raw_response,
        )

    async def route_to_request(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentTaskRequest:
        result = await self.route(
            task,
            context=context,
            parent_messages=parent_messages,
            metadata=metadata,
        )

        return result.to_task_request(
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata={
                "route": result.to_dict(),
                **dict(metadata or {}),
            },
        )

    async def route_to_requests(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> list[SubAgentTaskRequest]:
        result = await self.route(
            task,
            context=context,
            parent_messages=parent_messages,
            metadata=metadata,
        )

        return result.to_task_requests(
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata={
                "route": result.to_dict(),
                **dict(metadata or {}),
            },
        )

    async def route_and_run(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        workspace_path: str | Path | None = None,
        parent_messages: list[Any] | None = None,
        metadata: dict[str, Any] | None = None,
        concurrent_pipeline: bool = False,
    ):
        route_result = await self.route(
            task,
            context=context,
            parent_messages=parent_messages,
            metadata=metadata,
        )

        requests = route_result.to_task_requests(
            context=context,
            workspace_path=workspace_path,
            parent_messages=parent_messages,
            metadata={
                "route": route_result.to_dict(),
                **dict(metadata or {}),
            },
        )

        if route_result.needs_pipeline:
            results = await self.manager.run_many(
                requests,
                concurrent=concurrent_pipeline,
            )
            return route_result, results

        result = await self.manager.run_agent(
            requests[0].agent_name,
            requests[0].task,
            context=requests[0].context,
            workspace_path=requests[0].workspace_path,
            parent_messages=requests[0].parent_messages,
            working_memory=requests[0].working_memory,
            metadata=requests[0].metadata,
            tool_scope=requests[0].tool_scope,
            max_steps=requests[0].max_steps,
            llm=requests[0].llm,
            run_id=requests[0].run_id,
        )

        return route_result, result


def create_llm_subagent_router(
    *,
    manager: SubAgentManager,
    llm: SubAgentLLMCallable | None = None,
    config: SubAgentRouterConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> LLMSubAgentRouter:
    return LLMSubAgentRouter(
        manager=manager,
        llm=llm,
        config=config,
        metadata=metadata,
    )