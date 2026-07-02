from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import SubAgentLLMCallable
from pywork.subagents.manager import SubAgentManager
from pywork.teams.mailbox import MailboxMessage, safe_jsonable
from pywork.teams.team import (
    Team,
    TeamSharedTask,
    TeamTaskPriority,
    TeamTaskStatus,
    create_team,
    normalize_team_task_priority,
)
from pywork.teams.teammate import (
    TeammateExecutionMode,
    TeammateMessageHandleResult,
)


class SwarmError(Exception):
    """Swarm 基础异常。"""


class SwarmPlanError(SwarmError):
    """Swarm 任务规划异常。"""


class SwarmCancelledError(SwarmError):
    """Swarm 被取消。"""


class SwarmStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SwarmStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class SwarmEventType(str, Enum):
    STARTED = "started"
    PLAN_CREATED = "plan_created"
    TASK_CREATED = "task_created"
    TASK_DISPATCHED = "task_dispatched"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


SwarmEventHandler = Callable[["SwarmEvent"], Any | Awaitable[Any]]


def now_timestamp() -> float:
    return time.time()


def new_swarm_id(prefix: str = "swarm") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_swarm_run_id(prefix: str = "swarm_run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_swarm_step_id(index: int) -> str:
    return f"swarm_step_{index:02d}"


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def normalize_swarm_strategy(value: SwarmStrategy | str | None) -> SwarmStrategy:
    if isinstance(value, SwarmStrategy):
        return value

    text = str(value or SwarmStrategy.SEQUENTIAL.value).strip().lower()

    try:
        return SwarmStrategy(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in SwarmStrategy)
        raise SwarmPlanError(
            f"Invalid swarm strategy {value!r}. Valid strategies: {valid}"
        ) from exc


def coerce_llm_content(response: Any) -> str:
    if isinstance(response, str):
        return response

    if isinstance(response, Mapping):
        return str(response.get("content") or response.get("text") or "")

    return str(getattr(response, "content", "") or response)


def extract_json_object_text(text: str) -> str:
    stripped = text.strip()

    if not stripped:
        raise SwarmPlanError("empty planner response")

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

    raise SwarmPlanError("planner response does not contain JSON object")


def parse_json_object(text: str) -> dict[str, Any]:
    json_text = extract_json_object_text(text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise SwarmPlanError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise SwarmPlanError("planner JSON must be an object")

    return payload


def infer_team_role_from_task(task: str) -> str:
    text = task.lower()

    if any(word in text for word in ("报错", "错误", "异常", "失败", "traceback", "debug", "failed", "error")):
        return "debugger"

    if any(word in text for word in ("测试", "验证", "pytest", "verify", "test", "运行测试")):
        return "verifier"

    if any(word in text for word in ("审查", "评审", "review", "风险", "安全", "权限", "coverage")):
        return "reviewer"

    if any(word in text for word in ("计划", "规划", "设计", "拆解", "方案", "plan", "design", "roadmap")):
        return "planner"

    return "general"


@dataclass(slots=True)
class SwarmEvent:
    event_type: SwarmEventType
    swarm_id: str
    run_id: str | None = None
    step_id: str | None = None
    team_task_id: str | None = None
    teammate_id: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "swarm_id": self.swarm_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "team_task_id": self.team_task_id,
            "teammate_id": self.teammate_id,
            "message": self.message,
            "metadata": safe_jsonable(self.metadata),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class SwarmTaskStep:
    step_id: str
    title: str
    description: str = ""
    role: str | None = None
    assigned_to: str | None = None
    priority: TeamTaskPriority = TeamTaskPriority.NORMAL
    depends_on: list[str] = field(default_factory=list)
    task_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def task_text(self) -> str:
        return self.description or self.title

    @property
    def resolved_role(self) -> str:
        return self.role or infer_team_role_from_task(self.task_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "title": self.title,
            "description": self.description,
            "role": self.resolved_role,
            "assigned_to": self.assigned_to,
            "priority": self.priority.value,
            "depends_on": list(self.depends_on),
            "task_id": self.task_id,
            "metadata": safe_jsonable(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        index: int,
    ) -> SwarmTaskStep:
        step_id = str(data.get("step_id") or data.get("id") or new_swarm_step_id(index)).strip()
        description = str(data.get("description") or data.get("task") or "").strip()
        title = str(data.get("title") or "").strip()

        if not title and description:
            title = description.splitlines()[0][:80]

        if not title:
            raise SwarmPlanError(f"step {step_id!r} has empty title")

        raw_depends_on = data.get("depends_on") or data.get("dependencies") or []

        if not isinstance(raw_depends_on, list):
            raw_depends_on = []

        metadata = data.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            step_id=step_id,
            title=title,
            description=description,
            role=(
                str(data["role"])
                if data.get("role") is not None
                else None
            ),
            assigned_to=(
                str(data["assigned_to"])
                if data.get("assigned_to") is not None
                else None
            ),
            priority=normalize_team_task_priority(data.get("priority")),
            depends_on=[str(item) for item in raw_depends_on],
            task_id=(
                str(data["task_id"])
                if data.get("task_id") is not None
                else None
            ),
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class SwarmPlan:
    task: str
    steps: list[SwarmTaskStep]
    strategy: SwarmStrategy = SwarmStrategy.SEQUENTIAL
    summary: str = ""
    raw_response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_dependencies(self) -> bool:
        return any(step.depends_on for step in self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "strategy": self.strategy.value,
            "summary": self.summary,
            "steps": [
                step.to_dict()
                for step in self.steps
            ],
            "raw_response": self.raw_response,
            "metadata": safe_jsonable(self.metadata),
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        fallback_task: str,
        raw_response: str = "",
    ) -> SwarmPlan:
        raw_steps = payload.get("steps") or payload.get("tasks") or []

        if not isinstance(raw_steps, list):
            raise SwarmPlanError("planner JSON field `steps` must be an array")

        steps: list[SwarmTaskStep] = []

        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, Mapping):
                raise SwarmPlanError(f"step {index} must be an object")

            steps.append(
                SwarmTaskStep.from_mapping(
                    item,
                    index=index,
                )
            )

        if not steps:
            raise SwarmPlanError("planner returned no valid steps")

        metadata = payload.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            task=str(payload.get("task") or fallback_task),
            steps=steps,
            strategy=normalize_swarm_strategy(payload.get("strategy")),
            summary=str(payload.get("summary") or ""),
            raw_response=raw_response,
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class SwarmRunRequest:
    task: str
    workspace_path: str | Path = "."
    metadata: dict[str, Any] = field(default_factory=dict)

    plan: SwarmPlan | None = None
    steps: Sequence[SwarmTaskStep | Mapping[str, Any]] | None = None
    strategy: SwarmStrategy | str | None = None

    use_llm_planning: bool = True
    allow_fallback_plan: bool = True
    clear_existing_tasks: bool = False

    assignment_strategy: str | None = None
    teammate_execution_mode: TeammateExecutionMode | str = TeammateExecutionMode.DIRECT

    max_concurrency: int | None = None
    step_timeout_seconds: float | None = 5.0
    poll_timeout_seconds: float = 0.1
    fail_fast: bool = False


@dataclass(slots=True)
class SwarmTaskExecution:
    step_id: str
    team_task_id: str
    assigned_to: str | None
    status: TeamTaskStatus
    success: bool
    handle_result: TeammateMessageHandleResult | None = None
    result_messages: list[MailboxMessage] = field(default_factory=list)
    shared_task: TeamSharedTask | None = None
    error: str | None = None
    started_at: float = field(default_factory=now_timestamp)
    finished_at: float | None = None

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at is None:
            return None

        return int((self.finished_at - self.started_at) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "team_task_id": self.team_task_id,
            "assigned_to": self.assigned_to,
            "status": self.status.value,
            "success": self.success,
            "handle_result": (
                self.handle_result.to_dict()
                if self.handle_result is not None
                else None
            ),
            "result_messages": [
                message.to_dict()
                for message in self.result_messages
            ],
            "shared_task": (
                self.shared_task.to_dict()
                if self.shared_task is not None
                else None
            ),
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class SwarmRunResult:
    swarm_id: str
    run_id: str
    task: str
    status: SwarmStatus
    success: bool
    plan: SwarmPlan
    executions: list[SwarmTaskExecution]
    summary: str
    error: str | None = None
    started_at: float = field(default_factory=now_timestamp)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> int | None:
        if self.finished_at is None:
            return None

        return int((self.finished_at - self.started_at) * 1000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "swarm_id": self.swarm_id,
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status.value,
            "success": self.success,
            "plan": self.plan.to_dict(),
            "executions": [
                execution.to_dict()
                for execution in self.executions
            ],
            "summary": self.summary,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class SwarmConfig:
    max_concurrency: int = 4
    default_strategy: SwarmStrategy = SwarmStrategy.SEQUENTIAL
    assignment_strategy: str = "round_robin"
    include_raw_plan_response: bool = True
    allow_fallback_plan: bool = True


SWARM_PLANNER_PROMPT = """
You are PyWork's Swarm planner.

Your job is to decompose a user task into shared team tasks.

Available teammate roles:
- planner: planning and design.
- reviewer: code review, safety review, permission review.
- debugger: diagnose errors, failing tests, runtime issues.
- verifier: run or decide verification checks.
- general: ordinary implementation or analysis.

Return JSON only. Do not include markdown.

JSON schema:
{
  "task": "original or rewritten task",
  "strategy": "sequential | parallel",
  "summary": "short explanation",
  "steps": [
    {
      "step_id": "swarm_step_01",
      "title": "short task title",
      "description": "specific task for the teammate",
      "role": "planner | reviewer | debugger | verifier | general",
      "assigned_to": null,
      "priority": "low | normal | high | urgent",
      "depends_on": [],
      "metadata": {}
    }
  ],
  "metadata": {}
}

Planning rules:
- Use sequential when tasks depend on each other.
- Use parallel only when tasks are independent.
- Keep steps focused and actionable.
- Prefer 1-5 steps.
""".strip()


class Swarm:
    """
    Team / Swarm 编排器。

    职责：
    - 规划团队任务
    - 写入 Team.shared_task_list
    - 分配任务给 Teammate
    - 驱动 Teammate 处理 mailbox 消息
    - 收集团队结果
    - 汇总最终结果
    """

    def __init__(
        self,
        *,
        swarm_id: str | None = None,
        team: Team | None = None,
        manager: SubAgentManager | None = None,
        llm: SubAgentLLMCallable | None = None,
        planning_llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path = ".",
        config: SwarmConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.swarm_id = swarm_id or new_swarm_id()
        self.workspace_path = Path(workspace_path)
        self.config = config or SwarmConfig()
        self.metadata = metadata or {}

        self.team = team or create_team(
            team_id=f"{self.swarm_id}_team",
            name=f"{self.swarm_id} Team",
            manager=manager,
            llm=llm,
            tool_definitions=tool_definitions,
            workspace_path=workspace_path,
            metadata={
                "owner": "Swarm",
                "swarm_id": self.swarm_id,
                **self.metadata,
            },
        )

        self.planning_llm = planning_llm or llm
        self.status = SwarmStatus.IDLE
        self.current_run_id: str | None = None
        self.active_task_ids: set[str] = set()
        self.last_result: SwarmRunResult | None = None
        self._cancel_requested = False
        self._event_handlers: list[SwarmEventHandler] = []

    def add_event_handler(
        self,
        handler: SwarmEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: SwarmEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: SwarmEvent,
    ) -> None:
        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    def build_planning_messages(
        self,
        request: SwarmRunRequest,
    ) -> list[dict[str, Any]]:
        payload = {
            "task": request.task,
            "team": self.team.to_dict(),
            "workspace_path": str(request.workspace_path),
            "preferred_strategy": (
                request.strategy.value
                if isinstance(request.strategy, SwarmStrategy)
                else request.strategy
            ),
            "metadata": safe_jsonable(request.metadata),
        }

        return [
            {
                "role": "system",
                "content": SWARM_PLANNER_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

    async def create_plan(
        self,
        request: SwarmRunRequest,
    ) -> SwarmPlan:
        if request.plan is not None:
            return request.plan

        if request.steps is not None:
            steps: list[SwarmTaskStep] = []

            for index, item in enumerate(request.steps, start=1):
                if isinstance(item, SwarmTaskStep):
                    steps.append(item)
                elif isinstance(item, Mapping):
                    steps.append(
                        SwarmTaskStep.from_mapping(
                            item,
                            index=index,
                        )
                    )
                else:
                    raise SwarmPlanError("request.steps items must be SwarmTaskStep or mapping")

            return SwarmPlan(
                task=request.task,
                steps=steps,
                strategy=normalize_swarm_strategy(
                    request.strategy or self.config.default_strategy
                ),
                summary="Plan provided by caller.",
                metadata={
                    "source": "caller_steps",
                },
            )

        if request.use_llm_planning and self.planning_llm is not None:
            try:
                response = await maybe_await(
                    self.planning_llm(
                        self.build_planning_messages(request),
                        tools=None,
                        metadata={
                            "component": "swarm_planner",
                            "swarm_id": self.swarm_id,
                            "team_id": self.team.team_id,
                            **self.metadata,
                            **dict(request.metadata or {}),
                        },
                    )
                )

                raw = coerce_llm_content(response)
                payload = parse_json_object(raw)

                plan = SwarmPlan.from_payload(
                    payload,
                    fallback_task=request.task,
                    raw_response=raw if self.config.include_raw_plan_response else "",
                )

                if request.strategy is not None:
                    plan.strategy = normalize_swarm_strategy(request.strategy)

                return plan

            except Exception:
                if not request.allow_fallback_plan and not self.config.allow_fallback_plan:
                    raise

        return self.create_fallback_plan(request)

    def create_fallback_plan(
        self,
        request: SwarmRunRequest,
    ) -> SwarmPlan:
        role = infer_team_role_from_task(request.task)

        return SwarmPlan(
            task=request.task,
            strategy=normalize_swarm_strategy(
                request.strategy or self.config.default_strategy
            ),
            summary="Fallback single-step swarm plan.",
            steps=[
                SwarmTaskStep(
                    step_id="swarm_step_01",
                    title=request.task.splitlines()[0][:80],
                    description=request.task,
                    role=role,
                    priority=TeamTaskPriority.NORMAL,
                    metadata={
                        "fallback": True,
                    },
                )
            ],
            metadata={
                "fallback": True,
            },
        )

    def materialize_plan_tasks(
        self,
        *,
        run_id: str,
        plan: SwarmPlan,
        request: SwarmRunRequest,
    ) -> dict[str, TeamSharedTask]:
        if request.clear_existing_tasks:
            self.team.clear_shared_tasks(include_active=True)

        tasks_by_step: dict[str, TeamSharedTask] = {}

        for step in plan.steps:
            task_id = step.task_id or f"{run_id}_{step.step_id}"

            task = self.team.create_shared_task(
                step.title,
                description=step.description,
                role=step.resolved_role,
                assigned_to=None,
                priority=step.priority,
                payload={
                    "swarm_step": step.to_dict(),
                    "swarm_run_id": run_id,
                },
                metadata={
                    "swarm_id": self.swarm_id,
                    "swarm_run_id": run_id,
                    "step_id": step.step_id,
                    **dict(step.metadata or {}),
                },
                task_id=task_id,
            )

            tasks_by_step[step.step_id] = task

        return tasks_by_step

    async def run(
        self,
        request: SwarmRunRequest,
    ) -> SwarmRunResult:
        if self.status in {
            SwarmStatus.PLANNING,
            SwarmStatus.RUNNING,
        }:
            raise SwarmError("Swarm is already running")

        run_id = new_swarm_run_id()
        started_at = now_timestamp()

        self.current_run_id = run_id
        self._cancel_requested = False
        self.status = SwarmStatus.PLANNING

        await self.emit_event(
            SwarmEvent(
                event_type=SwarmEventType.STARTED,
                swarm_id=self.swarm_id,
                run_id=run_id,
                message="swarm started",
            )
        )

        try:
            plan = await self.create_plan(request)

            await self.emit_event(
                SwarmEvent(
                    event_type=SwarmEventType.PLAN_CREATED,
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    message="swarm plan created",
                    metadata={
                        "plan": plan.to_dict(),
                    },
                )
            )

            tasks_by_step = self.materialize_plan_tasks(
                run_id=run_id,
                plan=plan,
                request=request,
            )

            for step_id, task in tasks_by_step.items():
                await self.emit_event(
                    SwarmEvent(
                        event_type=SwarmEventType.TASK_CREATED,
                        swarm_id=self.swarm_id,
                        run_id=run_id,
                        step_id=step_id,
                        team_task_id=task.task_id,
                        message=f"team task created: {task.task_id}",
                    )
                )

            self.status = SwarmStatus.RUNNING

            if plan.strategy == SwarmStrategy.PARALLEL and not plan.has_dependencies:
                executions = await self._run_parallel(
                    run_id=run_id,
                    plan=plan,
                    request=request,
                    tasks_by_step=tasks_by_step,
                )
            else:
                executions = await self._run_sequential(
                    run_id=run_id,
                    plan=plan,
                    request=request,
                    tasks_by_step=tasks_by_step,
                )

            finished_at = now_timestamp()

            if self._cancel_requested:
                status = SwarmStatus.CANCELLED
                success = False
            else:
                success = (
                    len(executions) == len(plan.steps)
                    and all(execution.success for execution in executions)
                )
                status = SwarmStatus.COMPLETED if success else SwarmStatus.FAILED

            summary = self.summarize_results(
                plan=plan,
                executions=executions,
                status=status,
            )

            result = SwarmRunResult(
                swarm_id=self.swarm_id,
                run_id=run_id,
                task=request.task,
                status=status,
                success=success,
                plan=plan,
                executions=executions,
                summary=summary,
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "team_id": self.team.team_id,
                    "strategy": plan.strategy.value,
                    "step_count": len(plan.steps),
                    "execution_count": len(executions),
                },
            )

            self.status = status
            self.last_result = result

            await self.emit_event(
                SwarmEvent(
                    event_type=(
                        SwarmEventType.CANCELLED
                        if status == SwarmStatus.CANCELLED
                        else SwarmEventType.COMPLETED
                        if status == SwarmStatus.COMPLETED
                        else SwarmEventType.FAILED
                    ),
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    message=f"swarm {status.value}",
                    metadata={
                        "result": result.to_dict(),
                    },
                )
            )

            return result

        except Exception as exc:
            finished_at = now_timestamp()
            self.status = SwarmStatus.FAILED

            plan = self.create_fallback_plan(request)

            result = SwarmRunResult(
                swarm_id=self.swarm_id,
                run_id=run_id,
                task=request.task,
                status=SwarmStatus.FAILED,
                success=False,
                plan=plan,
                executions=[],
                summary=f"Swarm failed: {exc}",
                error=str(exc),
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "error_type": type(exc).__name__,
                },
            )

            self.last_result = result

            await self.emit_event(
                SwarmEvent(
                    event_type=SwarmEventType.FAILED,
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    message=str(exc),
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )

            return result

        finally:
            self.current_run_id = None
            self.active_task_ids.clear()
            self._cancel_requested = False

            if self.status in {
                SwarmStatus.COMPLETED,
                SwarmStatus.FAILED,
                SwarmStatus.CANCELLED,
            }:
                self.status = SwarmStatus.IDLE

    async def _run_sequential(
        self,
        *,
        run_id: str,
        plan: SwarmPlan,
        request: SwarmRunRequest,
        tasks_by_step: Mapping[str, TeamSharedTask],
    ) -> list[SwarmTaskExecution]:
        executions: list[SwarmTaskExecution] = []
        executions_by_step: dict[str, SwarmTaskExecution] = {}

        for step in plan.steps:
            if self._cancel_requested:
                break

            failed_dependencies = [
                dependency
                for dependency in step.depends_on
                if dependency in executions_by_step
                and not executions_by_step[dependency].success
            ]

            if failed_dependencies:
                task = tasks_by_step[step.step_id]
                task.mark_cancelled(
                    reason=f"dependency failed: {', '.join(failed_dependencies)}"
                )

                execution = SwarmTaskExecution(
                    step_id=step.step_id,
                    team_task_id=task.task_id,
                    assigned_to=task.assigned_to,
                    status=task.status,
                    success=False,
                    shared_task=task,
                    error=task.error,
                    finished_at=now_timestamp(),
                )

                executions.append(execution)
                executions_by_step[step.step_id] = execution

                if request.fail_fast:
                    break

                continue

            execution = await self._dispatch_and_process_step(
                run_id=run_id,
                step=step,
                request=request,
                task=tasks_by_step[step.step_id],
            )

            executions.append(execution)
            executions_by_step[step.step_id] = execution

            if request.fail_fast and not execution.success:
                break

        return executions

    async def _run_parallel(
        self,
        *,
        run_id: str,
        plan: SwarmPlan,
        request: SwarmRunRequest,
        tasks_by_step: Mapping[str, TeamSharedTask],
    ) -> list[SwarmTaskExecution]:
        concurrency = request.max_concurrency or self.config.max_concurrency
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_one(step: SwarmTaskStep) -> SwarmTaskExecution:
            async with semaphore:
                return await self._dispatch_and_process_step(
                    run_id=run_id,
                    step=step,
                    request=request,
                    task=tasks_by_step[step.step_id],
                )

        executions = await asyncio.gather(
            *[
                run_one(step)
                for step in plan.steps
            ]
        )

        return list(executions)

    def resolve_dispatch_role(
        self,
        requested_role: str | None,
    ) -> str | None:
        """
        Resolve the actual role used for Team assignment.

        If the exact requested role is unavailable, fall back to a general
        teammate when possible. This keeps Swarm robust when the planner asks
        for a specialist role that the current Team roster does not have.
        """
        if requested_role and self.team.roster.available_members(role=requested_role):
            return requested_role

        if requested_role and requested_role != "general":
            if self.team.roster.available_members(role="general"):
                return "general"

        if self.team.roster.available_members(role=None):
            return None

        return requested_role

    async def _dispatch_and_process_step(
        self,
        *,
        run_id: str,
        step: SwarmTaskStep,
        request: SwarmRunRequest,
        task: TeamSharedTask,
    ) -> SwarmTaskExecution:
        started_at = now_timestamp()
        self.active_task_ids.add(task.task_id)

        handle_result: TeammateMessageHandleResult | None = None
        result_messages: list[MailboxMessage] = []

        try:

            if self._cancel_requested:
                task.mark_cancelled("swarm cancelled before dispatch")

                return SwarmTaskExecution(
                    step_id=step.step_id,
                    team_task_id=task.task_id,
                    assigned_to=task.assigned_to,
                    status=task.status,
                    success=False,
                    shared_task=task,
                    error=task.error,
                    started_at=started_at,
                    finished_at=now_timestamp(),
                )
            
            dispatch_role = self.resolve_dispatch_role(step.resolved_role)

            message = await self.team.dispatch_shared_task(
                task.task_id,
                teammate_id=step.assigned_to,
                role=dispatch_role,
                strategy=request.assignment_strategy or self.config.assignment_strategy,
                metadata={
                    "swarm_id": self.swarm_id,
                    "swarm_run_id": run_id,
                    "step_id": step.step_id,
                    "requested_role": step.resolved_role,
                    "dispatch_role": dispatch_role,
                },
            )

            await self.emit_event(
                SwarmEvent(
                    event_type=SwarmEventType.TASK_DISPATCHED,
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    step_id=step.step_id,
                    team_task_id=task.task_id,
                    teammate_id=message.recipient_id,
                    message=f"team task dispatched: {task.task_id}",
                    metadata={
                        "mailbox_message_id": message.message_id,
                    },
                )
            )

            self.team.mark_task_running(task.task_id)

            if self._cancel_requested:
                task.mark_cancelled("swarm cancelled before teammate execution")

                return SwarmTaskExecution(
                    step_id=step.step_id,
                    team_task_id=task.task_id,
                    assigned_to=task.assigned_to,
                    status=task.status,
                    success=False,
                    shared_task=task,
                    error=task.error,
                    started_at=started_at,
                    finished_at=now_timestamp(),
                )

            teammate = self.team.require_teammate(message.recipient_id)

            handle_result = await teammate.process_next_message(
                timeout=request.step_timeout_seconds,
                execution_mode=request.teammate_execution_mode,
            )

            result_messages = await self.team.collect_result_messages(
                timeout=request.poll_timeout_seconds,
            )

            updated_task = self.team.require_shared_task(task.task_id)

            if not updated_task.is_terminal and handle_result.task_result is not None:
                if handle_result.task_result.success:
                    updated_task.mark_succeeded(
                        {
                            "task_result": handle_result.task_result.to_dict(),
                            "source": "handle_result_fallback",
                        }
                    )
                else:
                    updated_task.mark_failed(
                        handle_result.task_result.error or "teammate task failed",
                        result={
                            "task_result": handle_result.task_result.to_dict(),
                            "source": "handle_result_fallback",
                        },
                    )

            if not updated_task.is_terminal and not handle_result.success:
                updated_task.mark_failed(
                    handle_result.error or "teammate message handling failed"
                )

            success = updated_task.status == TeamTaskStatus.SUCCEEDED
            finished_at = now_timestamp()

            execution = SwarmTaskExecution(
                step_id=step.step_id,
                team_task_id=updated_task.task_id,
                assigned_to=updated_task.assigned_to,
                status=updated_task.status,
                success=success,
                handle_result=handle_result,
                result_messages=result_messages,
                shared_task=updated_task,
                error=updated_task.error,
                started_at=started_at,
                finished_at=finished_at,
            )

            await self.emit_event(
                SwarmEvent(
                    event_type=(
                        SwarmEventType.TASK_COMPLETED
                        if success
                        else SwarmEventType.TASK_FAILED
                    ),
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    step_id=step.step_id,
                    team_task_id=updated_task.task_id,
                    teammate_id=updated_task.assigned_to,
                    message=f"team task {updated_task.status.value}: {updated_task.task_id}",
                    metadata={
                        "execution": execution.to_dict(),
                    },
                )
            )

            return execution

        except Exception as exc:
            if task.status != TeamTaskStatus.CANCELLED:
                task.mark_failed(str(exc))

            finished_at = now_timestamp()

            execution = SwarmTaskExecution(
                step_id=step.step_id,
                team_task_id=task.task_id,
                assigned_to=task.assigned_to,
                status=task.status,
                success=False,
                handle_result=handle_result,
                result_messages=result_messages,
                shared_task=task,
                error=str(exc),
                started_at=started_at,
                finished_at=finished_at,
            )

            await self.emit_event(
                SwarmEvent(
                    event_type=SwarmEventType.TASK_FAILED,
                    swarm_id=self.swarm_id,
                    run_id=run_id,
                    step_id=step.step_id,
                    team_task_id=task.task_id,
                    teammate_id=task.assigned_to,
                    message=str(exc),
                    metadata={
                        "error_type": type(exc).__name__,
                    },
                )
            )

            return execution

        finally:
            self.active_task_ids.discard(task.task_id)

    def summarize_results(
        self,
        *,
        plan: SwarmPlan,
        executions: Sequence[SwarmTaskExecution],
        status: SwarmStatus,
    ) -> str:
        lines = [
            f"Swarm status: {status.value}",
            f"Task: {plan.task}",
            f"Strategy: {plan.strategy.value}",
            f"Steps: {len(plan.steps)}",
            "",
            "Team task results:",
        ]

        if not executions:
            lines.append("- No task executions.")
            return "\n".join(lines)

        for index, execution in enumerate(executions, start=1):
            lines.append(
                f"{index}. {execution.step_id} → {execution.status.value}"
            )

            if execution.assigned_to:
                lines.append(f"   assigned_to: {execution.assigned_to}")

            if execution.error:
                lines.append(f"   error: {execution.error}")

            if execution.shared_task and execution.shared_task.result:
                content = execution.shared_task.result.get("content")

                if content:
                    first_line = str(content).strip().splitlines()[0]
                    lines.append(f"   summary: {first_line}")

        return "\n".join(lines)

    async def cancel_current(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        self._cancel_requested = True

        cancelled_count = 0
        active_task_ids = set(self.active_task_ids)

        if self.current_run_id:
            for task in self.team.shared_task_list.values():
                if (
                    task.metadata.get("swarm_run_id") == self.current_run_id
                    and not task.is_terminal
                ):
                    active_task_ids.add(task.task_id)

        for task_id in active_task_ids:
            task = self.team.get_shared_task(task_id)

            if task is not None and not task.is_terminal:
                task.mark_cancelled(reason or "swarm cancelled")
                cancelled_count += 1

        # Give teammate task-backed execution a chance to expose current_task_record_id.
        for _ in range(5):
            teammate_cancel_count = await self.team.cancel_all_current(
                reason=reason or "swarm cancelled"
            )

            cancelled_count += teammate_cancel_count

            if teammate_cancel_count:
                break

            await asyncio.sleep(0)

        self.status = SwarmStatus.CANCELLED

        return cancelled_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "swarm_id": self.swarm_id,
            "team_id": self.team.team_id,
            "workspace_path": str(self.workspace_path),
            "status": self.status.value,
            "current_run_id": self.current_run_id,
            "active_task_ids": sorted(self.active_task_ids),
            "team": self.team.to_dict(),
            "metadata": safe_jsonable(self.metadata),
        }


def create_swarm(
    *,
    swarm_id: str | None = None,
    team: Team | None = None,
    manager: SubAgentManager | None = None,
    llm: SubAgentLLMCallable | None = None,
    planning_llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    workspace_path: str | Path = ".",
    config: SwarmConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> Swarm:
    return Swarm(
        swarm_id=swarm_id,
        team=team,
        manager=manager,
        llm=llm,
        planning_llm=planning_llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        config=config,
        metadata=metadata,
    )


__all__ = [
    "Swarm",
    "SwarmCancelledError",
    "SwarmConfig",
    "SwarmError",
    "SwarmEvent",
    "SwarmEventHandler",
    "SwarmEventType",
    "SwarmPlan",
    "SwarmPlanError",
    "SwarmRunRequest",
    "SwarmRunResult",
    "SwarmStatus",
    "SwarmStrategy",
    "SwarmTaskExecution",
    "SwarmTaskStep",
    "create_swarm",
    "infer_team_role_from_task",
    "new_swarm_id",
    "new_swarm_run_id",
    "normalize_swarm_strategy",
]