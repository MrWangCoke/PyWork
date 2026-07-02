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

from pywork.coordinator.context_modifier import (
    WorkerContextModifier,
    create_default_context_modifier,
    normalize_worker_role,
)
from pywork.coordinator.worker import (
    CoordinatorWorker,
    WorkerExecutionMode,
    WorkerRunRequest,
    WorkerRunResult,
    WorkerStatus,
    create_worker,
    default_agent_for_worker_role,
)
from pywork.subagents.base import (
    SubAgentLLMCallable,
    SubAgentToolScope,
)
from pywork.subagents.manager import (
    SubAgentManager,
    create_default_subagent_manager,
)
from pywork.tasks.task import safe_jsonable


class CoordinatorError(Exception):
    """Coordinator 基础异常。"""


class CoordinatorPlanError(CoordinatorError):
    """Coordinator 任务分解异常。"""


class CoordinatorCancelledError(CoordinatorError):
    """Coordinator 被取消。"""


class CoordinatorStatus(str, Enum):
    IDLE = "idle"
    PLANNING = "planning"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CoordinatorPlanStrategy(str, Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class CoordinatorEventType(str, Enum):
    STARTED = "started"
    PLAN_CREATED = "plan_created"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


CoordinatorEventHandler = Callable[["CoordinatorEvent"], Any | Awaitable[Any]]


def now_timestamp() -> float:
    return time.time()


def new_coordinator_id(prefix: str = "coordinator") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_coordinator_run_id(prefix: str = "coord_run") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def new_step_id(index: int) -> str:
    return f"step_{index:02d}"


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value

    return value


def normalize_plan_strategy(value: CoordinatorPlanStrategy | str | None) -> CoordinatorPlanStrategy:
    if isinstance(value, CoordinatorPlanStrategy):
        return value

    text = str(value or CoordinatorPlanStrategy.SEQUENTIAL.value).strip().lower()

    try:
        return CoordinatorPlanStrategy(text)
    except ValueError as exc:
        valid = ", ".join(item.value for item in CoordinatorPlanStrategy)
        raise CoordinatorPlanError(
            f"Invalid coordinator strategy {value!r}. Valid strategies: {valid}"
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
        raise CoordinatorPlanError("empty planner response")

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

    raise CoordinatorPlanError("planner response does not contain JSON object")


def parse_json_object(text: str) -> dict[str, Any]:
    json_text = extract_json_object_text(text)

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise CoordinatorPlanError(str(exc)) from exc

    if not isinstance(payload, dict):
        raise CoordinatorPlanError("planner JSON must be an object")

    return payload


def infer_worker_role_from_task(task: str) -> str:
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
class CoordinatorEvent:
    event_type: CoordinatorEventType
    coordinator_id: str
    run_id: str | None = None
    step_id: str | None = None
    worker_id: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=now_timestamp)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "coordinator_id": self.coordinator_id,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "worker_id": self.worker_id,
            "message": self.message,
            "metadata": safe_jsonable(self.metadata),
            "timestamp": self.timestamp,
        }


@dataclass(slots=True)
class CoordinatorTaskStep:
    step_id: str
    task: str
    worker_role: str = "worker"
    agent_name: str | None = None
    depends_on: list[str] = field(default_factory=list)
    context_profile_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def normalized_worker_role(self) -> str:
        return normalize_worker_role(self.worker_role)

    @property
    def resolved_agent_name(self) -> str:
        return self.agent_name or default_agent_for_worker_role(self.normalized_worker_role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "task": self.task,
            "worker_role": self.normalized_worker_role,
            "agent_name": self.resolved_agent_name,
            "depends_on": list(self.depends_on),
            "context_profile_name": self.context_profile_name,
            "metadata": safe_jsonable(self.metadata),
        }

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        index: int,
    ) -> CoordinatorTaskStep:
        step_id = str(data.get("step_id") or data.get("id") or new_step_id(index)).strip()
        task = str(data.get("task") or data.get("description") or "").strip()

        if not task:
            raise CoordinatorPlanError(f"step {step_id!r} has empty task")

        raw_depends_on = data.get("depends_on") or data.get("dependencies") or []

        if not isinstance(raw_depends_on, list):
            raw_depends_on = []

        metadata = data.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        worker_role = str(
            data.get("worker_role")
            or data.get("role")
            or data.get("agent_name")
            or infer_worker_role_from_task(task)
        )

        return cls(
            step_id=step_id,
            task=task,
            worker_role=worker_role,
            agent_name=(
                str(data["agent_name"])
                if data.get("agent_name") is not None
                else None
            ),
            depends_on=[str(item) for item in raw_depends_on],
            context_profile_name=(
                str(data["context_profile_name"])
                if data.get("context_profile_name") is not None
                else None
            ),
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class CoordinatorPlan:
    task: str
    steps: list[CoordinatorTaskStep]
    strategy: CoordinatorPlanStrategy = CoordinatorPlanStrategy.SEQUENTIAL
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
    ) -> CoordinatorPlan:
        raw_steps = payload.get("steps") or payload.get("tasks") or []

        if not isinstance(raw_steps, list):
            raise CoordinatorPlanError("planner JSON field `steps` must be an array")

        steps: list[CoordinatorTaskStep] = []

        for index, item in enumerate(raw_steps, start=1):
            if not isinstance(item, Mapping):
                raise CoordinatorPlanError(f"step {index} must be an object")

            steps.append(
                CoordinatorTaskStep.from_mapping(
                    item,
                    index=index,
                )
            )

        if not steps:
            raise CoordinatorPlanError("planner returned no valid steps")

        metadata = payload.get("metadata")

        if not isinstance(metadata, dict):
            metadata = {}

        return cls(
            task=str(payload.get("task") or fallback_task),
            steps=steps,
            strategy=normalize_plan_strategy(payload.get("strategy")),
            summary=str(payload.get("summary") or ""),
            raw_response=raw_response,
            metadata=dict(metadata),
        )


@dataclass(slots=True)
class CoordinatorRunRequest:
    task: str
    workspace_path: str | Path = "."
    parent_messages: Sequence[Any] = field(default_factory=tuple)
    shared_memory: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    plan: CoordinatorPlan | None = None
    steps: Sequence[CoordinatorTaskStep | Mapping[str, Any]] | None = None
    strategy: CoordinatorPlanStrategy | str | None = None

    use_llm_planning: bool = True
    allow_fallback_plan: bool = True

    worker_execution_mode: WorkerExecutionMode | str = WorkerExecutionMode.DIRECT
    worker_tool_scope: SubAgentToolScope | None = None
    worker_max_steps: int | None = None

    max_concurrency: int | None = None
    timeout_seconds: float | None = None
    fail_fast: bool = False


@dataclass(slots=True)
class CoordinatorRunResult:
    coordinator_id: str
    run_id: str
    task: str
    status: CoordinatorStatus
    success: bool
    plan: CoordinatorPlan
    worker_results: list[WorkerRunResult]
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
            "coordinator_id": self.coordinator_id,
            "run_id": self.run_id,
            "task": self.task,
            "status": self.status.value,
            "success": self.success,
            "plan": self.plan.to_dict(),
            "worker_results": [
                result.to_dict()
                for result in self.worker_results
            ],
            "summary": self.summary,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "metadata": safe_jsonable(self.metadata),
        }


@dataclass(slots=True, frozen=True)
class CoordinatorConfig:
    max_concurrency: int = 4
    default_strategy: CoordinatorPlanStrategy = CoordinatorPlanStrategy.SEQUENTIAL
    allow_fallback_plan: bool = True
    include_raw_plan_response: bool = True


COORDINATOR_PLANNER_PROMPT = """
You are PyWork's Coordinator planner.

Your job is to decompose a user task into worker steps.

Available worker roles:
- planner: implementation planning and task decomposition.
- reviewer: code review, safety review, permission review, maintainability review.
- debugger: analyze errors, tracebacks, stuck runtime, failing tests.
- verifier: run or decide verification checks and summarize results.
- general: ordinary project analysis or general development work.

Return JSON only. Do not include markdown.

JSON schema:
{
  "task": "original or rewritten high-level task",
  "strategy": "sequential | parallel",
  "summary": "short explanation of the plan",
  "steps": [
    {
      "step_id": "step_01",
      "worker_role": "planner | reviewer | debugger | verifier | general",
      "agent_name": "planner | reviewer | debugger | verifier | general",
      "task": "specific subtask for this worker",
      "depends_on": [],
      "context_profile_name": "planner | reviewer | debugger | verifier | general",
      "metadata": {}
    }
  ],
  "metadata": {}
}

Planning rules:
- Use sequential when steps depend on previous output.
- Use parallel only when subtasks are independent.
- Keep steps focused and actionable.
- Do not invent unavailable worker roles.
- Prefer 1-4 steps unless the task clearly needs more.
""".strip()


class SubAgentCoordinator:
    """
    Coordinator / Worker 总控器。

    负责：
    - 分解任务
    - 分配 Worker
    - 执行 Worker 子任务
    - 汇总 Worker 结果
    """

    def __init__(
        self,
        *,
        coordinator_id: str | None = None,
        manager: SubAgentManager | None = None,
        context_modifier: WorkerContextModifier | None = None,
        llm: SubAgentLLMCallable | None = None,
        planning_llm: SubAgentLLMCallable | None = None,
        tool_definitions: Sequence[dict[str, Any]] | None = None,
        workspace_path: str | Path = ".",
        config: CoordinatorConfig | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.coordinator_id = coordinator_id or new_coordinator_id()
        self.workspace_path = Path(workspace_path)
        self.config = config or CoordinatorConfig()
        self.metadata = metadata or {}

        self.manager = manager or create_default_subagent_manager(
            llm=llm,
            tool_definitions=tool_definitions,
            workspace_path=workspace_path,
            metadata={
                "owner": "SubAgentCoordinator",
                "coordinator_id": self.coordinator_id,
                **self.metadata,
            },
        )

        self.context_modifier = context_modifier or create_default_context_modifier(
            metadata={
                "owner": "SubAgentCoordinator",
                "coordinator_id": self.coordinator_id,
            }
        )

        self.planning_llm = planning_llm or llm or self.manager.llm
        self.status = CoordinatorStatus.IDLE
        self.current_run_id: str | None = None
        self.active_workers: dict[str, CoordinatorWorker] = {}
        self.last_result: CoordinatorRunResult | None = None
        self._cancel_requested = False
        self._event_handlers: list[CoordinatorEventHandler] = []

    def add_event_handler(
        self,
        handler: CoordinatorEventHandler,
    ) -> None:
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)

    def remove_event_handler(
        self,
        handler: CoordinatorEventHandler,
    ) -> None:
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def emit_event(
        self,
        event: CoordinatorEvent,
    ) -> None:
        for handler in list(self._event_handlers):
            await maybe_await(handler(event))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinator_id": self.coordinator_id,
            "workspace_path": str(self.workspace_path),
            "status": self.status.value,
            "current_run_id": self.current_run_id,
            "active_workers": {
                step_id: worker.to_dict()
                for step_id, worker in self.active_workers.items()
            },
            "metadata": safe_jsonable(self.metadata),
        }

    def build_planning_messages(
        self,
        request: CoordinatorRunRequest,
    ) -> list[dict[str, Any]]:
        payload = {
            "task": request.task,
            "workspace_path": str(request.workspace_path),
            "parent_messages": safe_jsonable(list(request.parent_messages or [])[-8:]),
            "shared_memory": safe_jsonable(dict(request.shared_memory or {})),
            "artifacts": safe_jsonable(dict(request.artifacts or {})),
            "metadata": safe_jsonable(dict(request.metadata or {})),
            "preferred_strategy": (
                request.strategy.value
                if isinstance(request.strategy, CoordinatorPlanStrategy)
                else request.strategy
            ),
        }

        return [
            {
                "role": "system",
                "content": COORDINATOR_PLANNER_PROMPT,
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
        request: CoordinatorRunRequest,
    ) -> CoordinatorPlan:
        if request.plan is not None:
            return request.plan

        if request.steps is not None:
            steps: list[CoordinatorTaskStep] = []

            for index, item in enumerate(request.steps, start=1):
                if isinstance(item, CoordinatorTaskStep):
                    steps.append(item)
                elif isinstance(item, Mapping):
                    steps.append(
                        CoordinatorTaskStep.from_mapping(
                            item,
                            index=index,
                        )
                    )
                else:
                    raise CoordinatorPlanError("request.steps items must be CoordinatorTaskStep or mapping")

            return CoordinatorPlan(
                task=request.task,
                steps=steps,
                strategy=normalize_plan_strategy(
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
                            "component": "coordinator_planner",
                            "coordinator_id": self.coordinator_id,
                            **self.metadata,
                            **dict(request.metadata or {}),
                        },
                    )
                )

                raw = coerce_llm_content(response)
                payload = parse_json_object(raw)

                plan = CoordinatorPlan.from_payload(
                    payload,
                    fallback_task=request.task,
                    raw_response=raw if self.config.include_raw_plan_response else "",
                )

                if request.strategy is not None:
                    plan.strategy = normalize_plan_strategy(request.strategy)

                return plan

            except Exception:
                if not request.allow_fallback_plan and not self.config.allow_fallback_plan:
                    raise

        return self.create_fallback_plan(request)

    def create_fallback_plan(
        self,
        request: CoordinatorRunRequest,
    ) -> CoordinatorPlan:
        worker_role = infer_worker_role_from_task(request.task)

        return CoordinatorPlan(
            task=request.task,
            steps=[
                CoordinatorTaskStep(
                    step_id="step_01",
                    task=request.task,
                    worker_role=worker_role,
                    agent_name=default_agent_for_worker_role(worker_role),
                    context_profile_name=worker_role,
                    metadata={
                        "fallback": True,
                    },
                )
            ],
            strategy=normalize_plan_strategy(
                request.strategy or self.config.default_strategy
            ),
            summary="Fallback single-step plan.",
            metadata={
                "fallback": True,
            },
        )

    def create_worker_for_step(
        self,
        *,
        run_id: str,
        step: CoordinatorTaskStep,
        request: CoordinatorRunRequest,
    ) -> CoordinatorWorker:
        worker_id = f"{run_id}_{step.step_id}"

        return create_worker(
            worker_id=worker_id,
            worker_role=step.normalized_worker_role,
            agent_name=step.resolved_agent_name,
            description=f"Coordinator worker for {step.step_id}",
            workspace_path=request.workspace_path,
            manager=self.manager,
            context_modifier=self.context_modifier,
            tool_scope=request.worker_tool_scope,
            max_steps=request.worker_max_steps,
            metadata={
                "coordinator_id": self.coordinator_id,
                "coordinator_run_id": run_id,
                "step_id": step.step_id,
                **dict(step.metadata or {}),
            },
        )

    async def run(
        self,
        request: CoordinatorRunRequest,
    ) -> CoordinatorRunResult:
        if self.status in {
            CoordinatorStatus.PLANNING,
            CoordinatorStatus.RUNNING,
        }:
            raise CoordinatorError("Coordinator is already running")

        run_id = new_coordinator_run_id()
        started_at = now_timestamp()
        self.current_run_id = run_id
        self._cancel_requested = False
        self.status = CoordinatorStatus.PLANNING

        await self.emit_event(
            CoordinatorEvent(
                event_type=CoordinatorEventType.STARTED,
                coordinator_id=self.coordinator_id,
                run_id=run_id,
                message="coordinator started",
            )
        )

        try:
            plan = await self.create_plan(request)

            await self.emit_event(
                CoordinatorEvent(
                    event_type=CoordinatorEventType.PLAN_CREATED,
                    coordinator_id=self.coordinator_id,
                    run_id=run_id,
                    message="coordinator plan created",
                    metadata={
                        "plan": plan.to_dict(),
                    },
                )
            )

            self.status = CoordinatorStatus.RUNNING

            if plan.strategy == CoordinatorPlanStrategy.PARALLEL and not plan.has_dependencies:
                worker_results = await self._run_plan_parallel(
                    run_id=run_id,
                    plan=plan,
                    request=request,
                )
            else:
                worker_results = await self._run_plan_sequential(
                    run_id=run_id,
                    plan=plan,
                    request=request,
                )

            finished_at = now_timestamp()

            if self._cancel_requested:
                status = CoordinatorStatus.CANCELLED
                success = False
            else:
                success = all(result.success for result in worker_results)
                status = CoordinatorStatus.COMPLETED if success else CoordinatorStatus.FAILED

            summary = self.summarize_results(
                plan=plan,
                worker_results=worker_results,
                status=status,
            )

            result = CoordinatorRunResult(
                coordinator_id=self.coordinator_id,
                run_id=run_id,
                task=request.task,
                status=status,
                success=success,
                plan=plan,
                worker_results=worker_results,
                summary=summary,
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "strategy": plan.strategy.value,
                    "step_count": len(plan.steps),
                    "worker_result_count": len(worker_results),
                },
            )

            self.status = status
            self.last_result = result

            await self.emit_event(
                CoordinatorEvent(
                    event_type=(
                        CoordinatorEventType.CANCELLED
                        if status == CoordinatorStatus.CANCELLED
                        else CoordinatorEventType.COMPLETED
                        if status == CoordinatorStatus.COMPLETED
                        else CoordinatorEventType.FAILED
                    ),
                    coordinator_id=self.coordinator_id,
                    run_id=run_id,
                    message=f"coordinator {status.value}",
                    metadata={
                        "result": result.to_dict(),
                    },
                )
            )

            return result

        except Exception as exc:
            finished_at = now_timestamp()
            self.status = CoordinatorStatus.FAILED

            plan = self.create_fallback_plan(request)

            result = CoordinatorRunResult(
                coordinator_id=self.coordinator_id,
                run_id=run_id,
                task=request.task,
                status=CoordinatorStatus.FAILED,
                success=False,
                plan=plan,
                worker_results=[],
                summary=f"Coordinator failed: {exc}",
                error=str(exc),
                started_at=started_at,
                finished_at=finished_at,
                metadata={
                    "error_type": type(exc).__name__,
                },
            )

            self.last_result = result

            await self.emit_event(
                CoordinatorEvent(
                    event_type=CoordinatorEventType.FAILED,
                    coordinator_id=self.coordinator_id,
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
            self.active_workers.clear()
            self._cancel_requested = False

            if self.status in {
                CoordinatorStatus.COMPLETED,
                CoordinatorStatus.FAILED,
                CoordinatorStatus.CANCELLED,
            }:
                self.status = CoordinatorStatus.IDLE

    async def _run_plan_sequential(
        self,
        *,
        run_id: str,
        plan: CoordinatorPlan,
        request: CoordinatorRunRequest,
    ) -> list[WorkerRunResult]:
        results: list[WorkerRunResult] = []

        for step in plan.steps:
            if self._cancel_requested:
                break

            result = await self._run_step(
                run_id=run_id,
                step=step,
                plan=plan,
                request=request,
                previous_results=results,
            )

            results.append(result)

            if request.fail_fast and not result.success:
                break

        return results

    async def _run_plan_parallel(
        self,
        *,
        run_id: str,
        plan: CoordinatorPlan,
        request: CoordinatorRunRequest,
    ) -> list[WorkerRunResult]:
        concurrency = request.max_concurrency or self.config.max_concurrency
        semaphore = asyncio.Semaphore(max(1, concurrency))

        async def run_one(step: CoordinatorTaskStep) -> WorkerRunResult:
            async with semaphore:
                return await self._run_step(
                    run_id=run_id,
                    step=step,
                    plan=plan,
                    request=request,
                    previous_results=[],
                )

        results = await asyncio.gather(
            *[
                run_one(step)
                for step in plan.steps
            ]
        )

        return list(results)

    async def _run_step(
        self,
        *,
        run_id: str,
        step: CoordinatorTaskStep,
        plan: CoordinatorPlan,
        request: CoordinatorRunRequest,
        previous_results: Sequence[WorkerRunResult],
    ) -> WorkerRunResult:
        worker = self.create_worker_for_step(
            run_id=run_id,
            step=step,
            request=request,
        )

        self.active_workers[step.step_id] = worker

        await self.emit_event(
            CoordinatorEvent(
                event_type=CoordinatorEventType.STEP_STARTED,
                coordinator_id=self.coordinator_id,
                run_id=run_id,
                step_id=step.step_id,
                worker_id=worker.worker_id,
                message=f"step {step.step_id} started",
                metadata={
                    "step": step.to_dict(),
                },
            )
        )

        try:
            shared_memory = {
                **dict(request.shared_memory or {}),
                "coordinator": {
                    "coordinator_id": self.coordinator_id,
                    "run_id": run_id,
                    "strategy": plan.strategy.value,
                    "step_id": step.step_id,
                    "step_count": len(plan.steps),
                },
                "previous_results": [
                    result.to_dict()
                    for result in previous_results
                ],
            }

            result = await worker.execute(
                WorkerRunRequest(
                    task=step.task,
                    worker_id=worker.worker_id,
                    worker_role=step.normalized_worker_role,
                    agent_name=step.resolved_agent_name,
                    parent_task=request.task,
                    parent_messages=request.parent_messages,
                    shared_memory=shared_memory,
                    artifacts=request.artifacts,
                    workspace_path=request.workspace_path,
                    metadata={
                        **dict(request.metadata or {}),
                        **dict(step.metadata or {}),
                        "coordinator_id": self.coordinator_id,
                        "coordinator_run_id": run_id,
                        "step_id": step.step_id,
                    },
                    tool_scope=request.worker_tool_scope,
                    max_steps=request.worker_max_steps,
                    execution_mode=request.worker_execution_mode,
                    wait=True,
                    timeout_seconds=request.timeout_seconds,
                    context_profile_name=step.context_profile_name or step.normalized_worker_role,
                    parent_task_id=run_id,
                    run_id=f"{run_id}_{step.step_id}",
                )
            )

            await self.emit_event(
                CoordinatorEvent(
                    event_type=(
                        CoordinatorEventType.STEP_COMPLETED
                        if result.success
                        else CoordinatorEventType.STEP_FAILED
                    ),
                    coordinator_id=self.coordinator_id,
                    run_id=run_id,
                    step_id=step.step_id,
                    worker_id=worker.worker_id,
                    message=f"step {step.step_id} {'completed' if result.success else 'failed'}",
                    metadata={
                        "result": result.to_dict(),
                    },
                )
            )

            return result

        finally:
            self.active_workers.pop(step.step_id, None)

    def summarize_results(
        self,
        *,
        plan: CoordinatorPlan,
        worker_results: Sequence[WorkerRunResult],
        status: CoordinatorStatus,
    ) -> str:
        lines = [
            f"Coordinator status: {status.value}",
            f"Task: {plan.task}",
            f"Strategy: {plan.strategy.value}",
            f"Steps: {len(plan.steps)}",
            "",
            "Worker results:",
        ]

        if not worker_results:
            lines.append("- No worker results.")
            return "\n".join(lines)

        for index, result in enumerate(worker_results, start=1):
            lines.append(
                f"{index}. {result.worker_role}/{result.agent_name} → {result.status.value}"
            )

            if result.error:
                lines.append(f"   error: {result.error}")

            if result.content:
                first_line = result.content.strip().splitlines()[0]
                lines.append(f"   summary: {first_line}")

        return "\n".join(lines)

    async def cancel_current(
        self,
        *,
        reason: str | None = None,
    ) -> int:
        self._cancel_requested = True
        count = 0

        for worker in list(self.active_workers.values()):
            cancelled = await worker.cancel_current(
                reason=reason or "coordinator cancelled",
            )

            if cancelled:
                count += 1

        self.status = CoordinatorStatus.CANCELLED

        return count


def create_coordinator(
    *,
    coordinator_id: str | None = None,
    manager: SubAgentManager | None = None,
    context_modifier: WorkerContextModifier | None = None,
    llm: SubAgentLLMCallable | None = None,
    planning_llm: SubAgentLLMCallable | None = None,
    tool_definitions: Sequence[dict[str, Any]] | None = None,
    workspace_path: str | Path = ".",
    config: CoordinatorConfig | None = None,
    metadata: dict[str, Any] | None = None,
) -> SubAgentCoordinator:
    return SubAgentCoordinator(
        coordinator_id=coordinator_id,
        manager=manager,
        context_modifier=context_modifier,
        llm=llm,
        planning_llm=planning_llm,
        tool_definitions=tool_definitions,
        workspace_path=workspace_path,
        config=config,
        metadata=metadata,
    )


__all__ = [
    "CoordinatorConfig",
    "CoordinatorError",
    "CoordinatorEvent",
    "CoordinatorEventHandler",
    "CoordinatorEventType",
    "CoordinatorPlan",
    "CoordinatorPlanError",
    "CoordinatorPlanStrategy",
    "CoordinatorRunRequest",
    "CoordinatorRunResult",
    "CoordinatorStatus",
    "CoordinatorTaskStep",
    "SubAgentCoordinator",
    "create_coordinator",
    "infer_worker_role_from_task",
    "new_coordinator_id",
    "new_coordinator_run_id",
    "normalize_plan_strategy",
]