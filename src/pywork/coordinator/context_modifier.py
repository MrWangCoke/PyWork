from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any

from pywork.subagents.base import SubAgentContext


class ContextModifierError(Exception):
    """ContextModifier 基础异常。"""


class ContextProfileName(str, Enum):
    GENERAL = "general"
    PLANNER = "planner"
    REVIEWER = "reviewer"
    DEBUGGER = "debugger"
    VERIFIER = "verifier"
    WORKER = "worker"


WORKER_ROLE_ALIASES: dict[str, str] = {
    "general": "general",
    "default": "general",
    "assistant": "general",
    "通用": "general",
    "默认": "general",

    "planner": "planner",
    "plan": "planner",
    "planning": "planner",
    "计划": "planner",
    "规划": "planner",

    "reviewer": "reviewer",
    "review": "reviewer",
    "code_review": "reviewer",
    "审查": "reviewer",
    "评审": "reviewer",

    "debugger": "debugger",
    "debug": "debugger",
    "diagnose": "debugger",
    "diagnostic": "debugger",
    "调试": "debugger",
    "排错": "debugger",

    "verifier": "verifier",
    "verify": "verifier",
    "test": "verifier",
    "tester": "verifier",
    "验证": "verifier",
    "测试": "verifier",

    "worker": "worker",
    "executor": "worker",
    "执行": "worker",
}


DEFAULT_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"(?i)\b(api[_-]?key|token|secret|password|passwd|pwd)\b\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{8,})['\"]?"
        ),
        r"\1=<redacted>",
    ),
    (
        re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b"),
        "sk-<redacted>",
    ),
    (
        re.compile(r"\bghp_[A-Za-z0-9_]{12,}\b"),
        "ghp_<redacted>",
    ),
    (
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{12,}\b"),
        "github_pat_<redacted>",
    ),
    (
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL),
        "<redacted-private-key>",
    ),
)


@dataclass(slots=True, frozen=True)
class WorkerContextProfile:
    """
    Worker 上下文配置。

    max_messages:
        最多保留多少条父消息。

    max_chars_per_message:
        单条消息最多保留多少字符。

    max_total_chars:
        所有上下文消息合计最多保留多少字符。

    recent_messages:
        无论相关度如何，最近多少条消息会优先保留。

    relevance_threshold:
        非最近消息至少达到多少分才保留。

    role_keywords:
        该 Worker 角色特别关心的关键词。
    """

    name: ContextProfileName
    max_messages: int = 12
    max_chars_per_message: int = 2000
    max_total_chars: int = 12000
    recent_messages: int = 4
    relevance_threshold: int = 2
    include_system_messages: bool = True
    include_tool_messages: bool = True
    include_context_header: bool = True
    redact_secrets: bool = True
    role_keywords: tuple[str, ...] = ()
    description: str = ""


DEFAULT_CONTEXT_PROFILES: dict[str, WorkerContextProfile] = {
    "general": WorkerContextProfile(
        name=ContextProfileName.GENERAL,
        role_keywords=(
            "需求",
            "目标",
            "实现",
            "代码",
            "文件",
            "project",
            "task",
            "code",
            "implementation",
        ),
        description="General worker context.",
    ),
    "planner": WorkerContextProfile(
        name=ContextProfileName.PLANNER,
        max_messages=14,
        max_total_chars=14000,
        recent_messages=5,
        role_keywords=(
            "需求",
            "目标",
            "约束",
            "计划",
            "规划",
            "步骤",
            "架构",
            "设计",
            "拆分",
            "任务",
            "验收",
            "plan",
            "design",
            "architecture",
            "steps",
            "requirements",
            "constraints",
            "roadmap",
        ),
        description="Planning worker context.",
    ),
    "reviewer": WorkerContextProfile(
        name=ContextProfileName.REVIEWER,
        max_messages=14,
        max_total_chars=16000,
        recent_messages=5,
        role_keywords=(
            "代码",
            "diff",
            "review",
            "审查",
            "评审",
            "风险",
            "安全",
            "权限",
            "测试覆盖",
            "边界",
            "bug",
            "maintainability",
            "security",
            "permission",
            "coverage",
            "edge case",
        ),
        description="Code review worker context.",
    ),
    "debugger": WorkerContextProfile(
        name=ContextProfileName.DEBUGGER,
        max_messages=16,
        max_total_chars=18000,
        recent_messages=6,
        relevance_threshold=1,
        role_keywords=(
            "报错",
            "错误",
            "异常",
            "失败",
            "失败原因",
            "卡住",
            "没反应",
            "traceback",
            "exception",
            "error",
            "failed",
            "failure",
            "pytest",
            "assert",
            "stack",
            "stderr",
            "stdout",
            "timeout",
            "hang",
        ),
        description="Debugging worker context.",
    ),
    "verifier": WorkerContextProfile(
        name=ContextProfileName.VERIFIER,
        max_messages=14,
        max_total_chars=15000,
        recent_messages=5,
        role_keywords=(
            "测试",
            "验证",
            "运行",
            "pytest",
            "compileall",
            "通过",
            "失败",
            "结果",
            "stdout",
            "stderr",
            "exit_code",
            "verify",
            "test",
            "command",
            "check",
            "validation",
        ),
        description="Verification worker context.",
    ),
    "worker": WorkerContextProfile(
        name=ContextProfileName.WORKER,
        max_messages=12,
        max_total_chars=12000,
        recent_messages=4,
        role_keywords=(
            "执行",
            "实现",
            "子任务",
            "worker",
            "execute",
            "implementation",
        ),
        description="Generic worker context.",
    ),
}


@dataclass(slots=True)
class ContextModificationRequest:
    worker_id: str
    worker_role: str
    task: str
    workspace_path: str | Path = "."
    parent_task: str | None = None
    parent_messages: Sequence[Any] = field(default_factory=tuple)
    shared_memory: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    profile_name: str | None = None

    max_messages: int | None = None
    max_chars_per_message: int | None = None
    max_total_chars: int | None = None
    recent_messages: int | None = None


@dataclass(slots=True)
class ContextModificationResult:
    worker_id: str
    worker_role: str
    profile_name: str
    task: str
    workspace_path: str
    parent_task: str | None
    messages: list[dict[str, Any]]
    working_memory: dict[str, Any]
    metadata: dict[str, Any]
    selected_message_count: int
    original_message_count: int
    omitted_message_count: int
    total_chars: int

    def to_subagent_context(self) -> SubAgentContext:
        return SubAgentContext(
            task=self.task,
            workspace_path=self.workspace_path,
            parent_messages=list(self.messages),
            working_memory=dict(self.working_memory),
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "worker_role": self.worker_role,
            "profile_name": self.profile_name,
            "task": self.task,
            "workspace_path": self.workspace_path,
            "parent_task": self.parent_task,
            "messages": list(self.messages),
            "working_memory": dict(self.working_memory),
            "metadata": dict(self.metadata),
            "selected_message_count": self.selected_message_count,
            "original_message_count": self.original_message_count,
            "omitted_message_count": self.omitted_message_count,
            "total_chars": self.total_chars,
        }


def normalize_worker_role(value: str | None) -> str:
    text = str(value or "worker").strip().lower()

    return WORKER_ROLE_ALIASES.get(text, text or "worker")


def safe_jsonable(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str | int | float | bool):
        return value

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): safe_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple | set):
        return [
            safe_jsonable(item)
            for item in value
        ]

    if is_dataclass(value):
        return safe_jsonable(asdict(value))

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return safe_jsonable(value.to_dict())

    return str(value)


def stringify_content(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    try:
        return json.dumps(
            safe_jsonable(value),
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        return str(value)


def truncate_text(
    text: str,
    max_chars: int,
) -> str:
    if max_chars <= 0:
        return ""

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 20] + "\n...[truncated]..."


def redact_sensitive_text(text: str) -> str:
    result = text

    for pattern, replacement in DEFAULT_SECRET_PATTERNS:
        result = pattern.sub(replacement, result)

    return result


def message_to_dict(
    message: Any,
    *,
    max_chars: int,
    redact_secrets: bool = True,
) -> dict[str, Any]:
    if isinstance(message, Mapping):
        role = str(message.get("role", "user"))
        content = stringify_content(message.get("content", ""))
        name = message.get("name")
        tool_call_id = message.get("tool_call_id")
        metadata = message.get("metadata")
    else:
        role = str(getattr(message, "role", "user"))
        content = stringify_content(getattr(message, "content", ""))
        name = getattr(message, "name", None)
        tool_call_id = getattr(message, "tool_call_id", None)
        metadata = getattr(message, "metadata", None)

    if redact_secrets:
        content = redact_sensitive_text(content)

    content = truncate_text(content, max_chars)

    data: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    if name:
        data["name"] = str(name)

    if tool_call_id:
        data["tool_call_id"] = str(tool_call_id)

    if isinstance(metadata, Mapping):
        data["metadata"] = safe_jsonable(dict(metadata))

    return data


def extract_keywords(text: str) -> set[str]:
    lowered = text.lower()

    tokens = set(
        re.findall(
            r"[a-zA-Z_][a-zA-Z0-9_\-./]*|\d+|[\u4e00-\u9fff]{2,}",
            lowered,
        )
    )

    # 对中文短任务，整段中文经常被正则当成一个 token。
    # 这里额外加一些常见短词，提高相关度判断。
    common_zh = (
        "实现",
        "报错",
        "错误",
        "失败",
        "测试",
        "验证",
        "审查",
        "计划",
        "规划",
        "权限",
        "文件",
        "代码",
        "运行",
        "重试",
        "取消",
        "上下文",
        "worker",
        "agent",
    )

    for word in common_zh:
        if word in lowered:
            tokens.add(word)

    return {
        token
        for token in tokens
        if len(token) >= 2
    }


def message_text_for_scoring(message: Mapping[str, Any]) -> str:
    parts = [
        str(message.get("role", "")),
        str(message.get("name", "")),
        str(message.get("content", "")),
    ]

    metadata = message.get("metadata")

    if isinstance(metadata, Mapping):
        parts.append(stringify_content(metadata))

    return "\n".join(parts).lower()


def score_message_relevance(
    message: Mapping[str, Any],
    *,
    task_keywords: set[str],
    role_keywords: Sequence[str],
) -> int:
    text = message_text_for_scoring(message)
    score = 0

    for keyword in task_keywords:
        if keyword and keyword.lower() in text:
            score += 2

    for keyword in role_keywords:
        if keyword.lower() in text:
            score += 3

    role = str(message.get("role", "")).lower()

    if role == "system":
        score += 1

    if role == "tool":
        score += 1

    return score


def total_message_chars(messages: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        len(str(message.get("content", "")))
        for message in messages
    )


def trim_messages_to_total_chars(
    messages: list[dict[str, Any]],
    *,
    max_total_chars: int,
) -> list[dict[str, Any]]:
    if max_total_chars <= 0:
        return []

    total = 0
    kept_reversed: list[dict[str, Any]] = []

    for message in reversed(messages):
        content = str(message.get("content", ""))
        length = len(content)

        if total + length > max_total_chars:
            remaining = max_total_chars - total

            if remaining > 80:
                copied = dict(message)
                copied["content"] = truncate_text(content, remaining)
                kept_reversed.append(copied)

            break

        kept_reversed.append(message)
        total += length

    return list(reversed(kept_reversed))


class WorkerContextModifier:
    """
    Worker 上下文定制器。

    输入主上下文，输出适合指定 Worker 的 SubAgentContext。
    """

    def __init__(
        self,
        *,
        profiles: Mapping[str, WorkerContextProfile] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.profiles: dict[str, WorkerContextProfile] = {
            **DEFAULT_CONTEXT_PROFILES,
            **dict(profiles or {}),
        }
        self.metadata = metadata or {}

    def get_profile(
        self,
        worker_role: str,
        profile_name: str | None = None,
    ) -> WorkerContextProfile:
        if profile_name:
            key = normalize_worker_role(profile_name)
            profile = self.profiles.get(key)

            if profile is not None:
                return profile

        normalized_role = normalize_worker_role(worker_role)

        return self.profiles.get(
            normalized_role,
            self.profiles["worker"],
        )

    def build_context_header(
        self,
        *,
        request: ContextModificationRequest,
        profile: WorkerContextProfile,
    ) -> dict[str, Any]:
        parent_task = request.parent_task or "None"

        content = "\n".join(
            [
                "You are a focused PyWork Worker.",
                f"Worker id: {request.worker_id}",
                f"Worker role: {normalize_worker_role(request.worker_role)}",
                f"Context profile: {profile.name.value}",
                f"Parent task: {parent_task}",
                "",
                "Your assigned subtask:",
                request.task,
                "",
                "Use only the context below that is relevant to this subtask.",
                "Do not assume omitted context unless it is explicitly provided.",
            ]
        )

        return {
            "role": "system",
            "name": "worker_context_modifier",
            "content": content,
            "metadata": {
                "generated_by": "WorkerContextModifier",
                "profile": profile.name.value,
                "worker_id": request.worker_id,
                "worker_role": normalize_worker_role(request.worker_role),
            },
        }

    def build_working_memory(
        self,
        request: ContextModificationRequest,
        *,
        profile: WorkerContextProfile,
    ) -> dict[str, Any]:
        return {
            "worker_id": request.worker_id,
            "worker_role": normalize_worker_role(request.worker_role),
            "profile_name": profile.name.value,
            "task": request.task,
            "parent_task": request.parent_task,
            "shared_memory": safe_jsonable(dict(request.shared_memory or {})),
            "artifacts": safe_jsonable(dict(request.artifacts or {})),
            "context_modifier": {
                "max_messages": profile.max_messages,
                "max_chars_per_message": profile.max_chars_per_message,
                "max_total_chars": profile.max_total_chars,
                "recent_messages": profile.recent_messages,
            },
        }

    def select_messages(
        self,
        *,
        request: ContextModificationRequest,
        profile: WorkerContextProfile,
    ) -> list[dict[str, Any]]:
        max_chars_per_message = (
            request.max_chars_per_message
            if request.max_chars_per_message is not None
            else profile.max_chars_per_message
        )
        max_messages = (
            request.max_messages
            if request.max_messages is not None
            else profile.max_messages
        )
        recent_messages = (
            request.recent_messages
            if request.recent_messages is not None
            else profile.recent_messages
        )

        normalized_messages = [
            message_to_dict(
                message,
                max_chars=max_chars_per_message,
                redact_secrets=profile.redact_secrets,
            )
            for message in request.parent_messages
        ]

        if not normalized_messages:
            return []

        task_keywords = extract_keywords(
            "\n".join(
                item
                for item in [
                    request.task,
                    request.parent_task or "",
                    stringify_content(request.metadata),
                ]
            )
        )

        scored: list[tuple[int, int, dict[str, Any]]] = []

        recent_start = max(0, len(normalized_messages) - recent_messages)

        for index, message in enumerate(normalized_messages):
            role = str(message.get("role", "")).lower()

            if role == "system" and not profile.include_system_messages:
                continue

            if role == "tool" and not profile.include_tool_messages:
                continue

            score = score_message_relevance(
                message,
                task_keywords=task_keywords,
                role_keywords=profile.role_keywords,
            )

            is_recent = index >= recent_start

            if is_recent:
                score += 4

            if is_recent or score >= profile.relevance_threshold:
                scored.append((score, index, message))

        # 先按分数取前 max_messages，再恢复原始顺序。
        scored.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        selected = scored[:max_messages]

        selected.sort(
            key=lambda item: item[1],
        )

        return [
            message
            for _, _, message in selected
        ]

    def modify(
        self,
        request: ContextModificationRequest,
    ) -> ContextModificationResult:
        if not request.worker_id:
            raise ContextModifierError("worker_id is required")

        if not request.task.strip():
            raise ContextModifierError("task is required")

        profile = self.get_profile(
            request.worker_role,
            profile_name=request.profile_name,
        )

        selected_messages = self.select_messages(
            request=request,
            profile=profile,
        )

        if profile.include_context_header:
            selected_messages.insert(
                0,
                self.build_context_header(
                    request=request,
                    profile=profile,
                ),
            )

        max_total_chars = (
            request.max_total_chars
            if request.max_total_chars is not None
            else profile.max_total_chars
        )

        selected_messages = trim_messages_to_total_chars(
            selected_messages,
            max_total_chars=max_total_chars,
        )

        original_count = len(request.parent_messages)
        selected_count = len(selected_messages)
        omitted_count = max(0, original_count - selected_count)

        working_memory = self.build_working_memory(
            request,
            profile=profile,
        )

        metadata = {
            **self.metadata,
            **safe_jsonable(dict(request.metadata or {})),
            "worker_id": request.worker_id,
            "worker_role": normalize_worker_role(request.worker_role),
            "profile_name": profile.name.value,
            "original_message_count": original_count,
            "selected_message_count": selected_count,
            "omitted_message_count": omitted_count,
            "total_chars": total_message_chars(selected_messages),
            "context_modified": True,
        }

        return ContextModificationResult(
            worker_id=request.worker_id,
            worker_role=normalize_worker_role(request.worker_role),
            profile_name=profile.name.value,
            task=request.task,
            workspace_path=str(Path(request.workspace_path)),
            parent_task=request.parent_task,
            messages=selected_messages,
            working_memory=working_memory,
            metadata=metadata,
            selected_message_count=selected_count,
            original_message_count=original_count,
            omitted_message_count=omitted_count,
            total_chars=total_message_chars(selected_messages),
        )

    def modify_to_subagent_context(
        self,
        request: ContextModificationRequest,
    ) -> SubAgentContext:
        return self.modify(request).to_subagent_context()


def create_default_context_modifier(
    *,
    profiles: Mapping[str, WorkerContextProfile] | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkerContextModifier:
    return WorkerContextModifier(
        profiles=profiles,
        metadata=metadata,
    )


__all__ = [
    "ContextModificationRequest",
    "ContextModificationResult",
    "ContextModifierError",
    "ContextProfileName",
    "WorkerContextModifier",
    "WorkerContextProfile",
    "create_default_context_modifier",
    "extract_keywords",
    "message_to_dict",
    "normalize_worker_role",
    "redact_sensitive_text",
    "score_message_relevance",
    "truncate_text",
]