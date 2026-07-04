from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import PurePath
from typing import Any


PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.\\/:-]+\.[A-Za-z0-9]+)"
)

ROLE_LABELS: dict[str, str] = {
    "planner": "Planner",
    "reviewer": "Reviewer",
    "verifier": "Verifier",
    "debugger": "Debugger",
    "general": "Agent",
    "worker": "Worker",
    "coordinator": "Coordinator",
}

ACTION_KEYWORDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("review", "code review", "审查", "审核", "代码审查"), "review"),
    (("test", "pytest", "verify", "验证", "测试", "运行测试"), "test"),
    (("plan", "planning", "规划", "计划", "拆解", "方案"), "plan"),
    (("debug", "diagnose", "调试", "排查", "修 bug", "修bug"), "debug"),
    (("send", "message", "发送消息", "通知"), "send"),
)


def clean_text(value: Any) -> str:
    text = str(value or "").strip()

    if text.lower() in {"none", "null", "undefined"}:
        return ""

    return text


def get_value(value: Any, *names: str) -> Any:
    if value is None:
        return None

    if isinstance(value, Mapping):
        for name in names:
            item = value.get(name)

            if item not in {None, ""}:
                return item

        return None

    for name in names:
        item = getattr(value, name, None)

        if item not in {None, ""}:
            return item

    return None


def humanize_identifier(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return ""

    for prefix in (
        "task_",
        "run_",
        "agent_",
        "teammate_",
        "member_",
        "worker_",
    ):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
            break

    text = text.replace("_", " ").replace("-", " ").strip()

    if not text:
        return clean_text(value)

    return text[:1].upper() + text[1:]


def role_label(value: Any) -> str:
    text = clean_text(value).lower()

    if not text:
        return ""

    return ROLE_LABELS.get(
        text,
        humanize_identifier(text),
    )


def choose_label(
    *,
    friendly_name: Any = None,
    role: Any = None,
    agent_name: Any = None,
    fallback_id: Any = None,
    fallback: str = "-",
) -> str:
    for value in (
        friendly_name,
        role_label(role),
        role_label(agent_name),
        humanize_identifier(fallback_id),
    ):
        text = clean_text(value)

        if text:
            return text

    return fallback


def basename_from_text(text: str) -> str:
    match = PATH_PATTERN.search(text)

    if match is None:
        return ""

    path = match.group("path").replace("\\", "/")

    try:
        return PurePath(path).name
    except Exception:
        return path.rsplit("/", 1)[-1]


def status_text_value(status: Any) -> str:
    return clean_text(getattr(status, "value", status)).lower()


def action_label(
    action: str,
    *,
    status: Any = None,
) -> str:
    status_text = status_text_value(status)

    labels: dict[str, dict[str, str]] = {
        "review": {
            "active": "正在审查",
            "succeeded": "已审查",
            "failed": "审查失败",
            "cancelled": "已停止审查",
        },
        "test": {
            "active": "正在运行测试",
            "succeeded": "测试完成",
            "failed": "测试失败",
            "cancelled": "已停止测试",
        },
        "plan": {
            "active": "正在拆解任务",
            "succeeded": "已完成规划",
            "failed": "规划失败",
            "cancelled": "已停止规划",
        },
        "debug": {
            "active": "正在排查",
            "succeeded": "已完成排查",
            "failed": "排查失败",
            "cancelled": "已停止排查",
        },
        "send": {
            "active": "正在发送消息",
            "succeeded": "已发送消息",
            "failed": "发送失败",
            "cancelled": "已停止发送",
        },
        "generic": {
            "active": "正在处理",
            "succeeded": "已完成",
            "failed": "处理失败",
            "cancelled": "已停止",
        },
    }

    if status_text in {"succeeded", "success", "done", "completed"}:
        state = "succeeded"
    elif status_text in {"failed", "error"}:
        state = "failed"
    elif status_text in {"cancelled", "canceled", "aborted"}:
        state = "cancelled"
    else:
        state = "active"

    return labels.get(action, labels["generic"])[state]


def infer_action_text(task_text: str, *, status: Any = None) -> str:
    lowered = task_text.lower()

    for keywords, action in ACTION_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return action_label(
                action,
                status=status,
            )

    status_text = status_text_value(status)

    if status_text in {"running", "queued", "retrying"}:
        return "正在处理"

    if status_text in {"succeeded", "success", "done", "completed"}:
        return "已完成"

    if status_text in {"failed", "error"}:
        return "处理失败"

    if status_text in {"cancelled", "canceled", "aborted"}:
        return "已停止"

    return "正在处理"


def short_task_text(task_text: str, *, max_chars: int = 42) -> str:
    text = clean_text(task_text)

    text = re.sub(
        r"^SubAgent\s+[A-Za-z0-9_-]+\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…"


def friendly_task_title(row: Any) -> str:
    task_text = clean_text(
        get_value(
            row,
            "name",
            "title",
            "task",
            "description",
            "current_task",
        )
    )

    if not task_text:
        return humanize_identifier(
            get_value(
                row,
                "task_id",
                "id",
                "current_task_record_id",
            )
        ) or "Task"

    agent = friendly_agent_label(
        {
            "agent_name": get_value(row, "agent", "agent_name"),
            "agent_id": get_value(row, "agent_id"),
            "role": get_value(row, "role"),
        }
    )
    action = infer_action_text(
        task_text,
        status=get_value(row, "status"),
    )
    target = basename_from_text(task_text)

    if agent != "-":
        if target:
            return f"{agent} {action} {target}"

        return f"{agent} {action}: {short_task_text(task_text)}"

    return short_task_text(task_text)


def friendly_agent_label(row: Any) -> str:
    return choose_label(
        friendly_name=get_value(row, "name", "display_name", "friendly_name"),
        role=get_value(row, "role"),
        agent_name=get_value(row, "agent_name", "agent"),
        fallback_id=get_value(row, "agent_id", "id"),
        fallback="-",
    )


def friendly_agent_activity(row: Any) -> str:
    task_text = clean_text(
        get_value(
            row,
            "current_task",
            "task",
            "name",
            "title",
            "description",
        )
    )

    if not task_text:
        status = clean_text(get_value(row, "status"))

        if status:
            return status

        return "空闲"

    action = infer_action_text(
        task_text,
        status=get_value(row, "status"),
    )
    target = basename_from_text(task_text)

    if target:
        return f"{action} {target}"

    return f"{action}: {short_task_text(task_text)}"


def friendly_team_member_label(row: Any) -> str:
    return choose_label(
        friendly_name=get_value(row, "name", "display_name", "friendly_name"),
        role=get_value(row, "role"),
        agent_name=get_value(row, "agent_name", "agent"),
        fallback_id=get_value(row, "teammate_id", "member_id", "id"),
        fallback="-",
    )


def friendly_team_member_activity(row: Any) -> str:
    task_id = clean_text(get_value(row, "current_task_record_id"))
    run_id = clean_text(get_value(row, "current_run_id"))
    is_busy = bool(get_value(row, "is_busy"))
    status = clean_text(get_value(row, "status"))

    if is_busy:
        return "正在处理任务"

    if task_id or run_id:
        return "有活动记录"

    if status:
        return status

    return "空闲"


def friendly_assignee_label(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return "-"

    return role_label(text) or humanize_identifier(text)
