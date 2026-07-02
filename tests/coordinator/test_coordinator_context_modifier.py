from __future__ import annotations

from pathlib import Path

import pytest

from pywork.coordinator.context_modifier import (
    ContextModificationRequest,
    ContextModifierError,
    ContextProfileName,
    WorkerContextModifier,
    WorkerContextProfile,
    create_default_context_modifier,
    normalize_worker_role,
)


def test_normalize_worker_role() -> None:
    assert normalize_worker_role("debug") == "debugger"
    assert normalize_worker_role("review") == "reviewer"
    assert normalize_worker_role("verify") == "verifier"
    assert normalize_worker_role("plan") == "planner"
    assert normalize_worker_role("unknown") == "unknown"


def test_context_modifier_builds_planner_context(tmp_path: Path) -> None:
    modifier = create_default_context_modifier()

    request = ContextModificationRequest(
        worker_id="worker_1",
        worker_role="planner",
        task="规划 coordinator 的实现步骤",
        workspace_path=tmp_path,
        parent_task="实现 Coordinator / Worker 系统",
        parent_messages=[
            {
                "role": "user",
                "content": "我们需要实现 coordinator、worker、context_modifier。",
            },
            {
                "role": "assistant",
                "content": "建议先做 context_modifier，再做 worker，最后做 coordinator。",
            },
        ],
        shared_memory={
            "project": "PyWork",
        },
    )

    result = modifier.modify(request)

    assert result.worker_id == "worker_1"
    assert result.worker_role == "planner"
    assert result.profile_name == "planner"
    assert result.task == "规划 coordinator 的实现步骤"
    assert result.workspace_path == str(tmp_path)
    assert result.messages[0]["role"] == "system"
    assert "Your assigned subtask" in result.messages[0]["content"]
    assert result.working_memory["shared_memory"]["project"] == "PyWork"
    assert result.metadata["context_modified"] is True


def test_context_modifier_debugger_prefers_error_context() -> None:
    modifier = create_default_context_modifier()

    messages = [
        {
            "role": "user",
            "content": "这里是一些普通需求描述。",
        },
        {
            "role": "assistant",
            "content": "实现了一个普通模块。",
        },
        {
            "role": "tool",
            "name": "pytest",
            "content": "FAILED tests/test_x.py::test_case\nTraceback: ValueError: bad value",
        },
        {
            "role": "user",
            "content": "为什么这个测试失败？",
        },
    ]

    request = ContextModificationRequest(
        worker_id="debug_worker",
        worker_role="debugger",
        task="分析 pytest 失败原因",
        parent_messages=messages,
        max_messages=3,
    )

    result = modifier.modify(request)

    text = "\n".join(
        message["content"]
        for message in result.messages
    )

    assert "Traceback" in text
    assert "pytest" in text
    assert result.worker_role == "debugger"


def test_context_modifier_reviewer_prefers_review_context() -> None:
    modifier = create_default_context_modifier()

    messages = [
        {
            "role": "user",
            "content": "今天天气不错。",
        },
        {
            "role": "assistant",
            "content": "diff --git a/a.py b/a.py\n+ dangerous permission bypass",
        },
        {
            "role": "user",
            "content": "审查一下这次权限改动有没有风险。",
        },
    ]

    request = ContextModificationRequest(
        worker_id="review_worker",
        worker_role="reviewer",
        task="审查权限系统改动",
        parent_messages=messages,
    )

    result = modifier.modify(request)

    text = "\n".join(
        message["content"]
        for message in result.messages
    )

    assert "diff" in text
    assert "permission" in text
    assert result.profile_name == "reviewer"


def test_context_modifier_redacts_secrets() -> None:
    modifier = create_default_context_modifier()

    request = ContextModificationRequest(
        worker_id="worker_secret",
        worker_role="debugger",
        task="检查环境变量泄露问题",
        parent_messages=[
            {
                "role": "user",
                "content": "api_key=sk-abcdefghijklmnop1234567890 token=abcdef1234567890",
            }
        ],
    )

    result = modifier.modify(request)

    text = "\n".join(
        message["content"]
        for message in result.messages
    )

    assert "sk-abcdefghijklmnop" not in text
    assert "abcdef1234567890" not in text
    assert "<redacted>" in text


def test_context_modifier_trims_total_chars() -> None:
    modifier = create_default_context_modifier()

    request = ContextModificationRequest(
        worker_id="small_worker",
        worker_role="general",
        task="处理短上下文",
        parent_messages=[
            {
                "role": "user",
                "content": "A" * 1000,
            },
            {
                "role": "assistant",
                "content": "B" * 1000,
            },
        ],
        max_total_chars=300,
        max_chars_per_message=1000,
    )

    result = modifier.modify(request)

    assert result.total_chars <= 300


def test_context_modifier_to_subagent_context(tmp_path: Path) -> None:
    modifier = create_default_context_modifier()

    request = ContextModificationRequest(
        worker_id="verifier_worker",
        worker_role="verifier",
        task="运行测试并汇总结果",
        workspace_path=tmp_path,
        parent_messages=[
            {
                "role": "user",
                "content": "请运行 pytest tests/test_x.py",
            }
        ],
        shared_memory={
            "changed_files": [
                "src/a.py",
            ]
        },
    )

    context = modifier.modify_to_subagent_context(request)

    assert context.task == "运行测试并汇总结果"
    assert str(context.workspace_path) == str(tmp_path)
    assert context.parent_messages
    assert context.working_memory["shared_memory"]["changed_files"] == [
        "src/a.py",
    ]


def test_context_modifier_custom_profile() -> None:
    custom = WorkerContextProfile(
        name=ContextProfileName.WORKER,
        max_messages=1,
        max_total_chars=500,
        recent_messages=1,
        role_keywords=("custom-keyword",),
    )

    modifier = WorkerContextModifier(
        profiles={
            "custom": custom,
        }
    )

    request = ContextModificationRequest(
        worker_id="custom_worker",
        worker_role="custom",
        task="custom task",
        parent_messages=[
            {
                "role": "user",
                "content": "old irrelevant",
            },
            {
                "role": "assistant",
                "content": "custom-keyword important",
            },
        ],
        max_messages=1,
    )

    result = modifier.modify(request)

    assert result.profile_name == "worker"
    assert "custom-keyword" in "\n".join(
        message["content"]
        for message in result.messages
    )


def test_context_modifier_requires_worker_id_and_task() -> None:
    modifier = create_default_context_modifier()

    with pytest.raises(ContextModifierError):
        modifier.modify(
            ContextModificationRequest(
                worker_id="",
                worker_role="general",
                task="x",
            )
        )

    with pytest.raises(ContextModifierError):
        modifier.modify(
            ContextModificationRequest(
                worker_id="worker",
                worker_role="general",
                task="   ",
            )
        )