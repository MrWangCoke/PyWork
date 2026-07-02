from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pywork.subagents.base import (
    BaseSubAgent,
    SubAgentAbortSignal,
    SubAgentContext,
    SubAgentRunResult,
)


REVIEW_TARGET_PATH_PATTERN = re.compile(
    r"(?P<path>[A-Za-z0-9_.\\/:-]+\.py)",
    re.IGNORECASE,
)

REVIEW_FILE_MAX_CHARS = 200_000


def normalize_review_path(path: str) -> str:
    return path.strip().strip("`'\".,，。；;：:").replace("\\", "/")


def path_inside_workspace(
    path: Path,
    *,
    workspace_path: Path,
) -> bool:
    try:
        path.resolve().relative_to(workspace_path)
    except ValueError:
        return False

    return True


def canonicalize_review_target_path(
    path: str,
    *,
    workspace_path: Path,
) -> str:
    normalized = normalize_review_path(path)

    direct_candidate = (workspace_path / normalized).resolve()

    if path_inside_workspace(direct_candidate, workspace_path=workspace_path) and direct_candidate.is_file():
        return normalized

    if normalized.startswith("src/utils/"):
        rewritten = "src/pywork/utils/" + normalized.removeprefix("src/utils/")
        rewritten_candidate = (workspace_path / rewritten).resolve()

        if path_inside_workspace(rewritten_candidate, workspace_path=workspace_path) and rewritten_candidate.is_file():
            return rewritten

    return normalized


class ReviewerSubAgent(BaseSubAgent):
    """
    代码审查子 Agent。

    第一版定位：
    - 审查代码正确性
    - 检查潜在 bug
    - 检查权限绕过风险
    - 检查边界情况
    - 检查测试覆盖
    - 给出可执行的修改建议

    注意：
    - 默认 readonly
    - 默认不允许写文件
    - 默认不允许执行 shell 命令
    - 它只负责"审查"，不负责"修改"
    """

    name = "reviewer"
    role = "reviewer"
    description = "Code review subagent"

    default_system_prompt = """
You are PyWork's code review subagent.

Your job:
- Review code for correctness, maintainability, safety, and test coverage.
- Look for bugs, edge cases, permission bypasses, brittle logic, and unclear APIs.
- Check whether the implementation matches the intended behavior.
- Identify missing tests or weak test assertions.
- Give actionable review comments that another agent or developer can apply.

Rules:
- Do not modify files.
- Do not run shell commands.
- Do not perform implementation.
- Stay inside the assigned workspace.
- Use available read/search tools only when needed.
- Prioritize correctness and safety over style.
- Be specific: mention files, functions, classes, or behaviors when possible.
- If there is not enough context, state what context is missing.

Output format:
1. Summary
2. Issues found
3. Safety and permission concerns
4. Test coverage gaps
5. Suggested fixes
6. Recommended next action
""".strip()

    default_allowed_tools = frozenset(
        {
            "file_read",
            "glob",
            "grep",
        }
    )

    default_permission_mode = "readonly"

    def find_review_target_path(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[str | None, Path | None]:
        workspace_path = (
            context.resolved_workspace_path()
            if context is not None
            else Path(".").expanduser().resolve()
        )

        candidates: list[str] = []

        for source in (
            getattr(context, "metadata", None),
            metadata,
        ):
            if isinstance(source, dict):
                value = source.get("review_target_path") or source.get("target_path")

                if value:
                    candidates.append(str(value))

        for match in REVIEW_TARGET_PATH_PATTERN.finditer(task):
            candidates.append(match.group("path"))

        for candidate in candidates:
            target_path = canonicalize_review_target_path(
                candidate,
                workspace_path=workspace_path,
            )
            resolved = (workspace_path / target_path).resolve()

            if not path_inside_workspace(resolved, workspace_path=workspace_path):
                continue

            if resolved.is_file():
                return target_path, resolved

        return None, None

    def build_enriched_review_task(
        self,
        *,
        original_task: str,
        target_path: str,
        file_content: str,
        truncated: bool,
    ) -> str:
        truncation_note = (
            "\n\nNote: The file content was truncated before review."
            if truncated
            else ""
        )

        return f"""
{original_task}

The target file has been preloaded for review.

Review target path:
`{target_path}`

File content:
```python
{file_content}
```

{truncation_note}

Now produce the review using this exact structure:

Summary
Issues found
Safety and permission concerns
Test coverage gaps
Suggested fixes

Recommended next action
""".strip()

    async def run(
        self,
        task: str,
        *,
        context: SubAgentContext | None = None,
        abort_signal: SubAgentAbortSignal | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SubAgentRunResult:
        metadata = dict(metadata or {})

        target_path, resolved_path = self.find_review_target_path(
            task,
            context=context,
            metadata=metadata,
        )

        review_metadata: dict[str, Any] = {
            "review_target_path": target_path,
            "review_file_loaded": False,
        }

        enriched_task = task

        if target_path is not None and resolved_path is not None:
            file_content = resolved_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
            truncated = len(file_content) > REVIEW_FILE_MAX_CHARS

            if truncated:
                file_content = file_content[:REVIEW_FILE_MAX_CHARS]

            enriched_task = self.build_enriched_review_task(
                original_task=task,
                target_path=target_path,
                file_content=file_content,
                truncated=truncated,
            )

            review_metadata.update(
                {
                    "review_target_path": target_path,
                    "review_resolved_path": str(resolved_path),
                    "review_file_loaded": True,
                    "review_file_size_chars": len(file_content),
                    "review_file_truncated": truncated,
                }
            )

        result = await super().run(
            enriched_task,
            context=context,
            abort_signal=abort_signal,
            metadata={
                **metadata,
                **review_metadata,
            },
        )

        result.metadata["reviewer"] = review_metadata

        return result
