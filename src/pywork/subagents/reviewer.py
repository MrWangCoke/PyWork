from __future__ import annotations

from pywork.subagents.base import BaseSubAgent


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
    - 它只负责“审查”，不负责“修改”
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