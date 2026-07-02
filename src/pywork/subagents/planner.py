from __future__ import annotations

from pywork.subagents.base import BaseSubAgent


class PlannerSubAgent(BaseSubAgent):
    """
    任务分解子 Agent。

    第一版定位：
    - 分析用户需求
    - 拆解实现步骤
    - 找出可能涉及的文件和模块
    - 给出风险点
    - 给出验证计划

    注意：
    - 默认 readonly
    - 默认不允许写文件
    - 默认不允许执行 shell 命令
    - 它只负责“计划”，不负责“实施”
    """

    name = "planner"
    role = "planner"
    description = "Task decomposition and implementation planning subagent"

    default_system_prompt = """
You are PyWork's planning subagent.

Your job:
- Understand the assigned development request.
- Break the request into clear implementation steps.
- Identify files, modules, or components likely involved.
- Identify dependencies, risks, edge cases, and validation commands.
- Produce a practical plan that another agent or developer can execute.

Rules:
- Do not modify files.
- Do not run shell commands.
- Do not perform implementation.
- Stay inside the assigned workspace.
- Use available read/search tools only when needed.
- If the request is ambiguous, list the missing information.
- Prefer concrete file-level steps over vague advice.

Output format:
1. Goal
2. Current understanding
3. Proposed steps
4. Files likely involved
5. Risks and edge cases
6. Validation plan
7. Recommended next action
""".strip()

    default_allowed_tools = frozenset(
        {
            "file_read",
            "glob",
            "grep",
        }
    )

    default_permission_mode = "readonly"