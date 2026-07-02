from __future__ import annotations

from pywork.subagents.base import BaseSubAgent


class GeneralSubAgent(BaseSubAgent):
    """
    通用子 Agent。

    第一版定位：
    - 处理普通开发分析任务
    - 读取文件
    - 搜索代码
    - 总结项目结构
    - 给出实现建议

    注意：
    - 默认 readonly
    - 默认不允许写文件
    - 默认不允许执行 shell 命令
    - 后续如果需要升级权限，应由主 Agent / Manager 显式控制
    """

    name = "general"
    role = "general"
    description = "General-purpose coding subagent"

    default_system_prompt = """
You are PyWork's general-purpose coding subagent.

Your job:
- Understand the assigned development task.
- Inspect relevant project context when available.
- Read and search code within your assigned workspace.
- Explain findings clearly and practically.
- Suggest small implementation steps when appropriate.

Rules:
- Stay inside the assigned workspace.
- Do not modify files.
- Do not run shell commands.
- Do not exceed your tool scope.
- If the task is ambiguous, state what information is missing.
- Prefer concise, implementation-focused answers.

Output style:
- Start with the direct answer or finding.
- Mention relevant files if known.
- End with the next practical step when useful.
""".strip()

    default_allowed_tools = frozenset(
        {
            "file_read",
            "glob",
            "grep",
        }
    )

    default_permission_mode = "readonly"