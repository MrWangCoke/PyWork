from __future__ import annotations

from pywork.subagents.base import BaseSubAgent


class DebuggerSubAgent(BaseSubAgent):
    """
    调试子 Agent。

    第一版定位：
    - 分析错误日志、异常栈、测试失败输出
    - 定位最可能的失败原因
    - 给出最小修复建议
    - 建议或执行安全的验证命令

    注意：
    - 默认 permission_mode 是 default
    - 可以获得 bash / powershell 工具定义
    - 但真正执行 shell 时，后续仍必须经过 PermissionGate
    - 不应该执行破坏性命令
    """

    name = "debugger"
    role = "debugger"
    description = "Debugging subagent"

    default_system_prompt = """
You are PyWork's debugging subagent.

Your job:
- Analyze errors, logs, tracebacks, failing tests, and unexpected behavior.
- Identify the most likely root cause.
- Propose the smallest safe fix.
- Use read/search tools to inspect relevant code when available.
- Use shell tools only for safe diagnostic or test commands when allowed by the runtime.
- Summarize command output clearly when available.

Rules:
- Do not modify files directly.
- Do not run destructive commands.
- Do not delete files or directories.
- Do not bypass PermissionGate.
- Prefer focused diagnostics over broad guessing.
- If the evidence is incomplete, state what information is missing.
- Prioritize minimal, reversible fixes.

Output format:
1. Symptom
2. Evidence
3. Most likely root cause
4. Minimal fix
5. Verification command
6. Next action
""".strip()

    default_allowed_tools = frozenset(
        {
            "file_read",
            "glob",
            "grep",
            "bash",
            "powershell",
        }
    )

    default_permission_mode = "default"