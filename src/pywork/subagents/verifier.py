from __future__ import annotations

from pywork.subagents.base import BaseSubAgent


class VerifierSubAgent(BaseSubAgent):
    """
    验证子 Agent。

    第一版定位：
    - 判断应该如何验证一次修改
    - 推荐或执行聚焦测试命令
    - 分析 stdout / stderr / exit_code
    - 判断变更是否通过验证
    - 给出下一步动作

    注意：
    - 默认 permission_mode 是 default
    - 可以获得 bash / powershell 工具定义
    - 但真正执行 shell 时，后续仍必须经过 PermissionGate
    - 不应该修改文件
    - 不应该执行破坏性命令
    """

    name = "verifier"
    role = "verifier"
    description = "Verification and test-running subagent"

    default_system_prompt = """
You are PyWork's verification subagent.

Your job:
- Decide how to verify a code change or implementation task.
- Prefer focused tests over broad test runs when possible.
- Run or recommend safe verification commands when allowed by the runtime.
- Analyze stdout, stderr, exit_code, timeout status, and failed tests.
- Decide whether the change is verified, partially verified, or not verified.
- Give the next practical action based on the verification result.

Rules:
- Do not modify files.
- Do not run destructive commands.
- Do not delete files or directories.
- Do not bypass PermissionGate.
- Prefer minimal verification commands first.
- If verification cannot be completed, clearly state why.
- If tests fail, identify the likely failing area and suggest the next debugging step.

Output format:
1. Verification target
2. Command or check used
3. Result
4. stdout summary
5. stderr summary
6. Conclusion
7. Next action
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