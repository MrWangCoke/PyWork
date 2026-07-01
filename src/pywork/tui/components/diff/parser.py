from __future__ import annotations

import re

from pywork.tui.components.diff.models import DiffLine, DiffStats


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@(?P<section>.*)$"
)


def parse_hunk_header(line: str) -> tuple[int, int] | None:
    """
    解析 unified diff 的 hunk header。

    示例：
        @@ -1,7 +1,8 @@

    返回：
        (old_start, new_start)
    """
    match = HUNK_HEADER_RE.match(line)

    if match is None:
        return None

    return (
        int(match.group("old_start")),
        int(match.group("new_start")),
    )


def parse_unified_diff(diff_text: str) -> list[DiffLine]:
    """
    把 unified diff 文本解析成 DiffLine 列表。

    这个解析器只负责 TUI 展示，不负责校验 patch 是否可应用。
    """
    if not diff_text.strip():
        return [
            DiffLine(
                old_lineno=None,
                new_lineno=None,
                marker="",
                content="No changes.",
                kind="empty",
            )
        ]

    lines: list[DiffLine] = []

    old_lineno: int | None = None
    new_lineno: int | None = None

    for raw_line in diff_text.splitlines():
        line = raw_line.rstrip("\n")

        hunk_position = parse_hunk_header(line)

        if hunk_position is not None:
            old_lineno, new_lineno = hunk_position

            lines.append(
                DiffLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="hunk",
                )
            )
            continue

        if line.startswith("diff --git "):
            lines.append(
                DiffLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="file_header",
                )
            )
            continue

        if line.startswith("--- ") or line.startswith("+++ "):
            lines.append(
                DiffLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="file_header",
                )
            )
            continue

        if line.startswith("+"):
            lines.append(
                DiffLine(
                    old_lineno=None,
                    new_lineno=new_lineno,
                    marker="+",
                    content=line[1:],
                    kind="addition",
                )
            )

            if new_lineno is not None:
                new_lineno += 1

            continue

        if line.startswith("-"):
            lines.append(
                DiffLine(
                    old_lineno=old_lineno,
                    new_lineno=None,
                    marker="-",
                    content=line[1:],
                    kind="deletion",
                )
            )

            if old_lineno is not None:
                old_lineno += 1

            continue

        if line.startswith(" "):
            lines.append(
                DiffLine(
                    old_lineno=old_lineno,
                    new_lineno=new_lineno,
                    marker=" ",
                    content=line[1:],
                    kind="context",
                )
            )

            if old_lineno is not None:
                old_lineno += 1

            if new_lineno is not None:
                new_lineno += 1

            continue

        if line.startswith("\\"):
            lines.append(
                DiffLine(
                    old_lineno=None,
                    new_lineno=None,
                    marker="",
                    content=line,
                    kind="meta",
                )
            )
            continue

        lines.append(
            DiffLine(
                old_lineno=None,
                new_lineno=None,
                marker="",
                content=line,
                kind="meta",
            )
        )

    return lines


def collect_stats(lines: list[DiffLine]) -> DiffStats:
    """统计 DiffLine 列表。"""
    additions = sum(1 for line in lines if line.kind == "addition")
    deletions = sum(1 for line in lines if line.kind == "deletion")
    hunks = sum(1 for line in lines if line.kind == "hunk")

    files = sum(
        1
        for line in lines
        if line.kind == "file_header"
        and line.content.startswith("+++ ")
        and line.content.strip() != "+++ /dev/null"
    )

    return DiffStats(
        files=files,
        additions=additions,
        deletions=deletions,
        hunks=hunks,
        lines=len(lines),
    )


def line_number_width(lines: list[DiffLine]) -> int:
    """计算 old/new 行号列宽。"""
    max_lineno = 0

    for line in lines:
        if line.old_lineno is not None:
            max_lineno = max(max_lineno, line.old_lineno)

        if line.new_lineno is not None:
            max_lineno = max(max_lineno, line.new_lineno)

    return max(3, len(str(max_lineno)))