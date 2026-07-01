from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DiffLineKind = Literal[
    "file_header",
    "hunk",
    "addition",
    "deletion",
    "context",
    "meta",
    "empty",
]


@dataclass(slots=True, frozen=True)
class DiffLine:
    """
    一行用于 TUI 展示的 diff 数据。

    old_lineno:
        旧文件行号。新增行没有旧行号。

    new_lineno:
        新文件行号。删除行没有新行号。

    marker:
        diff 标记：
        + 新增
        - 删除
        空格 上下文
        空字符串 文件头 / hunk / meta

    content:
        展示内容，不包含行尾换行。

    kind:
        行类型，用来决定颜色。
    """

    old_lineno: int | None
    new_lineno: int | None
    marker: str
    content: str
    kind: DiffLineKind


@dataclass(slots=True, frozen=True)
class DiffStats:
    """Diff 展示统计。"""

    files: int = 0
    additions: int = 0
    deletions: int = 0
    hunks: int = 0
    lines: int = 0

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def is_empty(self) -> bool:
        return self.changed_lines == 0 and self.hunks == 0


@dataclass(slots=True, frozen=True)
class DiffRenderOptions:
    """Diff 渲染选项。"""

    show_header: bool = True
    show_line_numbers: bool = True
    max_lines: int | None = 1_000