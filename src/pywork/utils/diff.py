from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONTEXT_LINES = 3
DEFAULT_OLD_PREFIX = "a"
DEFAULT_NEW_PREFIX = "b"


class DiffError(Exception):
    """Diff 引擎基础异常。"""


class DiffParseError(DiffError):
    """unified diff 解析失败。"""


@dataclass(slots=True, frozen=True)
class DiffFileStat:
    """单个文件的 diff 统计。"""

    path: str
    old_path: str | None = None
    new_path: str | None = None
    additions: int = 0
    deletions: int = 0
    hunks: int = 0
    is_added_file: bool = False
    is_removed_file: bool = False
    is_renamed_file: bool = False
    is_binary_file: bool = False

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions


@dataclass(slots=True, frozen=True)
class DiffStats:
    """整个 diff 的统计。"""

    files_changed: int = 0
    additions: int = 0
    deletions: int = 0
    hunks: int = 0
    files: tuple[DiffFileStat, ...] = ()

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def is_empty(self) -> bool:
        return self.files_changed == 0 and self.changed_lines == 0


@dataclass(slots=True, frozen=True)
class UnifiedDiff:
    """生成后的 unified diff 结果。"""

    text: str
    old_path: str | None = None
    new_path: str | None = None
    stats: DiffStats = DiffStats()

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "old_path": self.old_path,
            "new_path": self.new_path,
            "stats": {
                "files_changed": self.stats.files_changed,
                "additions": self.stats.additions,
                "deletions": self.stats.deletions,
                "hunks": self.stats.hunks,
                "changed_lines": self.stats.changed_lines,
            },
        }


@dataclass(slots=True, frozen=True)
class DiffChange:
    """多文件 diff 的单个文件变化。"""

    path: str
    old_text: str
    new_text: str
    old_path: str | None = None
    new_path: str | None = None
    is_new_file: bool = False
    is_deleted_file: bool = False


def split_text_for_diff(text: str) -> list[str]:
    """
    把文本切成 difflib 需要的行列表。

    keepends=True 很重要：
    - 保留每行末尾的换行符
    - 生成的 diff 更接近真实文件 diff
    """
    if not text:
        return []

    return text.splitlines(keepends=True)


def normalize_diff_path(path: str | Path) -> str:
    """
    统一 diff 里的路径显示。

    Windows 路径会从：
        src\\pywork\\utils\\diff.py

    变成：
        src/pywork/utils/diff.py
    """
    text = str(path).replace("\\", "/").strip()

    if not text:
        return "."

    return text


def make_diff_path(
    path: str | Path,
    *,
    prefix: str | None,
) -> str:
    """
    生成 unified diff 文件头路径。

    例如：
        a/src/pywork/utils/diff.py
        b/src/pywork/utils/diff.py
        /dev/null
    """
    normalized = normalize_diff_path(path)

    if normalized == "/dev/null":
        return normalized

    if not prefix:
        return normalized

    return f"{prefix.rstrip('/')}/{normalized.lstrip('/')}"


def parse_unified_diff(diff_text: str) -> DiffStats:
    """
    使用 unidiff 解析 unified diff，并返回统计信息。

    这里不负责生成 diff。
    生成由 difflib 负责，解析统计由 unidiff 负责。
    """
    if not diff_text.strip():
        return DiffStats()

    try:
        from unidiff import PatchSet

        patch_set = PatchSet(diff_text.splitlines(keepends=True))
    except Exception as exc:  # pragma: no cover - 防御第三方解析异常
        raise DiffParseError(f"failed to parse unified diff: {exc}") from exc

    files: list[DiffFileStat] = []

    for patched_file in patch_set:
        file_stat = DiffFileStat(
            path=str(getattr(patched_file, "path", "") or ""),
            old_path=str(getattr(patched_file, "source_file", "") or "") or None,
            new_path=str(getattr(patched_file, "target_file", "") or "") or None,
            additions=int(getattr(patched_file, "added", 0) or 0),
            deletions=int(getattr(patched_file, "removed", 0) or 0),
            hunks=len(patched_file),
            is_added_file=bool(getattr(patched_file, "is_added_file", False)),
            is_removed_file=bool(getattr(patched_file, "is_removed_file", False)),
            is_renamed_file=bool(getattr(patched_file, "is_rename", False)),
            is_binary_file=bool(getattr(patched_file, "is_binary_file", False)),
        )

        files.append(file_stat)

    return DiffStats(
        files_changed=len(files),
        additions=sum(file.additions for file in files),
        deletions=sum(file.deletions for file in files),
        hunks=sum(file.hunks for file in files),
        files=tuple(files),
    )


def create_unified_diff(
    old_text: str,
    new_text: str,
    *,
    path: str | Path = "file",
    old_path: str | Path | None = None,
    new_path: str | Path | None = None,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    old_prefix: str | None = DEFAULT_OLD_PREFIX,
    new_prefix: str | None = DEFAULT_NEW_PREFIX,
    fromfiledate: str = "",
    tofiledate: str = "",
) -> UnifiedDiff:
    """
    基于两段文本生成 unified diff。

    典型输出：

        --- a/README.md
        +++ b/README.md
        @@ -1,3 +1,3 @@
         # PyWork
        -old
        +new
    """
    if context_lines < 0:
        raise ValueError("context_lines must be >= 0")

    resolved_old_path = old_path or path
    resolved_new_path = new_path or path

    old_lines = split_text_for_diff(old_text)
    new_lines = split_text_for_diff(new_text)

    if old_lines == new_lines:
        return UnifiedDiff(
            text="",
            old_path=normalize_diff_path(resolved_old_path),
            new_path=normalize_diff_path(resolved_new_path),
            stats=DiffStats(),
        )

    fromfile = make_diff_path(
        resolved_old_path,
        prefix=old_prefix,
    )
    tofile = make_diff_path(
        resolved_new_path,
        prefix=new_prefix,
    )

    diff_lines = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=fromfile,
        tofile=tofile,
        fromfiledate=fromfiledate,
        tofiledate=tofiledate,
        n=context_lines,
        lineterm="\n",
    )

    diff_text = "".join(diff_lines)

    return UnifiedDiff(
        text=diff_text,
        old_path=normalize_diff_path(resolved_old_path),
        new_path=normalize_diff_path(resolved_new_path),
        stats=parse_unified_diff(diff_text),
    )


def create_new_file_diff(
    new_text: str,
    *,
    path: str | Path,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    new_prefix: str | None = DEFAULT_NEW_PREFIX,
) -> UnifiedDiff:
    """生成新文件 diff。"""
    return create_unified_diff(
        "",
        new_text,
        path=path,
        old_path="/dev/null",
        new_path=path,
        context_lines=context_lines,
        old_prefix=None,
        new_prefix=new_prefix,
    )


def create_deleted_file_diff(
    old_text: str,
    *,
    path: str | Path,
    context_lines: int = DEFAULT_CONTEXT_LINES,
    old_prefix: str | None = DEFAULT_OLD_PREFIX,
) -> UnifiedDiff:
    """生成删除文件 diff。"""
    return create_unified_diff(
        old_text,
        "",
        path=path,
        old_path=path,
        new_path="/dev/null",
        context_lines=context_lines,
        old_prefix=old_prefix,
        new_prefix=None,
    )


def create_file_diff(
    path: str | Path,
    new_text: str,
    *,
    old_text: str | None = None,
    encoding: str = "utf-8",
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> UnifiedDiff:
    """
    基于文件当前内容和新内容生成 diff。

    old_text 为 None 时，会从磁盘读取 path 当前内容。
    """
    file_path = Path(path)

    current_text = (
        file_path.read_text(encoding=encoding)
        if old_text is None
        else old_text
    )

    return create_unified_diff(
        current_text,
        new_text,
        path=normalize_diff_path(file_path),
        context_lines=context_lines,
    )


def create_multi_file_diff(
    changes: list[DiffChange] | tuple[DiffChange, ...],
    *,
    context_lines: int = DEFAULT_CONTEXT_LINES,
) -> UnifiedDiff:
    """生成多文件 unified diff。"""
    parts: list[str] = []

    for change in changes:
        if change.is_new_file:
            file_diff = create_new_file_diff(
                change.new_text,
                path=change.new_path or change.path,
                context_lines=context_lines,
            )
        elif change.is_deleted_file:
            file_diff = create_deleted_file_diff(
                change.old_text,
                path=change.old_path or change.path,
                context_lines=context_lines,
            )
        else:
            file_diff = create_unified_diff(
                change.old_text,
                change.new_text,
                path=change.path,
                old_path=change.old_path,
                new_path=change.new_path,
                context_lines=context_lines,
            )

        if not file_diff.is_empty:
            parts.append(file_diff.text)

    diff_text = "".join(parts)

    return UnifiedDiff(
        text=diff_text,
        stats=parse_unified_diff(diff_text),
    )


def has_diff(old_text: str, new_text: str) -> bool:
    """判断两段文本是否有差异。"""
    return old_text != new_text


def compact_diff_text(
    diff_text: str,
    *,
    max_lines: int = 200,
    max_chars: int = 40_000,
) -> str:
    """
    压缩 diff 文本，避免 TUI / LLM 上下文被超长 diff 刷屏。
    """
    if max_lines <= 0:
        raise ValueError("max_lines must be > 0")

    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")

    lines = diff_text.splitlines(keepends=True)
    truncated_by_lines = len(lines) > max_lines

    compacted = "".join(lines[:max_lines])

    truncated_by_chars = len(compacted) > max_chars

    if truncated_by_chars:
        compacted = compacted[:max_chars]

    if truncated_by_lines or truncated_by_chars:
        compacted += (
            "\n"
            f"... diff truncated "
            f"(max_lines={max_lines}, max_chars={max_chars})\n"
        )

    return compacted


def render_diff_summary(stats: DiffStats) -> str:
    """把 diff 统计渲染成简短文本。"""
    if stats.is_empty:
        return "no changes"

    return (
        f"{stats.files_changed} file(s) changed, "
        f"{stats.additions} insertion(s), "
        f"{stats.deletions} deletion(s), "
        f"{stats.hunks} hunk(s)"
    )


def demo() -> None:
    old_text = "line 1\nline 2\nline 3\n"
    new_text = "line 1\nline two\nline 3\nline 4\n"

    diff = create_unified_diff(
        old_text,
        new_text,
        path="demo.txt",
    )

    print(diff.text)
    print(render_diff_summary(diff.stats))


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())