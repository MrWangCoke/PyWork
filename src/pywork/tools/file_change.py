from __future__ import annotations

import difflib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from pywork.tools.tool import ToolExecutionError, ToolValidationError


FileChangeOperation = Literal["write", "edit"]


BINARY_CHECK_BYTES = 4096
DEFAULT_DIFF_CONTEXT_LINES = 3
DEFAULT_MAX_DIFF_LINES = 400
DEFAULT_MAX_DIFF_CHARS = 80_000


@dataclass(slots=True, frozen=True)
class FileChangeStats:
    additions: int = 0
    deletions: int = 0
    diff_lines: int = 0
    truncated_by_lines: bool = False
    truncated_by_chars: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class FileChangePreview:
    """
    文件修改预览。

    注意：
    preview 阶段不写文件。
    这里只描述“将要发生什么变化”。
    """

    operation: FileChangeOperation
    path: str
    absolute_path: str
    workspace_path: str

    old_exists: bool
    new_exists: bool

    old_content: str
    new_content: str

    encoding: str
    create_dirs: bool = False
    overwrite: bool = True

    diff_text: str = ""
    diff_summary: str = ""
    stats: FileChangeStats = field(default_factory=FileChangeStats)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return self.old_content != self.new_content

    @property
    def is_create(self) -> bool:
        return not self.old_exists and self.new_exists

    @property
    def is_modify(self) -> bool:
        return self.old_exists and self.new_exists and self.has_changes

    @property
    def old_size(self) -> int:
        return len(self.old_content.encode(self.encoding, errors="replace"))

    @property
    def new_size(self) -> int:
        return len(self.new_content.encode(self.encoding, errors="replace"))

    def to_dict(self, *, include_content: bool = False) -> dict[str, Any]:
        data = {
            "operation": self.operation,
            "path": self.path,
            "absolute_path": self.absolute_path,
            "workspace_path": self.workspace_path,
            "old_exists": self.old_exists,
            "new_exists": self.new_exists,
            "encoding": self.encoding,
            "create_dirs": self.create_dirs,
            "overwrite": self.overwrite,
            "has_changes": self.has_changes,
            "is_create": self.is_create,
            "is_modify": self.is_modify,
            "old_size": self.old_size,
            "new_size": self.new_size,
            "diff_text": self.diff_text,
            "diff_summary": self.diff_summary,
            "stats": self.stats.to_dict(),
            "metadata": self.metadata,
        }

        if include_content:
            data["old_content"] = self.old_content
            data["new_content"] = self.new_content

        return data


@dataclass(slots=True, frozen=True)
class FileChangeApplyResult:
    operation: FileChangeOperation
    path: str
    absolute_path: str
    workspace_path: str

    applied: bool
    changed: bool

    old_size: int
    new_size: int

    diff_text: str = ""
    diff_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_probably_binary_bytes(data: bytes) -> bool:
    return b"\x00" in data[:BINARY_CHECK_BYTES]


def read_text_file_for_change(
    path: Path,
    *,
    encoding: str,
) -> str:
    if not path.exists():
        return ""

    if not path.is_file():
        raise ToolValidationError(f"path is not a file: {path}")

    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ToolExecutionError(f"failed to read file: {exc}") from exc

    if is_probably_binary_bytes(raw):
        raise ToolValidationError(f"file appears to be binary: {path}")

    try:
        return raw.decode(encoding)
    except UnicodeError as exc:
        raise ToolExecutionError(
            f"failed to decode file with encoding {encoding!r}: {exc}"
        ) from exc


def split_lines_for_diff(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def count_diff_changes(diff_text: str) -> tuple[int, int, int]:
    additions = 0
    deletions = 0
    lines = diff_text.splitlines()

    for line in lines:
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            additions += 1
        elif line.startswith("-"):
            deletions += 1

    return additions, deletions, len(lines)


def limit_diff_text(
    diff_text: str,
    *,
    max_lines: int,
    max_chars: int,
) -> tuple[str, bool, bool]:
    truncated_by_lines = False
    truncated_by_chars = False

    lines = diff_text.splitlines(keepends=True)

    if max_lines > 0 and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines.append("\n... diff truncated by max_diff_lines ...\n")
        truncated_by_lines = True

    text = "".join(lines)

    if max_chars > 0 and len(text) > max_chars:
        suffix = "\n... diff truncated by max_diff_chars ...\n"
        allowed = max(0, max_chars - len(suffix))
        text = text[:allowed] + suffix
        truncated_by_chars = True

    return text, truncated_by_lines, truncated_by_chars


def create_unified_diff_text(
    *,
    path: str,
    old_content: str,
    new_content: str,
    old_exists: bool,
    context_lines: int = DEFAULT_DIFF_CONTEXT_LINES,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
) -> tuple[str, FileChangeStats, str]:
    fromfile = f"a/{path}" if old_exists else "/dev/null"
    tofile = f"b/{path}"

    diff_lines = list(
        difflib.unified_diff(
            split_lines_for_diff(old_content),
            split_lines_for_diff(new_content),
            fromfile=fromfile,
            tofile=tofile,
            lineterm="",
            n=context_lines,
        )
    )

    diff_text = "\n".join(diff_lines)

    if diff_text:
        diff_text += "\n"

    diff_text, truncated_by_lines, truncated_by_chars = limit_diff_text(
        diff_text,
        max_lines=max_diff_lines,
        max_chars=max_diff_chars,
    )

    additions, deletions, diff_line_count = count_diff_changes(diff_text)

    stats = FileChangeStats(
        additions=additions,
        deletions=deletions,
        diff_lines=diff_line_count,
        truncated_by_lines=truncated_by_lines,
        truncated_by_chars=truncated_by_chars,
    )

    summary = render_file_change_summary(
        path=path,
        old_exists=old_exists,
        changed=old_content != new_content,
        stats=stats,
    )

    return diff_text, stats, summary


def render_file_change_summary(
    *,
    path: str,
    old_exists: bool,
    changed: bool,
    stats: FileChangeStats,
) -> str:
    if not changed:
        return f"No changes for {path}"

    action = "Create" if not old_exists else "Modify"

    suffix = ""

    if stats.truncated_by_lines or stats.truncated_by_chars:
        suffix = " (diff truncated)"

    return (
        f"{action} {path}: "
        f"+{stats.additions} -{stats.deletions}"
        f"{suffix}"
    )


def build_file_change_preview(
    *,
    operation: FileChangeOperation,
    path: str,
    file_path: Path,
    workspace_path: Path,
    old_content: str,
    new_content: str,
    old_exists: bool,
    encoding: str,
    create_dirs: bool = False,
    overwrite: bool = True,
    max_diff_lines: int = DEFAULT_MAX_DIFF_LINES,
    max_diff_chars: int = DEFAULT_MAX_DIFF_CHARS,
    metadata: dict[str, Any] | None = None,
) -> FileChangePreview:
    diff_text, stats, summary = create_unified_diff_text(
        path=path,
        old_content=old_content,
        new_content=new_content,
        old_exists=old_exists,
        max_diff_lines=max_diff_lines,
        max_diff_chars=max_diff_chars,
    )

    return FileChangePreview(
        operation=operation,
        path=path,
        absolute_path=str(file_path),
        workspace_path=str(workspace_path),
        old_exists=old_exists,
        new_exists=True,
        old_content=old_content,
        new_content=new_content,
        encoding=encoding,
        create_dirs=create_dirs,
        overwrite=overwrite,
        diff_text=diff_text,
        diff_summary=summary,
        stats=stats,
        metadata=metadata or {},
    )


def apply_file_change_preview(
    preview: FileChangePreview,
) -> FileChangeApplyResult:
    file_path = Path(preview.absolute_path)

    if preview.create_dirs:
        file_path.parent.mkdir(parents=True, exist_ok=True)

    if not file_path.parent.exists():
        raise ToolValidationError(f"parent directory does not exist: {file_path.parent}")

    try:
        file_path.write_text(
            preview.new_content,
            encoding=preview.encoding,
            newline="",
        )
    except OSError as exc:
        raise ToolExecutionError(f"failed to write file: {exc}") from exc

    return FileChangeApplyResult(
        operation=preview.operation,
        path=preview.path,
        absolute_path=preview.absolute_path,
        workspace_path=preview.workspace_path,
        applied=True,
        changed=preview.has_changes,
        old_size=preview.old_size,
        new_size=preview.new_size,
        diff_text=preview.diff_text,
        diff_summary=preview.diff_summary,
        metadata={
            **preview.metadata,
            "stats": preview.stats.to_dict(),
        },
    )