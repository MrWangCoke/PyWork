from __future__ import annotations

import difflib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


FileHistoryAction = Literal[
    "create",
    "modify",
    "delete",
    "write",
    "edit",
    "rollback",
    "snapshot",
]

FileVersionTarget = Literal["before", "after"]

DEFAULT_HISTORY_DIR = ".pywork/file_history"
DEFAULT_ENCODING = "utf-8"
BINARY_CHECK_BYTES = 4096


class FileHistoryError(Exception):
    """文件历史系统基础异常。"""


class FileHistoryValidationError(FileHistoryError):
    """文件历史参数校验失败。"""


class FileHistoryStorageError(FileHistoryError):
    """文件历史读写失败。"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_history_entry_id() -> str:
    return f"fh_{uuid4().hex}"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def content_to_bytes(
    content: str | bytes | None,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> bytes | None:
    if content is None:
        return None

    if isinstance(content, bytes):
        return content

    return content.encode(encoding)


def is_probably_binary_bytes(content: bytes) -> bool:
    return b"\x00" in content[:BINARY_CHECK_BYTES]


def decode_text_for_diff(
    content: bytes | None,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> str | None:
    if content is None:
        return ""

    if is_probably_binary_bytes(content):
        return None

    try:
        return content.decode(encoding)
    except UnicodeDecodeError:
        return None


def make_unified_diff(
    before_content: bytes | None,
    after_content: bytes | None,
    *,
    path: str,
    encoding: str = DEFAULT_ENCODING,
    context_lines: int = 3,
) -> str:
    """
    为历史记录生成 unified diff。

    如果内容疑似二进制，返回空字符串。
    """
    before_text = decode_text_for_diff(
        before_content,
        encoding=encoding,
    )
    after_text = decode_text_for_diff(
        after_content,
        encoding=encoding,
    )

    if before_text is None or after_text is None:
        return ""

    if before_text == after_text:
        return ""

    fromfile = "/dev/null" if before_content is None else f"a/{path}"
    tofile = "/dev/null" if after_content is None else f"b/{path}"

    diff_lines = difflib.unified_diff(
        before_text.splitlines(keepends=True),
        after_text.splitlines(keepends=True),
        fromfile=fromfile,
        tofile=tofile,
        n=context_lines,
        lineterm="\n",
    )

    return "".join(diff_lines)


def count_diff_changes(diff_text: str) -> tuple[int, int]:
    """统计 diff 里的新增 / 删除行数。"""
    additions = 0
    deletions = 0

    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue

        if line.startswith("+"):
            additions += 1
            continue

        if line.startswith("-"):
            deletions += 1
            continue

    return additions, deletions


def atomic_write_bytes(
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = path.with_name(f"{path.name}.tmp.{uuid4().hex}")

    try:
        tmp_path.write_bytes(content)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def atomic_write_text(
    path: Path,
    content: str,
    *,
    encoding: str = DEFAULT_ENCODING,
) -> None:
    atomic_write_bytes(
        path,
        content.encode(encoding),
    )


@dataclass(slots=True, frozen=True)
class FileHistoryEntry:
    """
    一次文件变更记录。

    before_snapshot:
        修改前文件内容快照。

    after_snapshot:
        修改后文件内容快照。

    rollback 到某个版本时：
        target="before" 使用 before_snapshot
        target="after" 使用 after_snapshot
    """

    entry_id: str
    path: str
    action: str
    created_at: str

    before_exists: bool
    after_exists: bool

    before_snapshot: str | None = None
    after_snapshot: str | None = None
    diff_path: str | None = None

    before_sha256: str | None = None
    after_sha256: str | None = None

    before_size: int = 0
    after_size: int = 0

    additions: int = 0
    deletions: int = 0

    tool_name: str | None = None
    call_id: str | None = None
    reason: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def changed_lines(self) -> int:
        return self.additions + self.deletions

    @property
    def is_create(self) -> bool:
        return not self.before_exists and self.after_exists

    @property
    def is_delete(self) -> bool:
        return self.before_exists and not self.after_exists

    @property
    def is_modify(self) -> bool:
        return self.before_exists and self.after_exists

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FileHistoryEntry:
        return cls(
            entry_id=str(data["entry_id"]),
            path=str(data["path"]),
            action=str(data["action"]),
            created_at=str(data["created_at"]),
            before_exists=bool(data.get("before_exists", False)),
            after_exists=bool(data.get("after_exists", False)),
            before_snapshot=data.get("before_snapshot"),
            after_snapshot=data.get("after_snapshot"),
            diff_path=data.get("diff_path"),
            before_sha256=data.get("before_sha256"),
            after_sha256=data.get("after_sha256"),
            before_size=int(data.get("before_size", 0) or 0),
            after_size=int(data.get("after_size", 0) or 0),
            additions=int(data.get("additions", 0) or 0),
            deletions=int(data.get("deletions", 0) or 0),
            tool_name=data.get("tool_name"),
            call_id=data.get("call_id"),
            reason=data.get("reason"),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True, frozen=True)
class FileRollbackResult:
    """文件回滚结果。"""

    restored_path: str
    source_entry_id: str
    target: FileVersionTarget
    restored_exists: bool
    rollback_entry: FileHistoryEntry | None = None


class FileHistoryStore:
    """
    文件历史存储。

    存储结构：

        .pywork/file_history/
        ├── index.jsonl
        ├── snapshots/
        │   ├── fh_xxx.before.bin
        │   └── fh_xxx.after.bin
        └── diffs/
            └── fh_xxx.diff

    index.jsonl 只保存元信息。
    snapshots 保存真实文件内容。
    diffs 保存 unified diff。
    """

    def __init__(
        self,
        workspace_path: str | Path,
        *,
        history_dir: str | Path | None = None,
        encoding: str = DEFAULT_ENCODING,
    ) -> None:
        self.workspace_path = Path(workspace_path).expanduser().resolve()
        self.encoding = encoding

        if history_dir is None:
            self.history_dir = self.workspace_path / DEFAULT_HISTORY_DIR
        else:
            raw_history_dir = Path(history_dir).expanduser()
            self.history_dir = (
                raw_history_dir
                if raw_history_dir.is_absolute()
                else self.workspace_path / raw_history_dir
            ).resolve()

        self.index_path = self.history_dir / "index.jsonl"
        self.snapshots_dir = self.history_dir / "snapshots"
        self.diffs_dir = self.history_dir / "diffs"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.history_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.snapshots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.diffs_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.index_path.exists():
            self.index_path.touch()

    def resolve_workspace_file(
        self,
        path_value: str | Path,
    ) -> tuple[Path, str]:
        """
        解析 workspace 内路径。

        返回：
            (绝对路径, workspace 相对路径)
        """
        raw_path = Path(path_value).expanduser()

        if not raw_path.is_absolute():
            raw_path = self.workspace_path / raw_path

        resolved = raw_path.resolve()

        try:
            relative_path = resolved.relative_to(self.workspace_path).as_posix()
        except ValueError as exc:
            raise FileHistoryValidationError(
                f"path is outside workspace: {path_value}"
            ) from exc

        return resolved, relative_path

    def _snapshot_rel_path(
        self,
        entry_id: str,
        side: FileVersionTarget,
    ) -> str:
        return f"snapshots/{entry_id}.{side}.bin"

    def _diff_rel_path(
        self,
        entry_id: str,
    ) -> str:
        return f"diffs/{entry_id}.diff"

    def _write_snapshot(
        self,
        entry_id: str,
        side: FileVersionTarget,
        content: bytes,
    ) -> str:
        rel_path = self._snapshot_rel_path(
            entry_id,
            side,
        )
        snapshot_path = self.history_dir / rel_path

        try:
            atomic_write_bytes(
                snapshot_path,
                content,
            )
        except OSError as exc:
            raise FileHistoryStorageError(
                f"failed to write history snapshot: {exc}"
            ) from exc

        return rel_path

    def _write_diff(
        self,
        entry_id: str,
        diff_text: str,
    ) -> str | None:
        if not diff_text:
            return None

        rel_path = self._diff_rel_path(entry_id)
        diff_path = self.history_dir / rel_path

        try:
            atomic_write_text(
                diff_path,
                diff_text,
                encoding=self.encoding,
            )
        except OSError as exc:
            raise FileHistoryStorageError(f"failed to write diff: {exc}") from exc

        return rel_path

    def _append_entry(
        self,
        entry: FileHistoryEntry,
    ) -> None:
        line = json.dumps(
            entry.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

        try:
            with self.index_path.open(
                "a",
                encoding=self.encoding,
            ) as file:
                file.write(line)
                file.write("\n")
        except OSError as exc:
            raise FileHistoryStorageError(
                f"failed to append file history entry: {exc}"
            ) from exc

    def read_snapshot(
        self,
        entry: FileHistoryEntry,
        *,
        target: FileVersionTarget,
    ) -> bytes | None:
        """
        读取某个历史记录的 before / after 快照。

        如果目标版本表示文件不存在，返回 None。
        """
        exists = entry.before_exists if target == "before" else entry.after_exists
        rel_path = entry.before_snapshot if target == "before" else entry.after_snapshot

        if not exists:
            return None

        if not rel_path:
            raise FileHistoryStorageError(
                f"history entry {entry.entry_id} has no {target} snapshot"
            )

        snapshot_path = self.history_dir / rel_path

        if not snapshot_path.exists():
            raise FileHistoryStorageError(
                f"history snapshot does not exist: {snapshot_path}"
            )

        try:
            return snapshot_path.read_bytes()
        except OSError as exc:
            raise FileHistoryStorageError(
                f"failed to read history snapshot: {exc}"
            ) from exc

    def read_diff(
        self,
        entry_or_id: FileHistoryEntry | str,
    ) -> str:
        """读取某条历史记录对应的 diff。"""
        entry = (
            self.get_entry(entry_or_id)
            if isinstance(entry_or_id, str)
            else entry_or_id
        )

        if entry.diff_path is None:
            return ""

        diff_path = self.history_dir / entry.diff_path

        if not diff_path.exists():
            return ""

        try:
            return diff_path.read_text(encoding=self.encoding)
        except OSError as exc:
            raise FileHistoryStorageError(f"failed to read diff: {exc}") from exc

    def record_change(
        self,
        path: str | Path,
        *,
        before_content: str | bytes | None,
        after_content: str | bytes | None,
        action: FileHistoryAction | str | None = None,
        tool_name: str | None = None,
        call_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
        skip_unchanged: bool = True,
    ) -> FileHistoryEntry | None:
        """
        记录一次文件变更。

        before_content=None 表示修改前文件不存在。
        after_content=None 表示修改后文件不存在。

        例子：
            创建文件：
                before_content=None
                after_content="hello"

            修改文件：
                before_content="old"
                after_content="new"

            删除文件：
                before_content="old"
                after_content=None
        """
        _, relative_path = self.resolve_workspace_file(path)

        before_bytes = content_to_bytes(
            before_content,
            encoding=self.encoding,
        )
        after_bytes = content_to_bytes(
            after_content,
            encoding=self.encoding,
        )

        before_exists = before_bytes is not None
        after_exists = after_bytes is not None

        if skip_unchanged and before_exists == after_exists and before_bytes == after_bytes:
            return None

        entry_id = new_history_entry_id()

        resolved_action = action or self.infer_action(
            before_exists=before_exists,
            after_exists=after_exists,
        )

        before_snapshot = None
        after_snapshot = None

        if before_bytes is not None:
            before_snapshot = self._write_snapshot(
                entry_id,
                "before",
                before_bytes,
            )

        if after_bytes is not None:
            after_snapshot = self._write_snapshot(
                entry_id,
                "after",
                after_bytes,
            )

        diff_text = make_unified_diff(
            before_bytes,
            after_bytes,
            path=relative_path,
            encoding=self.encoding,
        )
        additions, deletions = count_diff_changes(diff_text)
        diff_path = self._write_diff(
            entry_id,
            diff_text,
        )

        entry = FileHistoryEntry(
            entry_id=entry_id,
            path=relative_path,
            action=str(resolved_action),
            created_at=utc_now_iso(),
            before_exists=before_exists,
            after_exists=after_exists,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
            diff_path=diff_path,
            before_sha256=sha256_bytes(before_bytes) if before_bytes is not None else None,
            after_sha256=sha256_bytes(after_bytes) if after_bytes is not None else None,
            before_size=len(before_bytes) if before_bytes is not None else 0,
            after_size=len(after_bytes) if after_bytes is not None else 0,
            additions=additions,
            deletions=deletions,
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            metadata=metadata or {},
        )

        self._append_entry(entry)

        return entry

    def record_current_file(
        self,
        path: str | Path,
        *,
        action: FileHistoryAction | str = "snapshot",
        tool_name: str | None = None,
        call_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileHistoryEntry:
        """
        把当前磁盘上的文件状态记录成一个版本快照。

        文件不存在时，也会记录一个 after 不存在的快照。
        """
        file_path, _ = self.resolve_workspace_file(path)
        current_content = self.read_workspace_file_bytes(file_path)

        entry = self.record_change(
            path,
            before_content=None,
            after_content=current_content,
            action=action,
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            metadata=metadata,
            skip_unchanged=False,
        )

        if entry is None:
            raise FileHistoryStorageError("failed to record current file snapshot")

        return entry

    def record_change_from_disk(
        self,
        path: str | Path,
        *,
        before_content: str | bytes | None,
        action: FileHistoryAction | str = "modify",
        tool_name: str | None = None,
        call_id: str | None = None,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileHistoryEntry | None:
        """
        用“修改前内容 + 当前磁盘内容”记录历史。

        这个后面接 file_edit / file_write 很方便：

            before = old_text
            写文件
            history.record_change_from_disk(path, before_content=before)
        """
        file_path, _ = self.resolve_workspace_file(path)
        after_content = self.read_workspace_file_bytes(file_path)

        return self.record_change(
            path,
            before_content=before_content,
            after_content=after_content,
            action=action,
            tool_name=tool_name,
            call_id=call_id,
            reason=reason,
            metadata=metadata,
        )

    def read_workspace_file_bytes(
        self,
        file_path: Path,
    ) -> bytes | None:
        if not file_path.exists():
            return None

        if not file_path.is_file():
            raise FileHistoryValidationError(f"path is not a file: {file_path}")

        try:
            return file_path.read_bytes()
        except OSError as exc:
            raise FileHistoryStorageError(f"failed to read workspace file: {exc}") from exc

    def infer_action(
        self,
        *,
        before_exists: bool,
        after_exists: bool,
    ) -> FileHistoryAction:
        if not before_exists and after_exists:
            return "create"

        if before_exists and not after_exists:
            return "delete"

        return "modify"

    def list_entries(
        self,
        *,
        path: str | Path | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[FileHistoryEntry]:
        """列出历史记录。"""
        entries: list[FileHistoryEntry] = []

        filter_path: str | None = None
        if path is not None:
            _, filter_path = self.resolve_workspace_file(path)

        try:
            with self.index_path.open(
                "r",
                encoding=self.encoding,
            ) as file:
                for line in file:
                    line = line.strip()

                    if not line:
                        continue

                    entry = FileHistoryEntry.from_dict(json.loads(line))

                    if filter_path is not None and entry.path != filter_path:
                        continue

                    entries.append(entry)
        except OSError as exc:
            raise FileHistoryStorageError(f"failed to read history index: {exc}") from exc

        if newest_first:
            entries.reverse()

        if limit is not None:
            entries = entries[:limit]

        return entries

    def get_versions(
        self,
        path: str | Path,
        *,
        newest_first: bool = False,
    ) -> list[FileHistoryEntry]:
        """获取某个文件的全部版本记录。"""
        return self.list_entries(
            path=path,
            newest_first=newest_first,
        )

    def get_entry(
        self,
        entry_id: str,
    ) -> FileHistoryEntry:
        """根据 entry_id 获取历史记录。"""
        for entry in self.list_entries():
            if entry.entry_id == entry_id:
                return entry

        raise FileHistoryValidationError(f"history entry not found: {entry_id}")

    def get_latest_entry(
        self,
        path: str | Path,
    ) -> FileHistoryEntry | None:
        entries = self.get_versions(
            path,
            newest_first=True,
        )

        return entries[0] if entries else None

    def rollback_to(
        self,
        entry_id: str,
        *,
        target: FileVersionTarget = "after",
        create_history_entry: bool = True,
        reason: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> FileRollbackResult:
        """
        回滚到任意历史版本。

        target="after":
            回到这次修改完成后的版本。

        target="before":
            回到这次修改发生前的版本。
        """
        if target not in {"before", "after"}:
            raise FileHistoryValidationError("target must be 'before' or 'after'")

        source_entry = self.get_entry(entry_id)

        file_path, relative_path = self.resolve_workspace_file(source_entry.path)

        before_rollback_content = self.read_workspace_file_bytes(file_path)
        restored_content = self.read_snapshot(
            source_entry,
            target=target,
        )

        restored_exists = restored_content is not None

        if restored_exists:
            try:
                atomic_write_bytes(
                    file_path,
                    restored_content,
                )
            except OSError as exc:
                raise FileHistoryStorageError(f"failed to restore file: {exc}") from exc
        else:
            if file_path.exists():
                if not file_path.is_file():
                    raise FileHistoryValidationError(
                        f"cannot delete non-file path during rollback: {file_path}"
                    )

                try:
                    file_path.unlink()
                except OSError as exc:
                    raise FileHistoryStorageError(
                        f"failed to delete file during rollback: {exc}"
                    ) from exc

        rollback_entry: FileHistoryEntry | None = None

        if create_history_entry:
            rollback_metadata = {
                "rollback_to_entry_id": source_entry.entry_id,
                "rollback_target": target,
                **(metadata or {}),
            }

            rollback_entry = self.record_change(
                relative_path,
                before_content=before_rollback_content,
                after_content=restored_content,
                action="rollback",
                reason=reason or f"rollback to {source_entry.entry_id}:{target}",
                metadata=rollback_metadata,
                skip_unchanged=False,
            )

        return FileRollbackResult(
            restored_path=relative_path,
            source_entry_id=source_entry.entry_id,
            target=target,
            restored_exists=restored_exists,
            rollback_entry=rollback_entry,
        )

    def clear(self) -> None:
        """
        清空历史记录。

        谨慎使用：这只清理历史，不修改 workspace 文件。
        """
        if self.history_dir.exists():
            for child in self.history_dir.rglob("*"):
                if child.is_file():
                    child.unlink()

        self._ensure_dirs()


def demo() -> None:
    workspace = Path.cwd()
    store = FileHistoryStore(workspace)

    demo_path = Path(".pywork/tmp/file_history_demo.txt")
    absolute_demo_path = workspace / demo_path
    absolute_demo_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    before = None
    after = "version 1\n"

    entry1 = store.record_change(
        demo_path,
        before_content=before,
        after_content=after,
        action="create",
        reason="demo create",
    )

    absolute_demo_path.write_text(
        after,
        encoding=DEFAULT_ENCODING,
    )

    before = after
    after = "version 2\n"

    entry2 = store.record_change(
        demo_path,
        before_content=before,
        after_content=after,
        action="modify",
        reason="demo modify",
    )

    absolute_demo_path.write_text(
        after,
        encoding=DEFAULT_ENCODING,
    )

    print("Recorded entries:")
    for entry in store.get_versions(demo_path):
        print(
            entry.entry_id,
            entry.action,
            entry.path,
            f"+{entry.additions}",
            f"-{entry.deletions}",
        )

    if entry1 is not None:
        result = store.rollback_to(entry1.entry_id)
        print(f"Rolled back to: {result.source_entry_id}:{result.target}")

    if entry2 is not None:
        print("Latest diff:")
        print(store.read_diff(entry2))


def main() -> int:
    demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())