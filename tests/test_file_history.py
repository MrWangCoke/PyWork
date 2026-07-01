from pathlib import Path

import pytest

from pywork.storage.file_history import (
    FileHistoryStore,
    FileHistoryValidationError,
)


def test_record_create_file_history(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    entry = store.record_change(
        "hello.txt",
        before_content=None,
        after_content="hello\n",
        action="create",
        reason="test create",
    )

    assert entry is not None
    assert entry.path == "hello.txt"
    assert entry.is_create
    assert entry.after_exists
    assert not entry.before_exists
    assert entry.additions == 1
    assert entry.deletions == 0

    entries = store.get_versions("hello.txt")

    assert len(entries) == 1
    assert entries[0].entry_id == entry.entry_id

    diff = store.read_diff(entry)

    assert "--- /dev/null" in diff
    assert "+++ b/hello.txt" in diff
    assert "+hello" in diff


def test_record_modify_file_history(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    entry = store.record_change(
        "demo.txt",
        before_content="old\n",
        after_content="new\n",
        action="edit",
        tool_name="file_edit",
        call_id="call_1",
    )

    assert entry is not None
    assert entry.is_modify
    assert entry.tool_name == "file_edit"
    assert entry.call_id == "call_1"
    assert entry.additions == 1
    assert entry.deletions == 1

    diff = store.read_diff(entry)

    assert "-old" in diff
    assert "+new" in diff


def test_skip_unchanged_by_default(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    entry = store.record_change(
        "same.txt",
        before_content="same\n",
        after_content="same\n",
        action="modify",
    )

    assert entry is None
    assert store.get_versions("same.txt") == []


def test_record_delete_file_history(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    entry = store.record_change(
        "old.txt",
        before_content="old\n",
        after_content=None,
        action="delete",
    )

    assert entry is not None
    assert entry.is_delete
    assert entry.before_exists
    assert not entry.after_exists
    assert entry.additions == 0
    assert entry.deletions == 1

    diff = store.read_diff(entry)

    assert "--- a/old.txt" in diff
    assert "+++ /dev/null" in diff
    assert "-old" in diff


def test_rollback_to_after_version(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    file_path = tmp_path / "demo.txt"

    entry1 = store.record_change(
        "demo.txt",
        before_content=None,
        after_content="version 1\n",
        action="create",
    )
    assert entry1 is not None

    file_path.write_text(
        "version 1\n",
        encoding="utf-8",
    )

    entry2 = store.record_change(
        "demo.txt",
        before_content="version 1\n",
        after_content="version 2\n",
        action="modify",
    )
    assert entry2 is not None

    file_path.write_text(
        "version 2\n",
        encoding="utf-8",
    )

    result = store.rollback_to(entry1.entry_id)

    assert result.restored_path == "demo.txt"
    assert result.restored_exists
    assert file_path.read_text(encoding="utf-8") == "version 1\n"

    versions = store.get_versions("demo.txt")

    assert len(versions) == 3
    assert versions[-1].action == "rollback"


def test_rollback_to_before_delete_restores_file(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    file_path = tmp_path / "demo.txt"
    file_path.write_text(
        "alive\n",
        encoding="utf-8",
    )

    delete_entry = store.record_change(
        "demo.txt",
        before_content="alive\n",
        after_content=None,
        action="delete",
    )
    assert delete_entry is not None

    file_path.unlink()

    assert not file_path.exists()

    result = store.rollback_to(
        delete_entry.entry_id,
        target="before",
    )

    assert result.restored_exists
    assert file_path.read_text(encoding="utf-8") == "alive\n"


def test_rollback_to_after_delete_deletes_file(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    file_path = tmp_path / "demo.txt"

    delete_entry = store.record_change(
        "demo.txt",
        before_content="alive\n",
        after_content=None,
        action="delete",
    )
    assert delete_entry is not None

    file_path.write_text(
        "alive\n",
        encoding="utf-8",
    )

    result = store.rollback_to(
        delete_entry.entry_id,
        target="after",
    )

    assert not result.restored_exists
    assert not file_path.exists()


def test_record_current_file(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    file_path = tmp_path / "current.txt"
    file_path.write_text(
        "current\n",
        encoding="utf-8",
    )

    entry = store.record_current_file(
        "current.txt",
        reason="snapshot current file",
    )

    assert entry.action == "snapshot"
    assert entry.after_exists
    assert entry.after_size == len(file_path.read_bytes())


def test_reject_outside_workspace(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    with pytest.raises(FileHistoryValidationError):
        store.record_change(
            "../outside.txt",
            before_content=None,
            after_content="bad\n",
        )


def test_get_latest_entry(tmp_path: Path) -> None:
    store = FileHistoryStore(tmp_path)

    entry1 = store.record_change(
        "demo.txt",
        before_content=None,
        after_content="v1\n",
        action="create",
    )

    entry2 = store.record_change(
        "demo.txt",
        before_content="v1\n",
        after_content="v2\n",
        action="modify",
    )

    assert entry1 is not None
    assert entry2 is not None

    latest = store.get_latest_entry("demo.txt")

    assert latest is not None
    assert latest.entry_id == entry2.entry_id