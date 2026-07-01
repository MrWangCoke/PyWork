from pywork.utils.diff import (
    DiffChange,
    compact_diff_text,
    create_deleted_file_diff,
    create_multi_file_diff,
    create_new_file_diff,
    create_unified_diff,
    has_diff,
    parse_unified_diff,
    render_diff_summary,
)


def test_create_unified_diff_basic_change() -> None:
    diff = create_unified_diff(
        "line 1\nline 2\nline 3\n",
        "line 1\nline two\nline 3\nline 4\n",
        path="README.md",
    )

    assert "--- a/README.md" in diff.text
    assert "+++ b/README.md" in diff.text
    assert "-line 2" in diff.text
    assert "+line two" in diff.text
    assert "+line 4" in diff.text

    assert diff.stats.files_changed == 1
    assert diff.stats.additions == 2
    assert diff.stats.deletions == 1
    assert diff.stats.hunks == 1


def test_create_unified_diff_no_change() -> None:
    diff = create_unified_diff(
        "same\n",
        "same\n",
        path="same.txt",
    )

    assert diff.is_empty
    assert diff.text == ""
    assert diff.stats.is_empty


def test_create_new_file_diff() -> None:
    diff = create_new_file_diff(
        "hello\nworld\n",
        path="new.txt",
    )

    assert "--- /dev/null" in diff.text
    assert "+++ b/new.txt" in diff.text
    assert "+hello" in diff.text
    assert "+world" in diff.text
    assert diff.stats.additions == 2
    assert diff.stats.deletions == 0


def test_create_deleted_file_diff() -> None:
    diff = create_deleted_file_diff(
        "old\ncontent\n",
        path="old.txt",
    )

    assert "--- a/old.txt" in diff.text
    assert "+++ /dev/null" in diff.text
    assert "-old" in diff.text
    assert "-content" in diff.text
    assert diff.stats.additions == 0
    assert diff.stats.deletions == 2


def test_create_multi_file_diff() -> None:
    diff = create_multi_file_diff(
        [
            DiffChange(
                path="a.txt",
                old_text="a\n",
                new_text="aa\n",
            ),
            DiffChange(
                path="b.txt",
                old_text="",
                new_text="b\n",
                is_new_file=True,
            ),
        ]
    )

    assert "--- a/a.txt" in diff.text
    assert "+++ b/a.txt" in diff.text
    assert "+++ b/b.txt" in diff.text
    assert diff.stats.files_changed == 2
    assert diff.stats.additions == 2
    assert diff.stats.deletions == 1


def test_parse_unified_diff() -> None:
    diff = create_unified_diff(
        "one\ntwo\n",
        "one\nthree\n",
        path="demo.txt",
    )

    stats = parse_unified_diff(diff.text)

    assert stats.files_changed == 1
    assert stats.additions == 1
    assert stats.deletions == 1


def test_compact_diff_text() -> None:
    text = "".join(f"line {index}\n" for index in range(20))

    compacted = compact_diff_text(
        text,
        max_lines=5,
        max_chars=1_000,
    )

    assert "line 0" in compacted
    assert "line 4" in compacted
    assert "line 5" not in compacted
    assert "diff truncated" in compacted


def test_render_diff_summary() -> None:
    diff = create_unified_diff(
        "a\n",
        "b\n",
        path="demo.txt",
    )

    summary = render_diff_summary(diff.stats)

    assert "1 file(s) changed" in summary
    assert "1 insertion(s)" in summary
    assert "1 deletion(s)" in summary


def test_has_diff() -> None:
    assert has_diff("a", "b")
    assert not has_diff("a", "a")