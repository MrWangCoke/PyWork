from rich.text import Text

from pywork.tui.components.diff_viewer import (
    collect_diff_viewer_stats,
    line_number_width,
    parse_hunk_header,
    parse_unified_diff_for_view,
    render_unified_diff_text,
)


def test_parse_hunk_header() -> None:
    assert parse_hunk_header("@@ -1,7 +1,8 @@") == (1, 1)
    assert parse_hunk_header("@@ -10 +20 @@ section") == (10, 20)
    assert parse_hunk_header("not a hunk") is None


def test_parse_unified_diff_for_view_addition_and_deletion() -> None:
    diff_text = (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -1,3 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+three\n"
        " four\n"
    )

    lines = parse_unified_diff_for_view(diff_text)

    additions = [line for line in lines if line.kind == "addition"]
    deletions = [line for line in lines if line.kind == "deletion"]
    contexts = [line for line in lines if line.kind == "context"]

    assert len(additions) == 1
    assert additions[0].new_lineno == 2
    assert additions[0].content == "three"

    assert len(deletions) == 1
    assert deletions[0].old_lineno == 2
    assert deletions[0].content == "two"

    assert len(contexts) == 2
    assert contexts[0].old_lineno == 1
    assert contexts[0].new_lineno == 1


def test_collect_diff_viewer_stats() -> None:
    diff_text = (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -1,2 +1,3 @@\n"
        " one\n"
        "-two\n"
        "+three\n"
        "+four\n"
    )

    lines = parse_unified_diff_for_view(diff_text)
    stats = collect_diff_viewer_stats(lines)

    assert stats.files == 1
    assert stats.hunks == 1
    assert stats.additions == 2
    assert stats.deletions == 1
    assert stats.changed_lines == 3


def test_line_number_width() -> None:
    diff_text = (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -100,2 +100,2 @@\n"
        "-old\n"
        "+new\n"
    )

    lines = parse_unified_diff_for_view(diff_text)

    assert line_number_width(lines) == 3


def test_render_unified_diff_text() -> None:
    diff_text = (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
    )

    rendered = render_unified_diff_text(diff_text)

    assert isinstance(rendered, Text)
    plain = rendered.plain

    assert "old new │ diff" in plain
    assert "- │ old" in plain
    assert "+ │ new" in plain


def test_render_empty_diff_text() -> None:
    rendered = render_unified_diff_text("")

    assert "No changes." in rendered.plain