from rich.text import Text

from pywork.tui.components.diff import (
    DiffPanel,
    collect_stats,
    line_number_width,
    parse_hunk_header,
    parse_unified_diff,
    render_summary,
    render_unified_diff,
)


def make_diff() -> str:
    return (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -1,3 +1,4 @@\n"
        " one\n"
        "-two\n"
        "+three\n"
        " four\n"
        "+five\n"
    )


def test_parse_hunk_header() -> None:
    assert parse_hunk_header("@@ -1,3 +1,4 @@") == (1, 1)
    assert parse_hunk_header("@@ -10 +20 @@ section") == (10, 20)
    assert parse_hunk_header("not a hunk") is None


def test_parse_unified_diff_line_numbers() -> None:
    lines = parse_unified_diff(make_diff())

    additions = [line for line in lines if line.kind == "addition"]
    deletions = [line for line in lines if line.kind == "deletion"]
    contexts = [line for line in lines if line.kind == "context"]

    assert len(additions) == 2
    assert additions[0].new_lineno == 2
    assert additions[0].content == "three"

    assert additions[1].new_lineno == 4
    assert additions[1].content == "five"

    assert len(deletions) == 1
    assert deletions[0].old_lineno == 2
    assert deletions[0].content == "two"

    assert len(contexts) == 2
    assert contexts[0].old_lineno == 1
    assert contexts[0].new_lineno == 1


def test_collect_stats() -> None:
    lines = parse_unified_diff(make_diff())
    stats = collect_stats(lines)

    assert stats.files == 1
    assert stats.hunks == 1
    assert stats.additions == 2
    assert stats.deletions == 1
    assert stats.changed_lines == 3


def test_line_number_width() -> None:
    diff = (
        "--- a/demo.txt\n"
        "+++ b/demo.txt\n"
        "@@ -100,2 +100,2 @@\n"
        "-old\n"
        "+new\n"
    )

    lines = parse_unified_diff(diff)

    assert line_number_width(lines) == 3


def test_render_summary() -> None:
    lines = parse_unified_diff(make_diff())
    stats = collect_stats(lines)

    summary = render_summary(stats)

    assert isinstance(summary, Text)
    assert "1 file(s)" in summary.plain
    assert "+2" in summary.plain
    assert "-1" in summary.plain


def test_render_unified_diff() -> None:
    rendered = render_unified_diff(make_diff())

    assert isinstance(rendered, Text)

    plain = rendered.plain

    assert "old new   diff" in plain
    assert "two" in plain
    assert "three" in plain
    assert "five" in plain


def test_parse_empty_diff() -> None:
    lines = parse_unified_diff("")

    assert len(lines) == 1
    assert lines[0].kind == "empty"
    assert lines[0].content == "No changes."


def test_diff_panel_get_stats() -> None:
    panel = DiffPanel(make_diff())
    stats = panel.get_stats()

    assert stats.files == 1
    assert stats.additions == 2
    assert stats.deletions == 1