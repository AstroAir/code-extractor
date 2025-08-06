from __future__ import annotations

from pathlib import Path

from pysearch.matchers import find_text_regex_matches


def test_regex_line_col_mapping_simple() -> None:
    text = "abc\ndef\nxyz\ndefghi\n"
    # pattern 'def' occurs at line 1 (0-based) col 0 and line 3 col 0
    ms = find_text_regex_matches(text, pattern=r"def", use_regex=True)
    coords = {(m.line_index, m.start_col, m.end_col) for m in ms}
    assert (1, 0, 3) in coords
    assert (3, 0, 3) in coords
    assert len(ms) == 2


def test_regex_line_col_mapping_multiline_dotall() -> None:
    # DOTALL enables '.' to match newline; verify absolute span -> line/col mapping is correct
    text = "start\nfoo1\nbar2\nend\n"
    # Match "foo1\nbar2" as a single regex span
    ms = find_text_regex_matches(text, pattern=r"foo1.+?bar2", use_regex=True)
    assert len(ms) == 1
    m = ms[0]
    # "foo1" begins at line 1 col 0; "bar2" ends at line 2 col 4
    # end_col = start_col + length("foo1\\nbar2") = 0 + 9 = 9
    # But our representation is a single (line_index, start_col, end_col) anchored on start line.
    assert m.line_index == 1
    assert m.start_col == 0
    assert m.end_col == len("foo1\nbar2")  # 9


def test_regex_line_col_mapping_mixed_line_lengths() -> None:
    # Build lines of various lengths and search overlapping patterns
    lines = [
        "aaa bbb ccc",
        "dddd e",
        "fghij",
        "klm def nop",
        "qrstuv wxyz",
    ]
    text = "\n".join(lines)
    ms = find_text_regex_matches(text, pattern=r"def", use_regex=True)
    assert len(ms) == 1
    m = ms[0]
    # "def" lives on 0-based line 3, at column where "klm " is 4 chars
    assert m.line_index == 3
    assert m.start_col == 4
    assert m.end_col == 7