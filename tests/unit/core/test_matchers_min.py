from __future__ import annotations

from pysearch.search.matchers import find_text_regex_matches, group_matches_into_blocks


def test_find_text_and_regex_and_group():
    text = "a\nabc\nzzzabczzz\n"
    # text mode
    ms = find_text_regex_matches(text, "ab", use_regex=False)
    assert any(m.line_index == 2 for m in ms)
    # regex mode
    ms2 = find_text_regex_matches(text, r"a.+c", use_regex=True)
    blocks = group_matches_into_blocks(ms2)
    assert blocks and blocks[0][0] >= 1 and isinstance(blocks[0][2], list)

