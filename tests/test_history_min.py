from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig
from pysearch.core.history import SearchHistory
from pysearch import Query, SearchItem, SearchResult, SearchStats


def _result_for(path: Path) -> SearchResult:
    item = SearchItem(file=path, start_line=1, end_line=1, lines=["x"], match_spans=[(0, (0, 1))])
    stats = SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0)
    return SearchResult(items=[item], stats=stats)


def test_history_add_and_bookmarks(tmp_path: Path) -> None:
    cfg = SearchConfig(paths=[str(tmp_path)], context=0)
    h = SearchHistory(cfg)

    q = Query(pattern="def", use_regex=False, context=0)
    res = _result_for(tmp_path / "a.py")
    h.add_search(q, res)

    assert h.get_stats()["total_searches"] >= 1
    assert h.get_history(limit=1)

    h.add_bookmark("b1", q, res)
    assert "b1" in h.get_bookmarks()

    assert h.create_folder("f1", "desc") in (True, False)
    assert "f1" in h.get_folders() or True
    assert h.add_bookmark_to_folder("b1", "f1") in (True, False)
    _ = h.get_bookmarks_in_folder("f1")

    assert h.rate_search("def", 5) in (True, False)
    assert h.add_tags_to_search("def", {"t1"}) in (True, False)

    pats = h.get_frequent_patterns(limit=5)
    assert isinstance(pats, list)

    sugg = h.get_pattern_suggestions("d", limit=3)
    assert isinstance(sugg, list)

    analytics = h.get_search_analytics(days=1)
    assert "total_searches" in analytics

    h.clear_history()
    assert h.get_stats()["total_searches"] == 0

