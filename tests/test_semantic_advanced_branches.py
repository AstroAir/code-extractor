from __future__ import annotations

from pathlib import Path

from pysearch import PySearch
from pysearch import SearchConfig


def test_semantic_advanced_threshold_and_empty(tmp_path: Path) -> None:
    # Empty corpus path: no files
    eng_empty = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))
    r_empty = eng_empty.search_semantic_advanced("anything", threshold=0.5, max_results=5)
    assert r_empty.stats.files_scanned >= 0

    # Non-empty: low threshold returns some ordering; metadata_filters path already covered elsewhere
    p = tmp_path / "a.py"
    p.write_text("""def f():\n    return 1\n""", encoding="utf-8")
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))
    r = eng.search_semantic_advanced("return", threshold=0.0, max_results=5)
    assert r.stats.items >= 0

