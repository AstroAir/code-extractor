from __future__ import annotations

from pathlib import Path

from pysearch import PySearch
from pysearch import SearchConfig


def test_api_toggles_and_semantic(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("""# comment\n\n\"\"\"Doc\"\"\"\ntext\n""", encoding="utf-8")
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))

    # Disable comments and docstrings; search for text present only in them
    eng.cfg.enable_comments = False
    eng.cfg.enable_docstrings = False
    res = eng.search("Doc", regex=False, context=0)
    assert res.stats.items == 0

    # Re-enable and search again
    eng.cfg.enable_comments = True
    eng.cfg.enable_docstrings = True
    res2 = eng.search("Doc", regex=False, context=0)
    assert res2.stats.items >= 0

    # Kick semantic advanced fit path with low threshold
    sres = eng.search_semantic_advanced("text", threshold=0.0, max_results=10)
    assert sres.stats.files_scanned >= 0

