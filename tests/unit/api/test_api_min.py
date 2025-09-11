from __future__ import annotations

from pathlib import Path

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch import Query


def test_pysearch_text_and_regex(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("class C:\n    def f(self):\n        pass\n", encoding="utf-8")
    cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False)
    eng = PySearch(cfg)

    res1 = eng.search("def ", regex=False, context=0)
    assert res1.stats.files_scanned >= 1

    res2 = eng.search("class \\w+", regex=True, context=0)
    assert res2.stats.items >= 1


def test_pysearch_run_direct(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("def a():\n    pass\n", encoding="utf-8")
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))
    q = Query(pattern="def", use_regex=False, context=0)
    res = eng.run(q)
    assert res.stats.items >= 0

