import os
from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query, OutputFormat, ASTFilters


SAMPLE = """
def foo():
    pass

class Bar:
    def baz(self):
        return "ok"
"""


def test_smoke_search_text(tmp_path: Path):
    # Prepare temp project
    p = tmp_path / "proj" / "mod.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SAMPLE, encoding="utf-8")

    cfg = SearchConfig(paths=[str(tmp_path / "proj")], include=["**/*.py"], exclude=[])
    engine = PySearch(cfg)

    res = engine.search(pattern="def foo", regex=False, context=1, output=OutputFormat.TEXT)
    assert res.items, "expected text search to find 'def foo'"
    assert any("def foo" in "".join(it.lines) for it in res.items)


def test_smoke_search_ast(tmp_path: Path):
    p = tmp_path / "proj" / "mod.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(SAMPLE, encoding="utf-8")

    cfg = SearchConfig(paths=[str(tmp_path / "proj")], include=["**/*.py"], exclude=[])
    engine = PySearch(cfg)

    filters = ASTFilters(func_name="baz")
    q = Query(pattern="baz", use_regex=False, use_ast=True, context=1, output=OutputFormat.TEXT, filters=filters)
    res = engine.run(q)
    assert res.items, "expected ast search to find method 'baz'"
    # ensure context window contains class method
    assert any("def baz" in "".join(it.lines) for it in res.items)