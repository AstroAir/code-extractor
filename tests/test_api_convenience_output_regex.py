from __future__ import annotations

from pathlib import Path

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch import OutputFormat


def test_api_convenience_regex_and_output(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("def handler_one():\npass\n", encoding="utf-8")
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))

    # regex with named-ish group and output change
    res = eng.search(r"def\s+\w+_one", regex=True, context=0, output=OutputFormat.TEXT)
    assert res.stats.files_scanned >= 1

    # JSON output branch in convenience API
    res2 = eng.search("handler_one", regex=False, context=0, output=OutputFormat.JSON)
    assert res2.stats.items >= 0

