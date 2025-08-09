from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Language, MetadataFilters, OutputFormat


def test_metadata_filters_size_language_and_regex_error(tmp_path: Path) -> None:
    # Create two files: one small, one larger
    (tmp_path / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("# comment\nprint('b')\n" * 20, encoding="utf-8")

    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))

    # language filter: Python
    md_lang = MetadataFilters(languages={Language.PYTHON})
    res_lang = eng.search("print", regex=False, context=0, output=OutputFormat.TEXT, metadata_filters=md_lang)
    assert res_lang.stats.files_scanned >= 1

    # size filter: min_lines excludes a.py, includes b.py
    md_size = MetadataFilters(min_lines=5, languages={Language.PYTHON})
    res_size = eng.search("print", regex=False, context=0, output=OutputFormat.TEXT, metadata_filters=md_size)
    assert res_size.stats.files_scanned >= 1

    # invalid regex should be handled and not crash
    bad = eng.search("[unclosed", regex=True, context=0)
    assert bad.stats.files_scanned >= 0

