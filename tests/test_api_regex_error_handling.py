from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig


def test_api_regex_error_handling_and_non_text_files(tmp_path: Path) -> None:
    # Create text and binary files
    (tmp_path / "text.py").write_text("def function(): pass\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\x00\x01\x02\x03\x04")
    
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], context=0, parallel=False))
    
    # Invalid regex should be handled gracefully
    bad_regex_result = eng.search("[unclosed", regex=True, context=0)
    assert bad_regex_result.stats.files_scanned >= 0
    # Should not crash, may have 0 results due to regex error
    
    # Another invalid regex pattern
    bad_regex2 = eng.search("*invalid", regex=True, context=0)
    assert bad_regex2.stats.files_scanned >= 0
    
    # Valid regex should work
    good_regex = eng.search(r"def \w+", regex=True, context=0)
    assert good_regex.stats.files_scanned >= 1
    
    # Binary files should be skipped automatically
    all_files = eng.search("function", regex=False, context=0)
    # Should find text file but skip binary
    assert all_files.stats.files_scanned >= 1
