from __future__ import annotations

from pathlib import Path

from pysearch.indexing.cache_manager import CacheManager
from pysearch import SearchItem, SearchResult, SearchStats


def test_disk_cache_index_save_load(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    cm = CacheManager(backend="disk", cache_dir=cache_dir, default_ttl=60.0)

    # Create dummy search results
    item1 = SearchItem(file=tmp_path / "test1.py", start_line=1, end_line=1, lines=["data1"])
    result1 = SearchResult(items=[item1], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))

    item2 = SearchItem(file=tmp_path / "test2.py", start_line=1, end_line=1, lines=["data2"])
    result2 = SearchResult(items=[item2], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))

    # Set some entries
    cm.set("k1", result1)
    cm.set("k2", result2)

    # For disk cache, index is automatically saved
    # Verify index file exists
    index_file = cache_dir / "cache_index.json"
    assert index_file.exists()

    # Create new cache manager and verify data is accessible
    cm2 = CacheManager(backend="disk", cache_dir=cache_dir, default_ttl=60.0)

    # Verify data is accessible
    assert cm2.get("k1") is not None
    assert cm2.get("k2") is not None


def test_cache_manager_invalidate_multiple_keys(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    cm = CacheManager(backend="disk", cache_dir=cache_dir, default_ttl=60.0)

    # Create dummy search results with file dependencies
    test_file = tmp_path / "test.py"

    item1 = SearchItem(file=test_file, start_line=1, end_line=1, lines=["data1"])
    result1 = SearchResult(items=[item1], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))

    item2 = SearchItem(file=test_file, start_line=2, end_line=2, lines=["data2"])
    result2 = SearchResult(items=[item2], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))

    item3 = SearchItem(file=tmp_path / "other.py", start_line=1, end_line=1, lines=["data3"])
    result3 = SearchResult(items=[item3], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))

    # Set entries with file dependencies
    cm.set("search1_test.py", result1, file_dependencies={str(test_file)})
    cm.set("search2_test.py", result2, file_dependencies={str(test_file)})
    cm.set("other_file", result3)

    # Invalidate by file pattern
    cm.invalidate_by_file(str(test_file))

    # Check that file-related entries are gone but others remain
    assert cm.get("search1_test.py") is None
    assert cm.get("search2_test.py") is None
    assert cm.get("other_file") is not None
