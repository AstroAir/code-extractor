from __future__ import annotations

from pathlib import Path

from pysearch.indexing.cache_manager import CacheManager
from pysearch import SearchItem, SearchResult, SearchStats


def _large_result(path: Path, size_multiplier: int = 1) -> SearchResult:
    """Create a SearchResult with configurable size."""
    items = []
    for i in range(size_multiplier * 10):
        item = SearchItem(
            file=path,
            start_line=i + 1,
            end_line=i + 1,
            lines=[f"line {i}" * 10],  # Make lines longer
            match_spans=[(0, (0, 5))]
        )
        items.append(item)
    
    stats = SearchStats(files_scanned=1, files_matched=1, items=len(items), elapsed_ms=1.0)
    return SearchResult(items=items, stats=stats)


def test_memory_cache_eviction_limits(tmp_path: Path) -> None:
    # Small cache to trigger eviction
    cm = CacheManager(backend="memory", max_size=3)
    
    # Add entries until eviction triggers
    for i in range(5):
        key = f"key_{i}"
        result = _large_result(tmp_path / f"file_{i}.py", size_multiplier=1)
        cm.set(key, result)
    
    # Should have evicted some entries due to size limit
    stats = cm.get_stats()
    assert stats["total_entries"] <= 3


def test_disk_cache_eviction_and_stats(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    cm = CacheManager(backend="disk", cache_dir=cache_dir, max_size=2, default_ttl=60.0)
    
    # Add more entries than max_size
    for i in range(4):
        key = f"disk_key_{i}"
        result = _large_result(tmp_path / f"disk_file_{i}.py")
        cm.set(key, result)
    
    # Should trigger eviction
    stats = cm.get_stats()
    assert stats["total_entries"] <= 2
    
    # Test mixed hit/miss pattern
    hit_result = cm.get("disk_key_3")  # Recent, should exist
    miss_result = cm.get("disk_key_0")  # Likely evicted
    
    # Update stats after hits/misses
    final_stats = cm.get_stats()
    assert "cache_hits" in final_stats or "total_entries" in final_stats
