from __future__ import annotations

import time
from pathlib import Path

from pysearch.indexing.cache_manager import CacheManager
from pysearch import SearchItem, SearchResult, SearchStats


def _result(path: Path) -> SearchResult:
    item = SearchItem(file=path, start_line=1, end_line=1, lines=["x"], match_spans=[(0, (0, 1))])
    stats = SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0)
    return SearchResult(items=[item], stats=stats)


def test_disk_cache_roundtrip_and_cleanup(tmp_path: Path) -> None:
    cache_dir = tmp_path / ".cache"
    cm = CacheManager(backend="disk", cache_dir=cache_dir, default_ttl=0.1, compression=True)

    k = "k1"
    assert cm.set(k, _result(tmp_path / "a.py"))
    assert cm.get(k) is not None

    # wait to expire
    time.sleep(0.2)
    cm.cleanup_expired()
    assert cm.get(k) is None

    # stats and clear
    stats = cm.get_stats()
    assert "total_entries" in stats
    cm.clear()
    assert cm.get_stats()["total_entries"] == 0

