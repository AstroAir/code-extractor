from __future__ import annotations

import time
from pathlib import Path

from pysearch.cache_manager import CacheManager
from pysearch.types import SearchItem, SearchResult, SearchStats


def _dummy_result(tmp_path: Path) -> SearchResult:
    item = SearchItem(file=tmp_path / "a.py", start_line=1, end_line=1, lines=["x"])  # type: ignore[arg-type]
    return SearchResult(items=[item], stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))


def test_cache_set_get_expire(tmp_path: Path) -> None:
    cm = CacheManager(backend="memory", default_ttl=0.2)
    res = _dummy_result(tmp_path)
    assert cm.set("k", res)
    assert cm.get("k") is not None
    time.sleep(0.3)
    assert cm.get("k") is None


def test_invalidate_by_pattern_and_file(tmp_path: Path) -> None:
    cm = CacheManager(backend="memory", default_ttl=10)
    r1 = _dummy_result(tmp_path)
    r2 = _dummy_result(tmp_path)
    cm.set("alpha", r1, file_dependencies={str(tmp_path / "a.py")})
    cm.set("beta", r2, file_dependencies={str(tmp_path / "b.py")})

    assert cm.invalidate_by_pattern("a*") >= 1
    # Second set invalidates by file
    cm.set("gamma", _dummy_result(tmp_path), file_dependencies={str(tmp_path / "x.py")})
    assert cm.invalidate_by_file(str(tmp_path / "x.py")) >= 1


def test_stats_updates(tmp_path: Path) -> None:
    cm = CacheManager(backend="memory", default_ttl=10)
    res = _dummy_result(tmp_path)
    cm.set("k1", res)
    cm.get("k1")
    stats = cm.get_stats()
    assert "hit_rate" in stats and 0.0 <= stats["hit_rate"] <= 1.0

