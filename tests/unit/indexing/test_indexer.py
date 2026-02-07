"""Tests for pysearch.indexing.indexer module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.indexing.indexer import IndexRecord, Indexer


class TestIndexRecord:
    """Tests for IndexRecord dataclass."""

    def test_creation(self):
        r = IndexRecord(path="test.py", size=100, mtime=1.0, sha1="abc")
        assert r.path == "test.py"
        assert r.size == 100
        assert r.mtime == 1.0
        assert r.sha1 == "abc"

    def test_defaults(self):
        r = IndexRecord(path="x.py", size=0, mtime=0.0, sha1=None)
        assert r.last_accessed == 0.0
        assert r.access_count == 0


class TestIndexer:
    """Tests for Indexer class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        assert indexer.cfg is cfg

    def test_scan_empty_dir(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        assert indexer.count_indexed() == 0

    def test_scan_with_files(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "b.py").write_text("y = 2", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        assert indexer.count_indexed() >= 2

    def test_iter_all_paths(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        paths = list(indexer.iter_all_paths())
        assert len(paths) >= 1

    def test_save_and_load(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("hello", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer1 = Indexer(cfg)
        indexer1.scan()
        indexer1.save()

        indexer2 = Indexer(cfg)
        indexer2.load()
        assert indexer2.count_indexed() >= 1

    def test_get_cache_stats(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        stats = indexer.get_cache_stats()
        assert isinstance(stats, dict)
