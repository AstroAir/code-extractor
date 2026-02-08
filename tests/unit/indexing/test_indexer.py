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

    def test_update_file_new(self, tmp_path: Path):
        """update_file should add a new file to the index."""
        f = tmp_path / "new.py"
        f.write_text("a = 1", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.load()
        assert indexer.count_indexed() == 0
        result = indexer.update_file(f)
        assert result is True
        assert indexer.count_indexed() == 1

    def test_update_file_existing(self, tmp_path: Path):
        """update_file should update metadata for an already-indexed file."""
        f = tmp_path / "exist.py"
        f.write_text("a = 1", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        count_before = indexer.count_indexed()
        # Modify the file
        f.write_text("a = 2", encoding="utf-8")
        result = indexer.update_file(f)
        assert result is True
        assert indexer.count_indexed() == count_before  # no new entry

    def test_update_file_nonexistent(self, tmp_path: Path):
        """update_file should return False for a file that doesn't exist."""
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        result = indexer.update_file(tmp_path / "missing.py")
        assert result is False

    def test_remove_file(self, tmp_path: Path):
        """remove_file should remove a file from the index."""
        f = tmp_path / "rm.py"
        f.write_text("pass", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        assert indexer.count_indexed() >= 1
        result = indexer.remove_file(f)
        assert result is True
        assert indexer.count_indexed() == 0

    def test_remove_file_not_indexed(self, tmp_path: Path):
        """remove_file should return False for a file not in the index."""
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        result = indexer.remove_file(tmp_path / "never_added.py")
        assert result is False

    def test_clear(self, tmp_path: Path):
        """clear should remove all in-memory index data."""
        (tmp_path / "a.py").write_text("x = 1", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        assert indexer.count_indexed() >= 1
        indexer.save()
        indexer.clear()
        assert indexer.count_indexed() == 0

    def test_cleanup_old_entries(self, tmp_path: Path):
        """cleanup_old_entries should remove entries not accessed recently."""
        (tmp_path / "old.py").write_text("x = 1", encoding="utf-8")
        (tmp_path / "new.py").write_text("y = 2", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        count_before = indexer.count_indexed()
        assert count_before >= 2
        # All entries are fresh, so cleanup with days_old=0 should remove all
        # but with days_old=30 should remove none
        removed = indexer.cleanup_old_entries(days_old=30)
        assert removed == 0
        assert indexer.count_indexed() == count_before

    def test_cleanup_old_entries_removes_stale(self, tmp_path: Path):
        """cleanup_old_entries should remove stale entries."""
        (tmp_path / "stale.py").write_text("x = 1", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], cache_dir=tmp_path / "cache")
        indexer = Indexer(cfg)
        indexer.scan()
        assert indexer.count_indexed() >= 1
        # Manually set last_accessed to a very old time
        for rec in indexer._index.values():
            rec.last_accessed = 0.0
        removed = indexer.cleanup_old_entries(days_old=1)
        assert removed >= 1
        assert indexer.count_indexed() == 0
