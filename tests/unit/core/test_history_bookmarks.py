"""Tests for pysearch.core.history.history_bookmarks module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.history.history_bookmarks import BookmarkFolder, BookmarkManager
from pysearch.core.types import OutputFormat, Query, SearchResult, SearchStats


def _make_result() -> SearchResult:
    return SearchResult(
        items=[],
        stats=SearchStats(files_scanned=5, files_matched=2, items=3, elapsed_ms=100.0, indexed_files=50),
    )


class TestBookmarkFolder:
    """Tests for BookmarkFolder dataclass."""

    def test_creation(self):
        f = BookmarkFolder(name="work", description="Work searches")
        assert f.name == "work"
        assert f.description == "Work searches"
        assert f.created_time is not None
        assert f.bookmarks == set()

    def test_defaults(self):
        f = BookmarkFolder(name="test")
        assert f.description is None
        assert f.bookmarks is not None
        assert len(f.bookmarks) == 0


class TestBookmarkManager:
    """Tests for BookmarkManager class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr._loaded is False

    def test_add_and_get_bookmark(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        q = Query(pattern="find_main")
        r = _make_result()
        mgr.add_bookmark("main_search", q, r)
        bookmarks = mgr.get_bookmarks()
        assert "main_search" in bookmarks
        assert bookmarks["main_search"].query_pattern == "find_main"

    def test_remove_bookmark(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        assert mgr.remove_bookmark("bm1") is True
        assert "bm1" not in mgr.get_bookmarks()

    def test_remove_bookmark_nonexistent(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.remove_bookmark("nonexistent") is False

    def test_remove_bookmark_also_removes_from_folders(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        mgr.create_folder("f1")
        mgr.add_bookmark_to_folder("bm1", "f1")
        mgr.remove_bookmark("bm1")
        folder = mgr.get_folders()["f1"]
        assert "bm1" not in (folder.bookmarks or set())

    def test_create_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.create_folder("work", "Work searches") is True
        folders = mgr.get_folders()
        assert "work" in folders
        assert folders["work"].description == "Work searches"

    def test_create_folder_duplicate(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.create_folder("work") is True
        assert mgr.create_folder("work") is False

    def test_delete_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.create_folder("temp")
        assert mgr.delete_folder("temp") is True
        assert "temp" not in mgr.get_folders()

    def test_delete_folder_nonexistent(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.delete_folder("nonexistent") is False

    def test_add_bookmark_to_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        mgr.create_folder("f1")
        assert mgr.add_bookmark_to_folder("bm1", "f1") is True

    def test_add_bookmark_to_folder_invalid_bookmark(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.create_folder("f1")
        assert mgr.add_bookmark_to_folder("nonexistent", "f1") is False

    def test_add_bookmark_to_folder_invalid_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        assert mgr.add_bookmark_to_folder("bm1", "nonexistent") is False

    def test_remove_bookmark_from_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        mgr.create_folder("f1")
        mgr.add_bookmark_to_folder("bm1", "f1")
        assert mgr.remove_bookmark_from_folder("bm1", "f1") is True
        assert len(mgr.get_bookmarks_in_folder("f1")) == 0

    def test_remove_bookmark_from_folder_nonexistent_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.remove_bookmark_from_folder("bm1", "nonexistent") is False

    def test_get_bookmarks_in_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="find_x"), _make_result())
        mgr.add_bookmark("bm2", Query(pattern="find_y"), _make_result())
        mgr.create_folder("f1")
        mgr.add_bookmark_to_folder("bm1", "f1")
        mgr.add_bookmark_to_folder("bm2", "f1")
        bms = mgr.get_bookmarks_in_folder("f1")
        assert len(bms) == 2

    def test_get_bookmarks_in_folder_nonexistent(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        assert mgr.get_bookmarks_in_folder("nonexistent") == []

    def test_get_bookmarks_in_empty_folder(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.create_folder("empty")
        assert mgr.get_bookmarks_in_folder("empty") == []

    def test_search_bookmarks(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("find_functions", Query(pattern="def"), _make_result())
        mgr.add_bookmark("find_classes", Query(pattern="class"), _make_result())
        results = mgr.search_bookmarks("find")
        assert len(results) == 2

    def test_search_bookmarks_by_pattern(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="def main"), _make_result())
        results = mgr.search_bookmarks("main")
        assert len(results) == 1

    def test_search_bookmarks_no_match(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="def"), _make_result())
        results = mgr.search_bookmarks("xyz")
        assert results == []

    def test_get_bookmark_stats(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr = BookmarkManager(cfg)
        mgr.add_bookmark("bm1", Query(pattern="x"), _make_result())
        mgr.create_folder("f1")
        mgr.add_bookmark_to_folder("bm1", "f1")
        stats = mgr.get_bookmark_stats()
        assert stats["total_bookmarks"] == 1
        assert stats["total_folders"] == 1
        assert stats["bookmarks_in_folders"] == 1

    def test_persistence(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        mgr1 = BookmarkManager(cfg)
        mgr1.add_bookmark("persist", Query(pattern="test"), _make_result())
        mgr1.create_folder("pf")
        mgr1.add_bookmark_to_folder("persist", "pf")

        mgr2 = BookmarkManager(cfg)
        mgr2.load()
        assert "persist" in mgr2.get_bookmarks()
        assert "pf" in mgr2.get_folders()
        assert len(mgr2.get_bookmarks_in_folder("pf")) == 1
