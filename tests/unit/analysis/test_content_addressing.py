"""Tests for pysearch.analysis.content_addressing module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.analysis.content_addressing import (
    ContentAddress,
    ContentAddressedIndexer,
    GlobalCacheManager,
    IndexingProgressUpdate,
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)
from pysearch.core.config import SearchConfig
from pysearch.core.types import Language


# ---------------------------------------------------------------------------
# ContentAddress
# ---------------------------------------------------------------------------
class TestContentAddress:
    """Tests for ContentAddress dataclass."""

    def test_creation(self):
        ca = ContentAddress(
            path="test.py",
            content_hash="abc123",
            size=100,
            mtime=1.0,
        )
        assert ca.path == "test.py"
        assert ca.content_hash == "abc123"
        assert ca.size == 100
        assert ca.mtime == 1.0

    def test_default_language(self):
        ca = ContentAddress(path="x.py", content_hash="h", size=0, mtime=0.0)
        assert ca.language == Language.UNKNOWN

    def test_frozen(self):
        ca = ContentAddress(path="x.py", content_hash="h", size=0, mtime=0.0)
        with pytest.raises(AttributeError):
            ca.path = "other.py"  # type: ignore[misc]

    async def test_from_file(self, tmp_path: Path):
        f = tmp_path / "hello.py"
        f.write_text("print('hello')\n", encoding="utf-8")
        ca = await ContentAddress.from_file(str(f))
        assert len(ca.content_hash) == 64  # SHA256 hex digest length
        assert ca.size > 0
        assert ca.path == str(f)
        assert ca.language == Language.PYTHON
        # Verify deterministic: same file → same hash
        ca2 = await ContentAddress.from_file(str(f))
        assert ca.content_hash == ca2.content_hash

    async def test_from_file_nonexistent(self, tmp_path: Path):
        with pytest.raises((FileNotFoundError, OSError, Exception)):  # noqa: B017
            await ContentAddress.from_file(str(tmp_path / "missing.py"))


# ---------------------------------------------------------------------------
# IndexTag
# ---------------------------------------------------------------------------
class TestIndexTag:
    """Tests for IndexTag dataclass."""

    def test_creation(self):
        tag = IndexTag(directory="/repo", branch="main", artifact_id="chunks")
        assert tag.directory == "/repo"
        assert tag.branch == "main"
        assert tag.artifact_id == "chunks"

    def test_equality(self):
        t1 = IndexTag(directory="/a", branch="main", artifact_id="x")
        t2 = IndexTag(directory="/a", branch="main", artifact_id="x")
        assert t1 == t2

    def test_inequality(self):
        t1 = IndexTag(directory="/a", branch="main", artifact_id="x")
        t2 = IndexTag(directory="/a", branch="dev", artifact_id="x")
        assert t1 != t2

    def test_to_string(self):
        tag = IndexTag(directory="/repo", branch="main", artifact_id="chunks")
        assert tag.to_string() == "/repo::main::chunks"

    def test_from_string(self):
        tag = IndexTag.from_string("/repo::main::chunks")
        assert tag.directory == "/repo"
        assert tag.branch == "main"
        assert tag.artifact_id == "chunks"

    def test_from_string_roundtrip(self):
        original = IndexTag(directory="/a/b", branch="feat/x", artifact_id="idx")
        restored = IndexTag.from_string(original.to_string())
        assert restored == original

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Invalid tag string"):
            IndexTag.from_string("bad_format")

    def test_frozen(self):
        tag = IndexTag(directory="/a", branch="main", artifact_id="x")
        with pytest.raises(AttributeError):
            tag.directory = "/b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PathAndCacheKey
# ---------------------------------------------------------------------------
class TestPathAndCacheKey:
    """Tests for PathAndCacheKey dataclass."""

    def test_creation(self):
        pk = PathAndCacheKey(path="/a/b.py", cache_key="sha256hash")
        assert pk.path == "/a/b.py"
        assert pk.cache_key == "sha256hash"

    def test_mutable(self):
        pk = PathAndCacheKey(path="a", cache_key="b")
        pk.path = "c"
        assert pk.path == "c"


# ---------------------------------------------------------------------------
# RefreshIndexResults
# ---------------------------------------------------------------------------
class TestRefreshIndexResults:
    """Tests for RefreshIndexResults dataclass."""

    def test_creation_empty(self):
        results = RefreshIndexResults(compute=[], delete=[], add_tag=[], remove_tag=[])
        assert results.compute == []
        assert results.delete == []
        assert results.add_tag == []
        assert results.remove_tag == []

    def test_creation_with_items(self):
        pk = PathAndCacheKey(path="x.py", cache_key="h1")
        results = RefreshIndexResults(compute=[pk], delete=[], add_tag=[], remove_tag=[])
        assert len(results.compute) == 1
        assert results.compute[0].path == "x.py"


# ---------------------------------------------------------------------------
# IndexingProgressUpdate
# ---------------------------------------------------------------------------
class TestIndexingProgressUpdate:
    """Tests for IndexingProgressUpdate dataclass."""

    def test_creation(self):
        update = IndexingProgressUpdate(
            progress=0.5, description="Processing files", status="indexing"
        )
        assert update.progress == 0.5
        assert update.description == "Processing files"
        assert update.status == "indexing"
        assert update.warnings is None
        assert update.debug_info is None

    def test_with_optional_fields(self):
        update = IndexingProgressUpdate(
            progress=1.0,
            description="Done",
            status="done",
            warnings=["warn1"],
            debug_info="info",
        )
        assert update.warnings == ["warn1"]
        assert update.debug_info == "info"


# ---------------------------------------------------------------------------
# GlobalCacheManager
# ---------------------------------------------------------------------------
class TestGlobalCacheManager:
    """Tests for GlobalCacheManager class."""

    def test_init(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        assert mgr.cache_dir == tmp_path
        assert mgr.db_path == tmp_path / "global_cache.db"

    async def test_store_and_get(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("hash123", "chunks", {"key": "value"}, [tag])
        result = await mgr.get_cached_content("hash123", "chunks")
        assert result == {"key": "value"}

    async def test_get_nonexistent(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        result = await mgr.get_cached_content("nonexistent", "chunks")
        assert result is None

    async def test_store_overwrites_existing(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag])
        await mgr.store_cached_content("h1", "chunks", {"v": 2}, [tag])
        result = await mgr.get_cached_content("h1", "chunks")
        assert result == {"v": 2}

    async def test_get_tags_for_content(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag1 = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        tag2 = IndexTag(directory="/a", branch="dev", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag1, tag2])
        tags = await mgr.get_tags_for_content("h1", "chunks")
        assert len(tags) == 2
        tag_strings = {t.to_string() for t in tags}
        assert tag1.to_string() in tag_strings
        assert tag2.to_string() in tag_strings

    async def test_get_tags_for_nonexistent(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tags = await mgr.get_tags_for_content("missing", "chunks")
        assert tags == []

    async def test_remove_tag_keeps_content(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag1 = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        tag2 = IndexTag(directory="/a", branch="dev", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag1, tag2])
        should_delete = await mgr.remove_tag("h1", "chunks", tag1)
        assert should_delete is False
        # Content still accessible
        result = await mgr.get_cached_content("h1", "chunks")
        assert result == {"v": 1}

    async def test_remove_last_tag_deletes_content(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag])
        should_delete = await mgr.remove_tag("h1", "chunks", tag)
        assert should_delete is True
        result = await mgr.get_cached_content("h1", "chunks")
        assert result is None

    async def test_cleanup_orphaned_content(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag])
        # Remove all tags manually to create orphan
        conn = await mgr._get_connection()
        conn.execute("DELETE FROM cache_tags")
        conn.commit()
        removed = await mgr.cleanup_orphaned_content()
        assert removed == 1
        result = await mgr.get_cached_content("h1", "chunks")
        assert result is None

    async def test_cleanup_no_orphans(self, tmp_path: Path):
        mgr = GlobalCacheManager(tmp_path)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        await mgr.store_cached_content("h1", "chunks", {"v": 1}, [tag])
        removed = await mgr.cleanup_orphaned_content()
        assert removed == 0


# ---------------------------------------------------------------------------
# ContentAddressedIndexer
# ---------------------------------------------------------------------------
class TestContentAddressedIndexer:
    """Tests for ContentAddressedIndexer class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        indexer = ContentAddressedIndexer(cfg)
        assert indexer.config is cfg
        assert indexer.db_path.name == "tag_catalog.db"

    async def test_get_saved_items_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        indexer = ContentAddressedIndexer(cfg)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        items = await indexer.get_saved_items_for_tag(tag)
        assert items == []

    async def test_mark_complete_compute(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        indexer = ContentAddressedIndexer(cfg)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        items = [PathAndCacheKey(path="/a/b.py", cache_key="h1")]
        await indexer.mark_complete(items, "compute", tag)
        saved = await indexer.get_saved_items_for_tag(tag)
        assert len(saved) == 1
        assert saved[0]["path"] == "/a/b.py"
        assert saved[0]["content_hash"] == "h1"

    async def test_mark_complete_delete(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        indexer = ContentAddressedIndexer(cfg)
        tag = IndexTag(directory="/a", branch="main", artifact_id="chunks")
        items = [PathAndCacheKey(path="/a/b.py", cache_key="h1")]
        # First add
        await indexer.mark_complete(items, "compute", tag)
        assert len(await indexer.get_saved_items_for_tag(tag)) == 1
        # Then delete
        await indexer.mark_complete(items, "delete", tag)
        assert len(await indexer.get_saved_items_for_tag(tag)) == 0

    async def test_calculate_refresh_results_new_file(self, tmp_path: Path):
        f = tmp_path / "new.py"
        f.write_text("x = 1\n", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)])
        indexer = ContentAddressedIndexer(cfg)
        tag = IndexTag(directory=str(tmp_path), branch="main", artifact_id="chunks")
        current_files = {str(f): {"mtime": f.stat().st_mtime}}
        results = await indexer.calculate_refresh_results(
            tag, current_files, lambda p: f.read_text(encoding="utf-8")
        )
        assert isinstance(results, RefreshIndexResults)
        assert len(results.compute) == 1
        assert results.compute[0].path == str(f)
