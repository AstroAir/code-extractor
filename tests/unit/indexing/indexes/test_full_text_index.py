"""Tests for pysearch.indexing.indexes.full_text_index module."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysearch.analysis.content_addressing import (
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)
from pysearch.indexing.indexes.full_text_index import FullTextIndex


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.resolve_cache_dir.return_value = cache_dir
    return cfg


def _make_tag() -> IndexTag:
    return IndexTag(directory="/repo", branch="main", artifact_id="enhanced_full_text")


def _make_refresh_results(
    compute=None, delete=None, add_tag=None, remove_tag=None,
) -> RefreshIndexResults:
    return RefreshIndexResults(
        compute=compute or [],
        delete=delete or [],
        add_tag=add_tag or [],
        remove_tag=remove_tag or [],
    )


async def _seed_fts(idx: FullTextIndex, tag: IndexTag, count: int = 3):
    """Seed test data into the full-text index database."""
    conn = await idx._get_connection()
    tag_string = tag.to_string()
    now = time.time()

    for i in range(count):
        content = f"def function_{i}():\n    '''Docstring {i}'''\n    return {i}\n"
        lang = "python"
        ftype = ".py"
        path = f"/test/file{i}.py"

        conn.execute(
            "INSERT INTO fts_content (path, content, language, file_type) VALUES (?, ?, ?, ?)",
            (path, content, lang, ftype),
        )
        cursor = conn.execute(
            """INSERT INTO fts_metadata (path, content_hash, language, file_size, line_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (path, f"hash_{i}", lang, len(content), content.count("\n") + 1, now),
        )
        metadata_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO fts_tags (metadata_id, tag, created_at) VALUES (?, ?, ?)",
            (metadata_id, tag_string, now),
        )
    conn.commit()


class TestFullTextIndexInit:
    """Tests for FullTextIndex initialization."""

    def test_artifact_id(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        assert idx.artifact_id == "enhanced_full_text"

    def test_relative_expected_time(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        assert idx.relative_expected_time == 1.0

    def test_init_attributes(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        idx = FullTextIndex(config=cfg)
        assert idx.config is cfg
        assert idx.db_path == cfg.resolve_cache_dir() / "full_text.db"
        assert idx._connection is None


class TestFullTextIndexConnection:
    """Tests for database connection and table creation."""

    @pytest.mark.asyncio
    async def test_get_connection_creates_db(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        assert conn is not None
        assert idx._connection is conn

    @pytest.mark.asyncio
    async def test_get_connection_idempotent(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        conn1 = await idx._get_connection()
        conn2 = await idx._get_connection()
        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_tables_created(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()

        # Check fts_metadata regular table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_metadata'"
        )
        assert cursor.fetchone() is not None

        # Check fts_tags regular table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_tags'"
        )
        assert cursor.fetchone() is not None

        # Check fts_content virtual table
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='fts_content'"
        )
        assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_fts%'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        assert "idx_fts_metadata_path_hash" in index_names
        assert "idx_fts_tags_tag" in index_names


class TestFullTextIndexBuildFtsQuery:
    """Tests for FullTextIndex._build_fts_query method."""

    def _make_idx(self, tmp_path: Path) -> FullTextIndex:
        return FullTextIndex(config=_make_config(tmp_path))

    def test_single_term(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        result = idx._build_fts_query("hello")
        assert result == "hello"

    def test_multiple_terms(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        result = idx._build_fts_query("hello world")
        assert result == "hello AND world"

    def test_quotes_escaped(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        result = idx._build_fts_query('say "hello"')
        assert '""hello""' in result

    def test_empty_query(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        result = idx._build_fts_query("")
        assert result == ""


class TestFullTextIndexUpdate:
    """Tests for FullTextIndex.update method."""

    @pytest.mark.asyncio
    async def test_update_compute(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        item = PathAndCacheKey(path="/test/file.py", cache_key="abc123")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.full_text_index.read_text_safely",
            return_value="def hello():\n    pass\n",
        ), patch(
            "pysearch.indexing.indexes.full_text_index.detect_language",
        ) as mock_detect:
            mock_detect.return_value = MagicMock(value="python")

            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_called_once_with([item], "compute")
        assert any(u.status == "done" for u in updates)

        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM fts_metadata")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_update_compute_skips_empty(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/empty.py", cache_key="empty")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.full_text_index.read_text_safely",
            return_value=None,
        ):
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_compute_handles_exception(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/bad.py", cache_key="bad")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.full_text_index.read_text_safely",
            side_effect=Exception("read error"),
        ):
            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_not_called()
        assert any(u.status == "done" for u in updates)

    @pytest.mark.asyncio
    async def test_update_add_tag(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        new_tag = IndexTag(directory="/repo", branch="dev", artifact_id="enhanced_full_text")
        results = _make_refresh_results(add_tag=[item])

        async for _ in idx.update(new_tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "add_tag")

    @pytest.mark.asyncio
    async def test_update_remove_tag(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        results = _make_refresh_results(remove_tag=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "remove_tag")
        conn = await idx._get_connection()
        cursor = conn.execute(
            "SELECT COUNT(*) FROM fts_tags WHERE tag = ?", (tag.to_string(),)
        )
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_delete(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        results = _make_refresh_results(delete=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "delete")
        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM fts_metadata")
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_empty_results(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        results = _make_refresh_results()

        updates = []
        async for update in idx.update(tag, results, mark_complete):
            updates.append(update)

        mark_complete.assert_not_called()
        assert updates[-1].status == "done"
        assert updates[-1].progress == 1.0


class TestFullTextIndexRetrieve:
    """Tests for FullTextIndex.retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag)

        results = await idx.retrieve("function", tag)
        assert len(results) > 0
        assert all("path" in r for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_specific_term(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag)

        results = await idx.retrieve("function_0", tag)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=5)

        results = await idx.retrieve("function", tag, limit=2)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_retrieve_result_structure(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        results = await idx.retrieve("function", tag)
        if results:
            r = results[0]
            expected_keys = {
                "path", "content", "language", "file_type",
                "file_size", "line_count", "rank",
            }
            assert set(r.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag)

        results = await idx.retrieve("zzz_xyzzy_nonexistent_term_abc", tag)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_wrong_tag(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag)

        other_tag = IndexTag(directory="/other", branch="dev", artifact_id="other")
        results = await idx.retrieve("function", other_tag)
        assert len(results) == 0


class TestFullTextIndexSearchInFile:
    """Tests for FullTextIndex.search_in_file method."""

    @pytest.mark.asyncio
    async def test_search_in_file_basic(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        results = await idx.search_in_file("/test/file0.py", "function_0", tag)
        assert len(results) >= 1
        assert results[0]["line_number"] >= 1
        assert "line_content" in results[0]
        assert "context" in results[0]

    @pytest.mark.asyncio
    async def test_search_in_file_context_lines(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        results = await idx.search_in_file("/test/file0.py", "function_0", tag, context_lines=1)
        if results:
            assert "start_line" in results[0]
            assert "end_line" in results[0]

    @pytest.mark.asyncio
    async def test_search_in_file_no_match(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        results = await idx.search_in_file("/test/file0.py", "zzz_nonexistent", tag)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_in_file_not_found(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()

        results = await idx.search_in_file("/nonexistent.py", "anything", tag)
        assert results == []

    @pytest.mark.asyncio
    async def test_search_in_file_case_insensitive(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=1)

        results = await idx.search_in_file("/test/file0.py", "FUNCTION_0", tag)
        assert len(results) >= 1


class TestFullTextIndexGetStatistics:
    """Tests for FullTextIndex.get_statistics method."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()

        stats = await idx.get_statistics(tag)
        assert stats["total_files"] == 0
        assert stats["files_by_language"] == {}
        assert stats["total_size_bytes"] == 0
        assert stats["total_lines"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, tmp_path: Path):
        idx = FullTextIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_fts(idx, tag, count=3)

        stats = await idx.get_statistics(tag)
        assert stats["total_files"] == 3
        assert "python" in stats["files_by_language"]
        assert stats["files_by_language"]["python"] == 3
        assert stats["total_size_bytes"] > 0
        assert stats["total_lines"] > 0
