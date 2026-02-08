"""Tests for pysearch.indexing.indexes.chunk_index module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.analysis.content_addressing import (
    IndexingProgressUpdate,
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)
from pysearch.indexing.indexes.chunk_index import ChunkIndex


def _make_config(tmp_path: Path) -> MagicMock:
    """Create a mock config that resolves to a real tmp directory."""
    cfg = MagicMock()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.resolve_cache_dir.return_value = cache_dir
    cfg.chunk_size = 1000
    return cfg


def _make_tag() -> IndexTag:
    return IndexTag(directory="/repo", branch="main", artifact_id="enhanced_chunks")


def _make_refresh_results(
    compute: list[PathAndCacheKey] | None = None,
    delete: list[PathAndCacheKey] | None = None,
    add_tag: list[PathAndCacheKey] | None = None,
    remove_tag: list[PathAndCacheKey] | None = None,
) -> RefreshIndexResults:
    return RefreshIndexResults(
        compute=compute or [],
        delete=delete or [],
        add_tag=add_tag or [],
        remove_tag=remove_tag or [],
    )


class TestChunkIndexInit:
    """Tests for ChunkIndex initialization."""

    def test_artifact_id(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        assert idx.artifact_id == "enhanced_chunks"

    def test_relative_expected_time(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        assert idx.relative_expected_time == 1.2

    def test_init_attributes(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        idx = ChunkIndex(config=cfg)
        assert idx.config is cfg
        assert idx.db_path == cfg.resolve_cache_dir() / "chunks.db"
        assert idx._connection is None
        assert idx.chunking_engine is not None

    def test_init_chunking_config(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        cfg.chunk_size = 2000
        idx = ChunkIndex(config=cfg)
        assert idx.chunking_engine.config.max_chunk_size == 2000
        assert idx.chunking_engine.config.min_chunk_size == 50
        assert idx.chunking_engine.config.overlap_size == 100

    def test_init_default_chunk_size(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        del cfg.chunk_size  # triggers getattr default
        idx = ChunkIndex(config=cfg)
        assert idx.chunking_engine.config.max_chunk_size == 1000


class TestChunkIndexConnection:
    """Tests for ChunkIndex database connection and table creation."""

    @pytest.mark.asyncio
    async def test_get_connection_creates_db(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        assert conn is not None
        assert idx._connection is conn

    @pytest.mark.asyncio
    async def test_get_connection_returns_same(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        conn1 = await idx._get_connection()
        conn2 = await idx._get_connection()
        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_tables_created(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        # Verify code_chunks table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='code_chunks'"
        )
        assert cursor.fetchone() is not None
        # Verify chunk_tags table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='chunk_tags'"
        )
        assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_chunk%'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        assert "idx_chunks_path_hash" in index_names
        assert "idx_chunks_chunk_id" in index_names
        assert "idx_chunks_entity" in index_names
        assert "idx_chunk_tags_tag" in index_names


class TestChunkIndexUpdate:
    """Tests for ChunkIndex.update method."""

    @pytest.mark.asyncio
    async def test_update_compute_stores_chunks(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "test:1:10"
        mock_chunk.content = "def hello(): pass"
        mock_chunk.language.value = "python"
        mock_chunk.start_line = 1
        mock_chunk.end_line = 10
        mock_chunk.chunk_type = "function"
        mock_chunk.entity_name = "hello"
        mock_chunk.entity_type = MagicMock(value="function")
        mock_chunk.complexity_score = 0.5
        mock_chunk.quality_score = 0.8
        mock_chunk.dependencies = []
        mock_chunk.overlap_with = []
        mock_chunk.metadata = None

        item = PathAndCacheKey(path="/test/file.py", cache_key="abc123")
        results = _make_refresh_results(compute=[item])

        with patch.object(idx.chunking_engine, "chunk_file", new_callable=AsyncMock) as mock_cf:
            mock_cf.return_value = [mock_chunk]
            with patch(
                "pysearch.indexing.indexes.chunk_index.read_text_safely",
                return_value="def hello(): pass",
            ):
                updates = []
                async for update in idx.update(tag, results, mark_complete):
                    updates.append(update)

        mark_complete.assert_called_once_with([item], "compute")
        assert any(u.status == "done" for u in updates)

        # Verify data in database
        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM code_chunks")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_update_compute_skips_empty_content(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/empty.py", cache_key="empty")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.chunk_index.read_text_safely",
            return_value=None,
        ):
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_compute_handles_exception(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/bad.py", cache_key="bad")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.chunk_index.read_text_safely",
            side_effect=Exception("read error"),
        ):
            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_not_called()
        assert any(u.status == "done" for u in updates)

    @pytest.mark.asyncio
    async def test_update_add_tag(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        # Pre-insert a chunk
        import time

        conn.execute(
            """INSERT INTO code_chunks
            (chunk_id, path, content_hash, content, language, start_line, end_line,
             chunk_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("c1", "/test/f.py", "hash1", "code", "python", 1, 5, "function", time.time()),
        )
        conn.commit()

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="hash1")
        results = _make_refresh_results(add_tag=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "add_tag")
        cursor = conn.execute("SELECT COUNT(*) FROM chunk_tags")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_update_remove_tag(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        import time

        now = time.time()
        conn.execute(
            """INSERT INTO code_chunks
            (chunk_id, path, content_hash, content, language, start_line, end_line,
             chunk_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("c1", "/test/f.py", "hash1", "code", "python", 1, 5, "function", now),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
            (chunk_id, tag.to_string(), now),
        )
        conn.commit()

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="hash1")
        results = _make_refresh_results(remove_tag=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "remove_tag")
        cursor = conn.execute("SELECT COUNT(*) FROM chunk_tags")
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_delete(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        import time

        now = time.time()
        conn.execute(
            """INSERT INTO code_chunks
            (chunk_id, path, content_hash, content, language, start_line, end_line,
             chunk_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("c1", "/test/f.py", "hash1", "code", "python", 1, 5, "function", now),
        )
        chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute(
            "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
            (chunk_id, tag.to_string(), now),
        )
        conn.commit()

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="hash1")
        results = _make_refresh_results(delete=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "delete")
        cursor = conn.execute("SELECT COUNT(*) FROM code_chunks")
        assert cursor.fetchone()[0] == 0
        cursor = conn.execute("SELECT COUNT(*) FROM chunk_tags")
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_empty_results(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        results = _make_refresh_results()

        updates = []
        async for update in idx.update(tag, results, mark_complete):
            updates.append(update)

        mark_complete.assert_not_called()
        assert len(updates) == 1
        assert updates[0].status == "done"
        assert updates[0].progress == 1.0

    @pytest.mark.asyncio
    async def test_update_progress_updates(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        items = [
            PathAndCacheKey(path=f"/test/f{i}.py", cache_key=f"hash{i}") for i in range(3)
        ]
        results = _make_refresh_results(compute=items)

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "c"
        mock_chunk.content = "x"
        mock_chunk.language.value = "python"
        mock_chunk.start_line = 1
        mock_chunk.end_line = 1
        mock_chunk.chunk_type = "basic"
        mock_chunk.entity_name = None
        mock_chunk.entity_type = None
        mock_chunk.complexity_score = 0.0
        mock_chunk.quality_score = 0.0
        mock_chunk.dependencies = []
        mock_chunk.overlap_with = []
        mock_chunk.metadata = None

        with patch.object(idx.chunking_engine, "chunk_file", new_callable=AsyncMock) as mock_cf:
            mock_cf.return_value = [mock_chunk]
            with patch(
                "pysearch.indexing.indexes.chunk_index.read_text_safely",
                return_value="code",
            ):
                updates = []
                async for update in idx.update(tag, results, mark_complete):
                    updates.append(update)

        # Should have progress updates for each file + final done
        assert len(updates) >= 2
        assert updates[-1].status == "done"


class TestChunkIndexRetrieve:
    """Tests for ChunkIndex.retrieve method."""

    @pytest.mark.asyncio
    async def _seed_chunks(self, idx: ChunkIndex, tag: IndexTag, count: int = 3):
        """Helper to seed test chunks into the database."""
        import time

        conn = await idx._get_connection()
        tag_string = tag.to_string()
        now = time.time()

        for i in range(count):
            cursor = conn.execute(
                """INSERT INTO code_chunks
                (chunk_id, path, content_hash, content, language, start_line, end_line,
                 chunk_type, entity_name, entity_type, complexity_score, quality_score,
                 dependencies, overlap_with, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"chunk_{i}",
                    f"/test/file{i}.py",
                    f"hash_{i}",
                    f"def func_{i}(): pass  # content {i}",
                    "python",
                    i * 10 + 1,
                    i * 10 + 10,
                    "function" if i % 2 == 0 else "class",
                    f"func_{i}",
                    "function" if i % 2 == 0 else "class",
                    0.5 + i * 0.1,
                    0.6 + i * 0.1,
                    json.dumps([]),
                    json.dumps([]),
                    json.dumps({}),
                    now,
                ),
            )
            chunk_db_id = cursor.lastrowid
            conn.execute(
                "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
                (chunk_db_id, tag_string, now),
            )
        conn.commit()

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("func", tag)
        assert len(results) == 3
        assert all("chunk_id" in r for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_query_filter(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("content 0", tag)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "chunk_0"

    @pytest.mark.asyncio
    async def test_retrieve_empty_query(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("", tag)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag, count=5)

        results = await idx.retrieve("func", tag, limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_retrieve_filter_chunk_type(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("", tag, chunk_type="function")
        assert all(r["chunk_type"] == "function" for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_filter_language(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("", tag, language="python")
        assert len(results) == 3
        results = await idx.retrieve("", tag, language="javascript")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_filter_entity_type(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("", tag, entity_type="class")
        assert all(r["entity_type"] == "class" for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_filter_min_quality(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("", tag, min_quality=0.75)
        assert all(r["quality_score"] >= 0.75 for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_no_results(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        results = await idx.retrieve("nonexistent_xyzzy", tag)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_result_structure(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag, count=1)

        results = await idx.retrieve("", tag)
        assert len(results) == 1
        r = results[0]
        expected_keys = {
            "id", "chunk_id", "path", "content", "language",
            "start_line", "end_line", "chunk_type", "entity_name",
            "entity_type", "complexity_score", "quality_score",
            "dependencies", "overlap_with", "metadata",
        }
        assert set(r.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_retrieve_wrong_tag(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await self._seed_chunks(idx, tag)

        other_tag = IndexTag(directory="/other", branch="dev", artifact_id="other")
        results = await idx.retrieve("", other_tag)
        assert len(results) == 0


class TestChunkIndexGetChunksByFile:
    """Tests for ChunkIndex.get_chunks_by_file method."""

    @pytest.mark.asyncio
    async def test_get_chunks_by_file(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        import time

        now = time.time()
        tag_string = tag.to_string()
        for i in range(3):
            path = "/test/target.py" if i < 2 else "/test/other.py"
            cursor = conn.execute(
                """INSERT INTO code_chunks
                (chunk_id, path, content_hash, content, language, start_line, end_line,
                 chunk_type, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"c{i}", path, f"h{i}", f"content {i}", "python", i * 10 + 1, i * 10 + 10, "function", now),
            )
            conn.execute(
                "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
                (cursor.lastrowid, tag_string, now),
            )
        conn.commit()

        results = await idx.get_chunks_by_file("/test/target.py", tag)
        assert len(results) == 2
        assert all("chunk_id" in r for r in results)
        assert all("content_preview" in r for r in results)

    @pytest.mark.asyncio
    async def test_get_chunks_by_file_content_preview_truncation(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        import time

        long_content = "x" * 200
        cursor = conn.execute(
            """INSERT INTO code_chunks
            (chunk_id, path, content_hash, content, language, start_line, end_line,
             chunk_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("c1", "/test/f.py", "h1", long_content, "python", 1, 10, "function", time.time()),
        )
        conn.execute(
            "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
            (cursor.lastrowid, tag.to_string(), time.time()),
        )
        conn.commit()

        results = await idx.get_chunks_by_file("/test/f.py", tag)
        assert len(results) == 1
        assert results[0]["content_preview"].endswith("...")
        assert len(results[0]["content_preview"]) == 103  # 100 + "..."

    @pytest.mark.asyncio
    async def test_get_chunks_by_file_empty(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()  # ensure tables exist

        results = await idx.get_chunks_by_file("/nonexistent.py", tag)
        assert results == []


class TestChunkIndexGetStatistics:
    """Tests for ChunkIndex.get_statistics method."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()

        stats = await idx.get_statistics(tag)
        assert stats["total_chunks"] == 0
        assert stats["chunks_by_type"] == {}
        assert stats["chunks_by_language"] == {}
        assert stats["average_quality"] == 0.0
        assert stats["average_complexity"] == 0.0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, tmp_path: Path):
        idx = ChunkIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        conn = await idx._get_connection()

        import time

        now = time.time()
        tag_string = tag.to_string()

        for i, (ctype, lang) in enumerate([
            ("function", "python"),
            ("function", "python"),
            ("class", "javascript"),
        ]):
            cursor = conn.execute(
                """INSERT INTO code_chunks
                (chunk_id, path, content_hash, content, language, start_line, end_line,
                 chunk_type, complexity_score, quality_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"c{i}", f"/f{i}.py", f"h{i}", "code", lang, 1, 10, ctype, 0.5, 0.8, now),
            )
            conn.execute(
                "INSERT INTO chunk_tags (chunk_db_id, tag, created_at) VALUES (?, ?, ?)",
                (cursor.lastrowid, tag_string, now),
            )
        conn.commit()

        stats = await idx.get_statistics(tag)
        assert stats["total_chunks"] == 3
        assert stats["chunks_by_type"]["function"] == 2
        assert stats["chunks_by_type"]["class"] == 1
        assert stats["chunks_by_language"]["python"] == 2
        assert stats["chunks_by_language"]["javascript"] == 1
        assert stats["average_quality"] == pytest.approx(0.8)
        assert stats["average_complexity"] == pytest.approx(0.5)
