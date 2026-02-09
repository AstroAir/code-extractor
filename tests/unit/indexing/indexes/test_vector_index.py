"""Tests for pysearch.indexing.indexes.vector_index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.analysis.content_addressing import (
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)


def _lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401

        return True
    except ImportError:
        return False


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.resolve_cache_dir.return_value = cache_dir
    cfg.chunk_size = 1000
    cfg.embedding_provider = "huggingface"
    cfg.embedding_model = "all-MiniLM-L6-v2"
    cfg.embedding_batch_size = 100
    cfg.openai_api_key = None
    cfg.vector_db_provider = "lancedb"
    return cfg


def _make_tag() -> IndexTag:
    return IndexTag(directory="/repo", branch="main", artifact_id="enhanced_vectors")


def _make_refresh_results(
    compute=None,
    delete=None,
    add_tag=None,
    remove_tag=None,
) -> RefreshIndexResults:
    return RefreshIndexResults(
        compute=compute or [],
        delete=delete or [],
        add_tag=add_tag or [],
        remove_tag=remove_tag or [],
    )


_skip_no_lancedb = pytest.mark.skipif(not _lancedb_available(), reason="LanceDB not installed")


@_skip_no_lancedb
class TestVectorIndexInit:
    """Tests for VectorIndex initialization."""

    def test_artifact_id(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        assert idx.artifact_id == "enhanced_vectors"

    def test_relative_expected_time(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        assert idx.relative_expected_time == 3.0

    def test_init_attributes(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        cfg = _make_config(tmp_path)
        idx = VectorIndex(config=cfg)
        assert idx.config is cfg
        assert idx.chunking_engine is not None
        assert idx.vector_manager is not None

    def test_init_chunking_config(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        cfg = _make_config(tmp_path)
        cfg.chunk_size = 2000
        idx = VectorIndex(config=cfg)
        assert idx.chunking_engine.config.max_chunk_size == 2000

    def test_init_default_chunk_size(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        cfg = _make_config(tmp_path)
        del cfg.chunk_size
        idx = VectorIndex(config=cfg)
        assert idx.chunking_engine.config.max_chunk_size == 1000


@_skip_no_lancedb
class TestVectorIndexUpdate:
    """Tests for VectorIndex.update method."""

    @pytest.mark.asyncio
    async def test_update_compute_new_collection(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        mock_chunk = MagicMock()
        mock_chunk.chunk_id = "c1"
        mock_chunk.content = "def hello(): pass"

        item = PathAndCacheKey(path="/test/file.py", cache_key="abc123")
        results = _make_refresh_results(compute=[item])

        with (
            patch.object(idx.chunking_engine, "chunk_file", new_callable=AsyncMock) as mock_cf,
            patch(
                "pysearch.indexing.indexes.vector_index.read_text_safely",
                return_value="def hello(): pass",
            ),
            patch.object(
                idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
            ) as mock_exists,
            patch.object(idx.vector_manager, "index_chunks", new_callable=AsyncMock) as mock_index,
        ):
            mock_cf.return_value = [mock_chunk]
            mock_exists.return_value = False

            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_called_once_with([item], "compute")
        mock_index.assert_called_once()
        assert any(u.status == "done" for u in updates)

    @pytest.mark.asyncio
    async def test_update_compute_existing_collection(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        mock_chunk = MagicMock()
        item = PathAndCacheKey(path="/test/file.py", cache_key="abc")
        results = _make_refresh_results(compute=[item])

        with (
            patch.object(idx.chunking_engine, "chunk_file", new_callable=AsyncMock) as mock_cf,
            patch(
                "pysearch.indexing.indexes.vector_index.read_text_safely",
                return_value="code",
            ),
            patch.object(
                idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
            ) as mock_exists,
            patch.object(
                idx.vector_manager, "update_chunks", new_callable=AsyncMock
            ) as mock_update,
        ):
            mock_cf.return_value = [mock_chunk]
            mock_exists.return_value = True

            async for _ in idx.update(tag, results, mark_complete):
                pass

        mock_update.assert_called_once()
        mark_complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_compute_skips_empty(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/empty.py", cache_key="empty")
        results = _make_refresh_results(compute=[item])

        with (
            patch(
                "pysearch.indexing.indexes.vector_index.read_text_safely",
                return_value=None,
            ),
            patch.object(
                idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
            ) as mock_exists,
        ):
            mock_exists.return_value = False
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_add_tag(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="h1")
        results = _make_refresh_results(add_tag=[item])

        with patch.object(
            idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = False
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_called_once_with([item], "add_tag")

    @pytest.mark.asyncio
    async def test_update_remove_tag(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="h1")
        results = _make_refresh_results(remove_tag=[item])

        with (
            patch.object(
                idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
            ) as mock_exists,
            patch.object(idx.vector_manager, "delete_chunks", new_callable=AsyncMock),
        ):
            mock_exists.return_value = False
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_called_once_with([item], "remove_tag")

    @pytest.mark.asyncio
    async def test_update_delete(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/f.py", cache_key="h1")
        results = _make_refresh_results(delete=[item])

        with (
            patch.object(
                idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
            ) as mock_exists,
            patch.object(idx.vector_manager, "delete_chunks", new_callable=AsyncMock),
            patch.object(
                idx.vector_manager, "cleanup_orphaned_vectors", new_callable=AsyncMock
            ) as mock_cleanup,
        ):
            mock_exists.return_value = False
            mock_cleanup.return_value = 0

            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_called_once_with([item], "delete")
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_empty_results(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        results = _make_refresh_results()

        with patch.object(
            idx.vector_manager.vector_db, "collection_exists", new_callable=AsyncMock
        ) as mock_exists:
            mock_exists.return_value = False
            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_not_called()
        assert updates[-1].status == "done"


@_skip_no_lancedb
class TestVectorIndexRetrieve:
    """Tests for VectorIndex.retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        mock_result = MagicMock()
        mock_result.chunk_id = "c1"
        mock_result.content = "def hello(): pass"
        mock_result.file_path = "/test/f.py"
        mock_result.start_line = 1
        mock_result.end_line = 2
        mock_result.similarity_score = 0.95
        mock_result.metadata = {
            "language": "python",
            "chunk_type": "function",
            "entity_name": "hello",
            "complexity_score": 0.5,
            "quality_score": 0.8,
        }

        with patch.object(idx.vector_manager, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_result]
            results = await idx.retrieve("hello", tag)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        assert results[0]["similarity_score"] == 0.95
        assert results[0]["language"] == "python"

    @pytest.mark.asyncio
    async def test_retrieve_with_filters(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(idx.vector_manager, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = []
            await idx.retrieve(
                "query",
                tag,
                language="python",
                file_path="/test/f.py",
                chunk_type="function",
                similarity_threshold=0.5,
            )

        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["filters"]["language"] == "python"
        assert call_kwargs[1]["filters"]["file_path"] == "/test/f.py"
        assert call_kwargs[1]["filters"]["chunk_type"] == "function"
        assert call_kwargs[1]["similarity_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_retrieve_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(idx.vector_manager, "search", new_callable=AsyncMock) as mock_search:
            mock_search.side_effect = Exception("search error")
            results = await idx.retrieve("query", tag)

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_result_structure(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        mock_result = MagicMock()
        mock_result.chunk_id = "c1"
        mock_result.content = "code"
        mock_result.file_path = "/f.py"
        mock_result.start_line = 1
        mock_result.end_line = 5
        mock_result.similarity_score = 0.9
        mock_result.metadata = {}

        with patch.object(idx.vector_manager, "search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = [mock_result]
            results = await idx.retrieve("q", tag)

        expected_keys = {
            "chunk_id",
            "content",
            "file_path",
            "start_line",
            "end_line",
            "similarity_score",
            "language",
            "chunk_type",
            "entity_name",
            "complexity_score",
            "quality_score",
        }
        assert set(results[0].keys()) == expected_keys


@_skip_no_lancedb
class TestVectorIndexGetSimilarChunks:
    """Tests for VectorIndex.get_similar_chunks method."""

    @pytest.mark.asyncio
    async def test_get_similar_chunks(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        mock_result = MagicMock()
        mock_result.chunk_id = "c2"
        mock_result.content = "similar code"
        mock_result.file_path = "/f.py"
        mock_result.similarity_score = 0.85

        with (
            patch.object(
                idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
            ) as mock_embed,
            patch.object(
                idx.vector_manager.vector_db, "search_vectors", new_callable=AsyncMock
            ) as mock_sv,
        ):
            mock_embed.return_value = [0.1, 0.2, 0.3]
            mock_sv.return_value = [mock_result]

            results = await idx.get_similar_chunks("def hello(): pass", tag, limit=5)

        assert len(results) == 1
        assert results[0]["chunk_id"] == "c2"

    @pytest.mark.asyncio
    async def test_get_similar_chunks_excludes_self(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        mock_self = MagicMock()
        mock_self.chunk_id = "self_chunk"
        mock_self.content = "self"
        mock_self.file_path = "/f.py"
        mock_self.similarity_score = 1.0

        mock_other = MagicMock()
        mock_other.chunk_id = "other_chunk"
        mock_other.content = "other"
        mock_other.file_path = "/g.py"
        mock_other.similarity_score = 0.8

        with (
            patch.object(
                idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
            ) as mock_embed,
            patch.object(
                idx.vector_manager.vector_db, "search_vectors", new_callable=AsyncMock
            ) as mock_sv,
        ):
            mock_embed.return_value = [0.1]
            mock_sv.return_value = [mock_self, mock_other]

            results = await idx.get_similar_chunks("code", tag, exclude_chunk_id="self_chunk")

        assert len(results) == 1
        assert results[0]["chunk_id"] == "other_chunk"

    @pytest.mark.asyncio
    async def test_get_similar_chunks_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("embed error")
            results = await idx.get_similar_chunks("code", tag)

        assert results == []


@_skip_no_lancedb
class TestVectorIndexGetStatistics:
    """Tests for VectorIndex.get_statistics method."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "get_collection_stats", new_callable=AsyncMock
        ) as mock_stats:
            mock_stats.return_value = {
                "total_vectors": 100,
                "provider": "lancedb",
                "dimensions": 384,
            }
            stats = await idx.get_statistics(tag)

        assert stats["total_vectors"] == 100
        assert stats["provider"] == "lancedb"
        assert stats["dimensions"] == 384
        assert "embedding_provider" in stats
        assert "embedding_model" in stats

    @pytest.mark.asyncio
    async def test_get_statistics_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "get_collection_stats", new_callable=AsyncMock
        ) as mock_stats:
            mock_stats.side_effect = Exception("stats error")
            stats = await idx.get_statistics(tag)

        assert stats == {}


@_skip_no_lancedb
class TestVectorIndexOptimizeIndex:
    """Tests for VectorIndex.optimize_index method."""

    @pytest.mark.asyncio
    async def test_optimize_index(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "optimize_collection", new_callable=AsyncMock
        ) as mock_opt:
            await idx.optimize_index(tag)

        mock_opt.assert_called_once()

    @pytest.mark.asyncio
    async def test_optimize_index_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "optimize_collection", new_callable=AsyncMock
        ) as mock_opt:
            mock_opt.side_effect = Exception("optimize error")
            await idx.optimize_index(tag)  # should not raise


@_skip_no_lancedb
class TestVectorIndexRerankResults:
    """Tests for VectorIndex.rerank_results method."""

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))

        with patch.object(
            idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
        ):
            results = await idx.rerank_results([], "query")

        assert results == []

    @pytest.mark.asyncio
    async def test_rerank_applies_boosts(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))

        results = [
            {
                "similarity_score": 0.5,
                "quality_score": 0.9,
                "complexity_score": 0.5,
                "content": "def query(): pass",
                "entity_name": "query",
            },
            {
                "similarity_score": 0.8,
                "quality_score": 0.1,
                "complexity_score": 0.9,
                "content": "other code",
                "entity_name": "other",
            },
        ]

        with patch.object(
            idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("no embeddings")
            reranked = await idx.rerank_results(results, "query")

        assert all("enhanced_score" in r for r in reranked)
        # First result should have exact_match + entity_name_match boosts
        assert reranked[0]["enhanced_score"] > 0

    @pytest.mark.asyncio
    async def test_rerank_custom_boost_factors(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))

        results = [
            {
                "similarity_score": 0.5,
                "quality_score": 1.0,
                "complexity_score": 0.5,
                "content": "x",
                "entity_name": "",
            },
        ]
        boost = {
            "quality_score": 1.0,
            "complexity_score": 0.0,
            "exact_match": 0.0,
            "entity_name_match": 0.0,
            "cross_similarity": 0.0,
        }

        with patch.object(
            idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("no embeddings")
            reranked = await idx.rerank_results(results, "query", boost_factors=boost)

        # quality_score boost: 1.0 * 1.0 = 1.0 added to base 0.5
        assert reranked[0]["enhanced_score"] == pytest.approx(1.5)

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_enhanced_score(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))

        results = [
            {
                "similarity_score": 0.3,
                "quality_score": 0.0,
                "complexity_score": 0.0,
                "content": "",
                "entity_name": "",
            },
            {
                "similarity_score": 0.9,
                "quality_score": 0.0,
                "complexity_score": 0.0,
                "content": "",
                "entity_name": "",
            },
        ]

        with patch.object(
            idx.vector_manager.embedding_provider, "embed_query", new_callable=AsyncMock
        ) as mock_embed:
            mock_embed.side_effect = Exception("no embeddings")
            reranked = await idx.rerank_results(results, "query")

        assert reranked[0]["enhanced_score"] >= reranked[1]["enhanced_score"]


@_skip_no_lancedb
class TestVectorIndexBatchRetrieve:
    """Tests for VectorIndex.batch_retrieve method."""

    @pytest.mark.asyncio
    async def test_batch_retrieve(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        mock_result = MagicMock()
        mock_result.chunk_id = "c1"
        mock_result.content = "code"
        mock_result.file_path = "/f.py"
        mock_result.start_line = 1
        mock_result.end_line = 5
        mock_result.similarity_score = 0.9
        mock_result.metadata = {"language": "python", "chunk_type": "function", "entity_name": "fn"}

        with patch.object(idx.vector_manager, "batch_search", new_callable=AsyncMock) as mock_bs:
            mock_bs.return_value = [[mock_result], []]
            results = await idx.batch_retrieve(["query1", "query2"], tag)

        assert "query1" in results
        assert "query2" in results
        assert len(results["query1"]) == 1
        assert len(results["query2"]) == 0

    @pytest.mark.asyncio
    async def test_batch_retrieve_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(idx.vector_manager, "batch_search", new_callable=AsyncMock) as mock_bs:
            mock_bs.side_effect = Exception("batch error")
            results = await idx.batch_retrieve(["q1", "q2"], tag)

        assert results == {"q1": [], "q2": []}


@_skip_no_lancedb
class TestVectorIndexCleanupIndex:
    """Tests for VectorIndex.cleanup_index method."""

    @pytest.mark.asyncio
    async def test_cleanup_index(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "cleanup_orphaned_vectors", new_callable=AsyncMock
        ) as mock_cleanup:
            mock_cleanup.return_value = 5
            removed = await idx.cleanup_index(tag, valid_file_paths={"/a.py"})

        assert removed == 5
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_index_handles_exception(self, tmp_path: Path):
        from pysearch.indexing.indexes.vector_index import VectorIndex

        idx = VectorIndex(config=_make_config(tmp_path))
        tag = _make_tag()

        with patch.object(
            idx.vector_manager, "cleanup_orphaned_vectors", new_callable=AsyncMock
        ) as mock_cleanup:
            mock_cleanup.side_effect = Exception("cleanup error")
            removed = await idx.cleanup_index(tag)

        assert removed == 0
