"""Tests for pysearch.indexing.advanced.integration module."""

from __future__ import annotations

from pathlib import Path

from pysearch.core.config import SearchConfig
from pysearch.core.types import Language
from pysearch.indexing.advanced.integration import (
    IndexingOrchestrator,
    IndexSearchEngine,
    IndexSearchResult,
    SearchResultEnhancer,
    ensure_indexed,
    index_search,
)


def _make_config(tmp_path: Path) -> SearchConfig:
    return SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")


# ---------------------------------------------------------------------------
# IndexSearchResult
# ---------------------------------------------------------------------------
class TestIndexSearchResult:
    """Tests for IndexSearchResult dataclass."""

    def test_creation(self):
        r = IndexSearchResult(
            path="test.py",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            score=0.95,
        )
        assert r.path == "test.py"
        assert r.language == Language.PYTHON
        assert r.score == 0.95

    def test_defaults(self):
        r = IndexSearchResult(
            path="x.py",
            content="",
            start_line=1,
            end_line=1,
            language=Language.UNKNOWN,
            score=0.0,
        )
        assert r.score == 0.0
        assert r.entity_name is None
        assert r.entity_type is None
        assert r.similarity_score is None
        assert r.complexity_score is None
        assert r.quality_score is None
        assert r.dependencies == []
        assert r.context is None
        assert r.index_type == "unknown"
        assert r.chunk_id is None

    def test_with_all_fields(self):
        r = IndexSearchResult(
            path="a.py",
            content="class Foo: pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            score=0.8,
            entity_name="Foo",
            entity_type="class",
            similarity_score=0.9,
            complexity_score=2.5,
            quality_score=0.85,
            dependencies=["os", "sys"],
            context="surrounding code",
            index_type="code_snippets",
            chunk_id="a.py:1:1",
        )
        assert r.entity_name == "Foo"
        assert r.entity_type == "class"
        assert r.similarity_score == 0.9
        assert r.dependencies == ["os", "sys"]
        assert r.index_type == "code_snippets"
        assert r.chunk_id == "a.py:1:1"


# ---------------------------------------------------------------------------
# SearchResultEnhancer
# ---------------------------------------------------------------------------
class TestSearchResultEnhancer:
    """Tests for SearchResultEnhancer class."""

    def test_init(self, tmp_path: Path):
        enhancer = SearchResultEnhancer(_make_config(tmp_path))
        assert enhancer.config is not None

    async def test_enhance_results_empty(self, tmp_path: Path):
        enhancer = SearchResultEnhancer(_make_config(tmp_path))
        results = await enhancer.enhance_results([], "query")
        assert results == []

    async def test_enhance_results_basic(self, tmp_path: Path):
        enhancer = SearchResultEnhancer(_make_config(tmp_path))
        raw = [
            {
                "path": "test.py",
                "content": "x = 1",
                "start_line": 1,
                "end_line": 1,
                "language": "python",
                "score": 0.9,
            }
        ]
        results = await enhancer.enhance_results(raw, "x", include_context=False)
        assert len(results) == 1
        assert isinstance(results[0], IndexSearchResult)
        assert results[0].path == "test.py"
        assert results[0].score == 0.9

    async def test_enhance_results_with_context(self, tmp_path: Path):
        # Create a real file for context extraction
        f = tmp_path / "real.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")
        enhancer = SearchResultEnhancer(_make_config(tmp_path))
        raw = [
            {
                "path": str(f),
                "content": "line3",
                "start_line": 3,
                "end_line": 3,
                "language": "python",
                "score": 0.5,
            }
        ]
        results = await enhancer.enhance_results(raw, "line3", include_context=True)
        assert len(results) == 1
        assert results[0].context is not None

    async def test_extract_context_file_not_found(self, tmp_path: Path):
        enhancer = SearchResultEnhancer(_make_config(tmp_path))
        result = IndexSearchResult(
            path="/nonexistent/file.py",
            content="x",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            score=0.5,
        )
        ctx = await enhancer._extract_context(result, "x")
        assert ctx is None


# ---------------------------------------------------------------------------
# IndexingOrchestrator
# ---------------------------------------------------------------------------
class TestIndexingOrchestrator:
    """Tests for IndexingOrchestrator class."""

    def test_init(self, tmp_path: Path):
        orch = IndexingOrchestrator(_make_config(tmp_path))
        assert orch.indexing_engine is None
        assert orch.performance_monitor is None
        assert orch.last_index_time is None

    async def test_initialize_metadata_disabled(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        cfg.enable_metadata_indexing = False
        orch = IndexingOrchestrator(cfg)
        await orch.initialize()
        assert orch.indexing_engine is None

    async def test_ensure_indexed_no_engine(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        cfg.enable_metadata_indexing = False
        orch = IndexingOrchestrator(cfg)
        await orch.initialize()
        result = await orch.ensure_indexed()
        assert result is False

    async def test_get_health_status_disabled(self, tmp_path: Path):
        orch = IndexingOrchestrator(_make_config(tmp_path))
        status = await orch.get_health_status()
        assert status == {"status": "disabled"}

    async def test_cleanup_no_monitor(self, tmp_path: Path):
        orch = IndexingOrchestrator(_make_config(tmp_path))
        await orch.cleanup()  # should not raise


# ---------------------------------------------------------------------------
# IndexSearchEngine
# ---------------------------------------------------------------------------
class TestIndexSearchEngine:
    """Tests for IndexSearchEngine class."""

    def test_init(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        assert engine._initialized is False
        assert engine.orchestrator is not None
        assert engine.result_enhancer is not None

    async def test_search_returns_list(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.search("test")
        assert isinstance(results, list)

    async def test_initialize_idempotent(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        await engine.initialize()
        assert engine._initialized is True
        # Second call should be no-op
        await engine.initialize()
        assert engine._initialized is True

    # -- _deduplicate_results --
    def test_deduplicate_results_removes_dupes(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = [
            {"path": "a.py", "start_line": 1, "content": "x"},
            {"path": "a.py", "start_line": 1, "content": "x"},
            {"path": "b.py", "start_line": 2, "content": "y"},
        ]
        deduped = engine._deduplicate_results(results)
        assert len(deduped) == 2

    def test_deduplicate_results_empty(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        assert engine._deduplicate_results([]) == []

    # -- _rank_results --
    def test_rank_results_sorts_by_score(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = [
            {"content": "no match", "score": 0.1},
            {"content": "has query inside", "score": 0.5, "similarity_score": 0.8},
            {"content": "also query", "score": 0.9, "quality_score": 0.9},
        ]
        ranked = engine._rank_results(results, "query")
        assert all("final_score" in r for r in ranked)
        # Verify sorted descending
        scores = [r["final_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_results_exact_match_bonus(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = [
            {"content": "abc", "score": 0.5},
            {"content": "abc contains query word", "score": 0.5},
        ]
        ranked = engine._rank_results(results, "query")
        # The one with exact match should score higher
        assert ranked[0]["content"] == "abc contains query word"

    def test_rank_results_empty(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        assert engine._rank_results([], "q") == []

    # -- get_statistics --
    async def test_get_statistics(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        stats = await engine.get_statistics()
        assert isinstance(stats, dict)
        assert "initialized" in stats
        assert "health" in stats

    # -- search_entities (no engine) --
    async def test_search_entities_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.search_entities("foo")
        assert results == []

    # -- get_entities_by_file (no engine) --
    async def test_get_entities_by_file_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.get_entities_by_file("test.py")
        assert results == []

    # -- get_entity_by_id (no engine) --
    async def test_get_entity_by_id_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        result = await engine.get_entity_by_id(1)
        assert result is None

    # -- get_chunks_by_file (no engine) --
    async def test_get_chunks_by_file_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.get_chunks_by_file("test.py")
        assert results == []

    # -- search_in_file (no engine) --
    async def test_search_in_file_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.search_in_file("test.py", "query")
        assert results == []

    # -- get_similar_chunks (no engine) --
    async def test_get_similar_chunks_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        results = await engine.get_similar_chunks("content")
        assert results == []

    # -- optimize_indexes (no engine) --
    async def test_optimize_indexes_no_engine(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        await engine.optimize_indexes()  # should not raise

    # -- cleanup --
    async def test_cleanup(self, tmp_path: Path):
        engine = IndexSearchEngine(_make_config(tmp_path))
        await engine.cleanup()  # should not raise


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    async def test_index_search_returns_list(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        results = await index_search("test", cfg, limit=5)
        assert isinstance(results, list)

    async def test_ensure_indexed_returns_bool(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        result = await ensure_indexed(cfg, force=False)
        assert isinstance(result, bool)
