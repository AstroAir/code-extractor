"""Tests for pysearch.core.integrations.hybrid_search module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.integrations.hybrid_search import HybridSearchManager
from pysearch.core.types import SearchItem, SearchResult, SearchStats


class TestHybridSearchManager:
    """Tests for HybridSearchManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = HybridSearchManager(cfg)
        assert mgr.config is cfg
        assert mgr.semantic_engine is None
        assert mgr._error_collector is None
        assert mgr._logger is None

    def test_set_dependencies(self):
        mgr = HybridSearchManager(SearchConfig())
        ec = MagicMock()
        logger = MagicMock()
        mgr.set_dependencies(ec, logger)
        assert mgr._error_collector is ec
        assert mgr._logger is logger

    def test_cluster_results_by_similarity_empty(self):
        mgr = HybridSearchManager(SearchConfig())
        result = mgr.cluster_results_by_similarity([])
        assert result == []

    def test_cluster_results_by_similarity_fallback(self):
        mgr = HybridSearchManager(SearchConfig())
        items = [
            SearchItem(file=Path("a.py"), start_line=1, end_line=1, lines=["x"]),
            SearchItem(file=Path("b.py"), start_line=1, end_line=1, lines=["y"]),
        ]
        result = mgr.cluster_results_by_similarity(items)
        # Should return items even if internal clustering fails
        assert len(result) >= 1

    def test_deduplicate_results_empty(self):
        mgr = HybridSearchManager(SearchConfig())
        result = mgr.deduplicate_results([])
        assert result == []

    def test_deduplicate_results_fallback(self):
        mgr = HybridSearchManager(SearchConfig())
        items = [
            SearchItem(file=Path("a.py"), start_line=1, end_line=3, lines=["a", "b", "c"]),
            SearchItem(file=Path("a.py"), start_line=2, end_line=4, lines=["b", "c", "d"]),
        ]
        result = mgr.deduplicate_results(items)
        assert isinstance(result, list)

    def test_rank_results_empty(self):
        mgr = HybridSearchManager(SearchConfig())
        result = mgr.rank_results([], "test")
        assert result == []

    def test_rank_results_fallback(self):
        mgr = HybridSearchManager(SearchConfig())
        items = [
            SearchItem(file=Path("a.py"), start_line=1, end_line=1, lines=["test line"]),
        ]
        result = mgr.rank_results(items, "test", strategy="hybrid")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_traditional_only(self):
        mgr = HybridSearchManager(SearchConfig())
        mock_items = [
            SearchItem(file=Path("a.py"), start_line=1, end_line=1, lines=["match"], match_spans=[(0, (0, 5))]),
        ]
        mock_result = SearchResult(
            items=mock_items,
            stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0),
        )

        def traditional_func(query):
            return mock_result

        result = await mgr.hybrid_search(
            pattern="match",
            traditional_search_func=traditional_func,
            use_graphrag=False,
            use_metadata_index=False,
        )
        assert result["traditional"] is not None
        assert "traditional" in result["metadata"]["methods_used"]
        assert result["graphrag"] is None
        assert result["metadata_index"] is None

    @pytest.mark.asyncio
    async def test_hybrid_search_traditional_exception(self):
        mgr = HybridSearchManager(SearchConfig())
        mgr._logger = MagicMock()

        def failing_func(query):
            raise RuntimeError("search failed")

        result = await mgr.hybrid_search(
            pattern="fail",
            traditional_search_func=failing_func,
            use_graphrag=False,
            use_metadata_index=False,
        )
        assert result["traditional"] is None

    @pytest.mark.asyncio
    async def test_hybrid_search_metadata(self):
        mgr = HybridSearchManager(SearchConfig())

        def noop_func(query):
            return SearchResult(items=[], stats=SearchStats())

        result = await mgr.hybrid_search(
            pattern="test",
            traditional_search_func=noop_func,
            use_graphrag=False,
            use_metadata_index=False,
        )
        assert "pattern" in result["metadata"]
        assert "timestamp" in result["metadata"]
        assert "methods_used" in result["metadata"]
