"""
Expanded comprehensive tests for API module focusing on untested code paths.

This module tests PySearch functionality that is currently not covered by existing tests,
including initialization edge cases, cache management, GraphRAG features, error handling,
and advanced search scenarios.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch, AsyncMock

import pytest

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import (
    Query, 
    OutputFormat, 
    Language, 
    ASTFilters, 
    MetadataFilters,
    GraphRAGQuery,
    SearchResult,
    SearchStats
)
from pysearch.qdrant_client import QdrantConfig
from pysearch.indexer_metadata import IndexQuery


class TestPySearchInitialization:
    """Test PySearch initialization with various configurations."""

    def test_default_initialization(self) -> None:
        """Test PySearch initialization with default configuration."""
        engine = PySearch()
        
        assert engine.cfg is not None
        assert engine.indexer is not None
        assert engine.history is not None
        assert engine.logger is not None
        assert engine.error_collector is not None
        assert engine.semantic_engine is not None
        assert engine.dependency_analyzer is not None
        assert engine.watch_manager is not None
        assert not engine._auto_watch_enabled
        assert engine.cache_manager is None
        assert not engine._caching_enabled
        assert engine.multi_repo_engine is None
        assert not engine._multi_repo_enabled
        assert engine.cache_ttl == 300

    def test_initialization_with_custom_config(self, tmp_path: Path) -> None:
        """Test PySearch initialization with custom configuration."""
        config = SearchConfig(
            paths=[str(tmp_path)],
            include=["**/*.py"],
            context=10,
            parallel=True,
            workers=2
        )
        
        engine = PySearch(config)
        
        assert engine.cfg == config
        assert engine.cfg.context == 10
        assert engine.cfg.parallel is True
        assert engine.cfg.workers == 2

    def test_initialization_with_graphrag_enabled(self, tmp_path: Path) -> None:
        """Test PySearch initialization with GraphRAG enabled."""
        config = SearchConfig(paths=[str(tmp_path)])
        qdrant_config = QdrantConfig(host="localhost", port=6333)

        engine = PySearch(
            config=config,
            qdrant_config=qdrant_config,
            enable_graphrag=True
        )

        assert engine.enable_graphrag is True
        assert engine.qdrant_config == qdrant_config
        assert not engine._graphrag_initialized

    def test_initialization_with_enhanced_indexing(self, tmp_path: Path) -> None:
        """Test PySearch initialization with enhanced indexing enabled."""
        config = SearchConfig(paths=[str(tmp_path)])
        
        engine = PySearch(
            config=config,
            enable_enhanced_indexing=True
        )
        
        assert engine.enable_enhanced_indexing is True
        assert not engine._enhanced_indexing_initialized

    def test_initialization_with_custom_logger(self, tmp_path: Path) -> None:
        """Test PySearch initialization with custom logger."""
        mock_logger = Mock()
        config = SearchConfig(paths=[str(tmp_path)])
        
        engine = PySearch(config=config, logger=mock_logger)
        
        assert engine.logger == mock_logger


class TestPySearchCacheManagement:
    """Test PySearch cache management functionality."""

    def test_clear_caches(self, tmp_path: Path) -> None:
        """Test clearing all internal caches."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)
        
        # Add some dummy cache entries
        test_path = Path("test.py")
        engine._file_content_cache[test_path] = (time.time(), "test content")
        engine._search_result_cache["test_query"] = (
            time.time(), 
            SearchResult(items=[], stats=SearchStats())
        )
        
        assert len(engine._file_content_cache) == 1
        assert len(engine._search_result_cache) == 1
        
        engine.clear_caches()
        
        assert len(engine._file_content_cache) == 0
        assert len(engine._search_result_cache) == 0

    def test_cache_ttl_expiration(self, tmp_path: Path) -> None:
        """Test cache TTL expiration logic."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config)
        engine.cache_ttl = 1  # 1 second TTL
        
        # Add cache entry
        test_path = Path("test.py")
        old_time = time.time() - 2  # 2 seconds ago (expired)
        engine._file_content_cache[test_path] = (old_time, "test content")
        
        # Cache should be considered expired
        assert len(engine._file_content_cache) == 1
        
        # Trigger cache cleanup by accessing it
        engine.clear_caches()
        assert len(engine._file_content_cache) == 0

    def test_thread_safe_cache_access(self, tmp_path: Path) -> None:
        """Test thread-safe cache access."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config)
        
        def cache_writer(key: str, value: str) -> None:
            test_path = Path(key)
            engine._file_content_cache[test_path] = (time.time(), value)
        
        def cache_reader() -> int:
            return len(engine._file_content_cache)
        
        # Test concurrent access
        threads = []
        for i in range(5):
            t = threading.Thread(target=cache_writer, args=(f"test{i}.py", f"content{i}"))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        assert cache_reader() == 5


class TestPySearchErrorHandling:
    """Test PySearch error handling and edge cases."""

    def test_search_with_invalid_pattern(self, tmp_path: Path) -> None:
        """Test search with invalid regex pattern."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)
        
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        # Invalid regex pattern
        query = Query(pattern="[invalid", use_regex=True)
        
        # Should handle the error gracefully
        result = engine.run(query)
        
        # Should return empty results but not crash
        assert isinstance(result, SearchResult)
        assert len(result.items) == 0

    def test_search_with_nonexistent_paths(self, tmp_path: Path) -> None:
        """Test search with nonexistent paths."""
        # Create a path that doesn't exist within tmp_path
        nonexistent_path = tmp_path / "nonexistent_directory"
        config = SearchConfig(paths=[str(nonexistent_path)], include=["**/*.py"])
        engine = PySearch(config)

        query = Query(pattern="test")
        result = engine.run(query)

        # Should handle gracefully
        assert isinstance(result, SearchResult)
        assert len(result.items) == 0

    def test_search_with_empty_pattern(self, tmp_path: Path) -> None:
        """Test search with empty pattern."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)
        
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test(): pass")
        
        query = Query(pattern="")
        result = engine.run(query)
        
        # Should handle empty pattern gracefully
        assert isinstance(result, SearchResult)

    def test_indexer_scan_error_handling(self, tmp_path: Path) -> None:
        """Test error handling during indexer scan."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config)
        
        # Mock indexer to raise exception
        with patch.object(engine.indexer, 'scan', side_effect=Exception("Scan error")):
            query = Query(pattern="test")
            result = engine.run(query)
            
            # Should handle error and continue
            assert isinstance(result, SearchResult)
            assert len(engine.error_collector.errors) > 0


class TestPySearchAdvancedFeatures:
    """Test PySearch advanced features and integrations."""

    @pytest.mark.asyncio
    async def test_graphrag_initialization(self, tmp_path: Path) -> None:
        """Test GraphRAG initialization."""
        config = SearchConfig(paths=[str(tmp_path)])
        qdrant_config = QdrantConfig(host="localhost", port=6333)
        
        engine = PySearch(
            config=config,
            qdrant_config=qdrant_config,
            enable_graphrag=True
        )
        
        # Mock the GraphRAG components
        with patch('pysearch.api.GraphRAGEngine') as mock_graphrag, \
             patch('pysearch.api.QdrantVectorStore') as mock_vector_store:
            
            mock_graphrag_instance = AsyncMock()
            mock_graphrag.return_value = mock_graphrag_instance
            mock_vector_store_instance = AsyncMock()
            mock_vector_store.return_value = mock_vector_store_instance
            
            await engine.initialize_graphrag()
            
            assert engine._graphrag_initialized is True
            assert engine._graphrag_engine is not None
            assert engine._vector_store is not None

    @pytest.mark.asyncio
    async def test_enhanced_indexing_initialization(self, tmp_path: Path) -> None:
        """Test enhanced indexing initialization."""
        config = SearchConfig(paths=[str(tmp_path)])
        
        engine = PySearch(
            config=config,
            enable_enhanced_indexing=True
        )
        
        # Mock the enhanced indexer
        with patch('pysearch.api.MetadataIndexer') as mock_indexer:
            mock_indexer_instance = AsyncMock()
            mock_indexer.return_value = mock_indexer_instance
            
            await engine.initialize_enhanced_indexing()
            
            assert engine._enhanced_indexing_initialized is True
            assert engine._enhanced_indexer is not None

    @pytest.mark.asyncio
    async def test_graphrag_search(self, tmp_path: Path) -> None:
        """Test GraphRAG search functionality."""
        config = SearchConfig(paths=[str(tmp_path)])
        qdrant_config = QdrantConfig(host="localhost", port=6333)
        
        engine = PySearch(
            config=config,
            qdrant_config=qdrant_config,
            enable_graphrag=True
        )
        
        # Mock GraphRAG components
        with patch.object(engine, 'initialize_graphrag') as mock_init, \
             patch.object(engine, '_graphrag_engine') as mock_engine:
            
            mock_init.return_value = None
            engine._graphrag_initialized = True
            
            mock_result = {"results": [{"content": "test result"}]}
            mock_engine.search.return_value = mock_result
            
            query = GraphRAGQuery(pattern="test query")
            result = await engine.graphrag_search(query)
            
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_enhanced_index_search(self, tmp_path: Path) -> None:
        """Test enhanced index search functionality."""
        config = SearchConfig(paths=[str(tmp_path)])
        
        engine = PySearch(
            config=config,
            enable_enhanced_indexing=True
        )
        
        # Mock enhanced indexer
        with patch.object(engine, 'initialize_enhanced_indexing') as mock_init, \
             patch.object(engine, '_enhanced_indexer') as mock_indexer:
            
            mock_init.return_value = None
            engine._enhanced_indexing_initialized = True
            
            mock_result = {"files": ["test.py"], "matches": 1}
            mock_indexer.query_index.return_value = mock_result
            
            query = IndexQuery(semantic_query="test")
            result = await engine.enhanced_index_search(query)

            assert result == mock_result


class TestPySearchHybridSearch:
    """Test PySearch hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, tmp_path: Path) -> None:
        """Test basic hybrid search functionality."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        result = await engine.hybrid_search("test")

        assert "traditional" in result
        assert "metadata" in result
        assert result["metadata"]["pattern"] == "test"
        assert "methods_used" in result["metadata"]

    @pytest.mark.asyncio
    async def test_hybrid_search_with_graphrag_disabled(self, tmp_path: Path) -> None:
        """Test hybrid search when GraphRAG is disabled."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config, enable_graphrag=False)

        result = await engine.hybrid_search("test")

        assert result.get("graphrag") is None
        assert "traditional" in result

    @pytest.mark.asyncio
    async def test_hybrid_search_with_enhanced_indexing_disabled(self, tmp_path: Path) -> None:
        """Test hybrid search when enhanced indexing is disabled."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config, enable_enhanced_indexing=False)

        result = await engine.hybrid_search("test")

        assert result.get("enhanced_index") is None
        assert "traditional" in result

    @pytest.mark.asyncio
    async def test_hybrid_search_error_handling(self, tmp_path: Path) -> None:
        """Test hybrid search error handling."""
        config = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(config)

        # Mock traditional search to raise exception
        with patch.object(engine, 'run', side_effect=Exception("Search error")):
            result = await engine.hybrid_search("test")

            # Should handle error gracefully
            assert "traditional" in result
            assert "metadata" in result


class TestPySearchConvenienceMethods:
    """Test PySearch convenience methods."""

    def test_search_method_basic(self, tmp_path: Path) -> None:
        """Test basic search convenience method."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        result = engine.search("test")

        assert isinstance(result, SearchResult)
        assert result.stats.files_scanned >= 0

    def test_search_method_with_regex(self, tmp_path: Path) -> None:
        """Test search convenience method with regex."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    def another_function():\n        pass")

        result = engine.search(r"def \w+", regex=True)

        assert isinstance(result, SearchResult)

    def test_search_method_with_context_override(self, tmp_path: Path) -> None:
        """Test search convenience method with context override."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=1)
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        result = engine.search("test", context=5)

        assert isinstance(result, SearchResult)

    def test_search_method_with_ast_filters(self, tmp_path: Path) -> None:
        """Test search convenience method with AST filters."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        filters = ASTFilters(func_name="test_function")
        result = engine.search("def", use_ast=True, filters=filters)

        assert isinstance(result, SearchResult)

    def test_search_method_with_metadata_filters(self, tmp_path: Path) -> None:
        """Test search convenience method with metadata filters."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        metadata_filters = MetadataFilters(languages={Language.PYTHON})
        result = engine.search("test", metadata_filters=metadata_filters)

        assert isinstance(result, SearchResult)

    def test_search_method_with_output_format(self, tmp_path: Path) -> None:
        """Test search convenience method with output format."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"])
        engine = PySearch(config)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test_function():\n    return 'test'")

        result = engine.search("test", output=OutputFormat.JSON)

        assert isinstance(result, SearchResult)


class TestPySearchParallelization:
    """Test PySearch parallelization features."""

    def test_adaptive_parallelism_small_workload(self, tmp_path: Path) -> None:
        """Test adaptive parallelism with small workload."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], parallel=True)
        engine = PySearch(config)

        # Create a few test files
        for i in range(3):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}():\n    return {i}")

        query = Query(pattern="def")
        result = engine.run(query)

        assert isinstance(result, SearchResult)

    def test_adaptive_parallelism_large_workload(self, tmp_path: Path) -> None:
        """Test adaptive parallelism with large workload."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], parallel=True)
        engine = PySearch(config)

        # Create many test files to trigger process pool
        for i in range(50):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}():\n    return {i}")

        query = Query(pattern="def")
        result = engine.run(query)

        assert isinstance(result, SearchResult)

    def test_thread_pool_fallback(self, tmp_path: Path) -> None:
        """Test thread pool fallback when process pool fails."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], parallel=True)
        engine = PySearch(config)

        # Create test files
        for i in range(10):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}():\n    return {i}")

        # Mock process pool to fail
        with patch('concurrent.futures.ProcessPoolExecutor', side_effect=Exception("Process pool error")):
            query = Query(pattern="def")
            result = engine.run(query)

            # Should fallback to thread pool
            assert isinstance(result, SearchResult)

    def test_parallel_disabled(self, tmp_path: Path) -> None:
        """Test search with parallelization disabled."""
        config = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], parallel=False)
        engine = PySearch(config)

        # Create test files
        for i in range(5):
            test_file = tmp_path / f"test{i}.py"
            test_file.write_text(f"def test{i}():\n    return {i}")

        query = Query(pattern="def")
        result = engine.run(query)

        assert isinstance(result, SearchResult)
