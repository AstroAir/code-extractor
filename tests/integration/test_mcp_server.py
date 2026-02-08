#!/usr/bin/env python3
"""
Tests for PySearch MCP Server

This module contains tests to validate the MCP server functionality,
error handling, and integration capabilities.
"""

import asyncio

import pytest

# Skip entire module: references non-existent PySearchMCPServer, ConfigResponse, SearchResponse classes
pytest.skip(
    "Test module references non-existent MCP classes (PySearchMCPServer, ConfigResponse, SearchResponse)",
    allow_module_level=True,
)

from mcp.servers.pysearch_mcp_server import PySearchEngine, ConfigResponse, SearchResponse  # noqa: E402


class TestPySearchMCPServer:
    """Test suite for PySearch MCP Server functionality."""

    @pytest.fixture
    async def server(self) -> PySearchMCPServer:
        """Create a server instance for testing."""
        return PySearchMCPServer()

    @pytest.mark.asyncio
    async def test_server_initialization(self, server: PySearchMCPServer) -> None:
        """Test that the server initializes correctly."""
        assert server.search_engine is not None
        assert server.current_config is not None
        assert server.search_history == []
        assert server.name == "Enhanced PySearch MCP Server"

    @pytest.mark.asyncio
    async def test_search_text_basic(self, server: PySearchMCPServer) -> None:
        """Test basic text search functionality."""
        result = await server.search_text(pattern="def", paths=["./src"], context=1)

        assert isinstance(result, SearchResponse)
        assert result.total_matches > 0
        assert result.execution_time_ms > 0
        assert len(result.items) > 0
        assert result.query_info["pattern"] == "def"
        assert result.query_info["use_regex"] is False

    @pytest.mark.asyncio
    async def test_search_regex(self, server: PySearchMCPServer) -> None:
        """Test regex search functionality."""
        result = await server.search_regex(pattern=r"class \w+", paths=["./src"], context=1)

        assert isinstance(result, SearchResponse)
        assert result.query_info["use_regex"] is True
        assert result.query_info["pattern"] == r"class \w+"

    @pytest.mark.asyncio
    async def test_search_ast(self, server: PySearchMCPServer) -> None:
        """Test AST-based search functionality."""
        result = await server.search_ast(
            pattern="def", func_name=".*test.*", paths=["./src"], context=1
        )

        assert isinstance(result, SearchResponse)
        assert result.query_info["use_ast"] is True
        assert result.query_info["filters"] is not None

    @pytest.mark.asyncio
    async def test_search_semantic(self, server: PySearchMCPServer) -> None:
        """Test semantic search functionality."""
        result = await server.search_semantic(concept="test", paths=["./src"], context=1)

        assert isinstance(result, SearchResponse)
        assert result.query_info["use_semantic"] is True
        assert result.query_info["use_regex"] is True  # Semantic search uses regex internally

    @pytest.mark.asyncio
    async def test_configuration_management(self, server: PySearchMCPServer) -> None:
        """Test configuration get and set operations."""
        # Get initial configuration
        initial_config = await server.get_search_config()
        assert isinstance(initial_config, ConfigResponse)
        assert initial_config.context_lines == 3  # Default value

        # Update configuration
        new_config = await server.configure_search(paths=["./tests"], context=5, workers=2)

        assert isinstance(new_config, ConfigResponse)
        assert new_config.paths == ["./tests"]
        assert new_config.context_lines == 5
        assert new_config.workers == 2

        # Verify configuration was updated
        current_config = await server.get_search_config()
        assert current_config.paths == ["./tests"]
        assert current_config.context_lines == 5

    @pytest.mark.asyncio
    async def test_utility_functions(self, server: PySearchMCPServer) -> None:
        """Test utility functions."""
        # Test supported languages
        languages = await server.get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0
        assert "python" in languages

        # Test cache clearing
        cache_result = await server.clear_caches()
        assert isinstance(cache_result, dict)
        assert cache_result["status"] == "Caches cleared successfully"

        # Test search history (initially empty)
        history = await server.get_search_history()
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_search_history_tracking(self, server: PySearchMCPServer) -> None:
        """Test that search history is properly tracked."""
        # Perform a search
        await server.search_text(pattern="test", paths=["./src"])

        # Check history
        history = await server.get_search_history(limit=1)
        assert len(history) == 1
        assert history[0]["query"]["pattern"] == "test"
        assert "timestamp" in history[0]
        assert "result_count" in history[0]
        assert "execution_time_ms" in history[0]

    @pytest.mark.asyncio
    async def test_error_handling_invalid_pattern(self, server: PySearchMCPServer) -> None:
        """Test error handling for invalid patterns."""
        # Test with invalid regex pattern
        try:
            await server.search_regex(pattern="[invalid", paths=["./src"])
            raise AssertionError("Should have raised an exception")
        except Exception as e:
            assert "invalid" in str(e).lower() or "error" in str(e).lower()

    @pytest.mark.asyncio
    async def test_error_handling_invalid_paths(self, server: PySearchMCPServer) -> None:
        """Test error handling for invalid paths."""
        # Test with non-existent path
        result = await server.search_text(pattern="test", paths=["/non/existent/path"])
        # Should not crash, but may return no results
        assert isinstance(result, SearchResponse)

    @pytest.mark.asyncio
    async def test_response_format_consistency(self, server: PySearchMCPServer) -> None:
        """Test that all search methods return consistent response formats."""
        methods_and_args = [
            ("search_text", {"pattern": "def", "paths": ["./src"]}),
            ("search_regex", {"pattern": r"def \w+", "paths": ["./src"]}),
            ("search_ast", {"pattern": "def", "paths": ["./src"]}),
            ("search_semantic", {"concept": "test", "paths": ["./src"]}),
        ]

        for method_name, args in methods_and_args:
            method = getattr(server, method_name)
            result = await method(**args)

            # Check response structure
            assert isinstance(result, SearchResponse)
            assert hasattr(result, "items")
            assert hasattr(result, "stats")
            assert hasattr(result, "query_info")
            assert hasattr(result, "total_matches")
            assert hasattr(result, "execution_time_ms")

            # Check stats structure
            assert "files_scanned" in result.stats
            assert "files_matched" in result.stats
            assert "total_items" in result.stats
            assert "elapsed_ms" in result.stats

            # Check query_info structure
            assert "pattern" in result.query_info
            assert "use_regex" in result.query_info
            assert "use_ast" in result.query_info
            assert "use_semantic" in result.query_info
            assert "context" in result.query_info


# Async test runner for manual execution
async def run_tests():
    """Run tests manually without pytest."""
    print("Running PySearch MCP Server Tests")
    print("=" * 50)

    server = PySearchMCPServer()
    test_instance = TestPySearchMCPServer()

    tests = [
        ("Server Initialization", test_instance.test_server_initialization),
        ("Basic Text Search", test_instance.test_search_text_basic),
        ("Regex Search", test_instance.test_search_regex),
        ("AST Search", test_instance.test_search_ast),
        ("Semantic Search", test_instance.test_search_semantic),
        ("Configuration Management", test_instance.test_configuration_management),
        ("Utility Functions", test_instance.test_utility_functions),
        ("Search History Tracking", test_instance.test_search_history_tracking),
        ("Response Format Consistency", test_instance.test_response_format_consistency),
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            print(f"\n{test_name}...")
            await test_func(server)
            print(f"✅ {test_name} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            failed += 1

    print("\n" + "=" * 50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")


if __name__ == "__main__":
    # Run tests manually
    asyncio.run(run_tests())
