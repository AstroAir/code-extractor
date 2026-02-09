#!/usr/bin/env python3
"""
Comprehensive Tests for Enhanced PySearch MCP Server Features

This module contains tests for all enhanced features including:
- Fuzzy search functionality
- Multi-pattern search with logical operators
- File content analysis and metrics
- Search result ranking
- Advanced filtering capabilities
- Resource management
- Progress reporting
- Context management
- Prompt templates
- Composition support
"""

import asyncio
import tempfile
from pathlib import Path

import pytest

# Skip entire module: references non-existent modules and classes
# (mcp.servers.mcp_server, mcp.shared.composition, mcp.shared.prompts, mcp.shared.resources)
pytest.skip(
    "Test module references non-existent MCP classes (PySearchMCPServer, SearchComposer, etc.)",
    allow_module_level=True,
)

from mcp.servers.mcp_server import (  # noqa: E402
    FuzzySearchConfig,
    MultiPatternQuery,
    PySearchMCPServer,
    SearchFilter,
    SearchOperator,
)
from mcp.shared.composition import SearchComposer  # noqa: E402
from mcp.shared.progress import ProgressAwareSearchServer  # noqa: E402
from mcp.shared.prompts import MCPPromptManager, PromptCategory  # noqa: E402
from mcp.shared.resources import MCPResourceManager  # noqa: E402


class TestMCPFeatures:
    """Test suite for enhanced PySearch MCP Server features."""

    @pytest.fixture
    async def mcp_server(self):
        """Create an MCP server instance for testing."""
        return PySearchMCPServer()

    @pytest.fixture
    async def progress_server(self):
        """Create a progress-aware server instance for testing."""
        return ProgressAwareSearchServer()

    @pytest.fixture
    async def resource_manager(self, mcp_server):
        """Create a resource manager instance for testing."""
        return MCPResourceManager(mcp_server)

    @pytest.fixture
    async def prompt_manager(self):
        """Create a prompt manager instance for testing."""
        return MCPPromptManager()

    @pytest.fixture
    async def search_composer(self, mcp_server):
        """Create a search composer instance for testing."""
        return SearchComposer(mcp_server)

    @pytest.fixture
    def temp_test_files(self):
        """Create temporary test files for analysis."""
        temp_dir = tempfile.mkdtemp()

        # Create test Python file
        python_file = Path(temp_dir) / "test_file.py"
        python_content = '''
def complex_function(a, b, c):
    """A complex function for testing."""
    if a > 0:
        for i in range(b):
            if i % 2 == 0:
                for j in range(c):
                    if j > i:
                        print(f"Complex logic: {i}, {j}")
                    else:
                        continue
            else:
                pass
    return a + b + c

class TestClass:
    """A test class."""
    
    def __init__(self):
        self.value = 0
    
    def method_with_todo(self):
        # TODO: Implement this method
        pass
'''
        python_file.write_text(python_content)

        # Create test JavaScript file
        js_file = Path(temp_dir) / "test_file.js"
        js_content = """
function performanceBottleneck(data) {
    // This function has performance issues
    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data.length; j++) {
            if (data[i] === data[j] && i !== j) {
                console.log("Duplicate found");
            }
        }
    }
}

const API_KEY = "hardcoded-api-key-123"; // Security issue

function sqlInjection(userInput) {
    const query = "SELECT * FROM users WHERE name = '" + userInput + "'";
    return query;
}
"""
        js_file.write_text(js_content)

        yield temp_dir

        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    # Fuzzy Search Tests
    @pytest.mark.asyncio
    async def test_fuzzy_search_basic(self, mcp_server, temp_test_files):
        """Test basic fuzzy search functionality."""
        if not hasattr(mcp_server, "search_fuzzy"):
            pytest.skip("Fuzzy search not available (rapidfuzz not installed)")

        config = FuzzySearchConfig(similarity_threshold=0.5, max_results=10)

        try:
            result = await mcp_server.search_fuzzy(
                pattern="function", paths=[temp_test_files], config=config, context=2
            )

            assert isinstance(result.items, list)
            assert result.query_info["use_fuzzy"] is True
            assert "fuzzy_config" in result.query_info

        except ValueError as e:
            if "Fuzzy search not available" in str(e):
                pytest.skip("Fuzzy search dependencies not installed")
            else:
                raise

    # Multi-pattern Search Tests
    @pytest.mark.asyncio
    async def test_multi_pattern_search_or(self, mcp_server, temp_test_files):
        """Test multi-pattern search with OR operator."""
        query = MultiPatternQuery(
            patterns=["function", "class"], operator=SearchOperator.OR, use_regex=False
        )

        result = await mcp_server.search_multi_pattern(
            query=query, paths=[temp_test_files], context=1
        )

        assert isinstance(result.items, list)
        assert result.query_info["operator"] == "OR"
        assert len(result.query_info["patterns"]) == 2

    @pytest.mark.asyncio
    async def test_multi_pattern_search_and(self, mcp_server, temp_test_files):
        """Test multi-pattern search with AND operator."""
        query = MultiPatternQuery(
            patterns=["def", "complex"], operator=SearchOperator.AND, use_regex=False
        )

        result = await mcp_server.search_multi_pattern(
            query=query, paths=[temp_test_files], context=1
        )

        assert isinstance(result.items, list)
        assert result.query_info["operator"] == "AND"

    # File Analysis Tests
    @pytest.mark.asyncio
    async def test_file_content_analysis(self, mcp_server, temp_test_files):
        """Test file content analysis functionality."""
        python_file = Path(temp_test_files) / "test_file.py"

        analysis = await mcp_server.analyze_file_content(
            str(python_file), include_complexity=True, include_quality_metrics=True
        )

        assert analysis.file_path == str(python_file)
        assert analysis.language == "python"
        assert analysis.functions_count > 0
        assert analysis.classes_count > 0
        assert analysis.complexity_score > 0
        assert 0 <= analysis.code_quality_score <= 100
        assert analysis.comments_ratio >= 0

    # Search Ranking Tests
    @pytest.mark.asyncio
    async def test_search_with_ranking(self, mcp_server, temp_test_files):
        """Test search with result ranking."""
        ranked_results = await mcp_server.search_with_ranking(
            pattern="function", paths=[temp_test_files], context=2, max_results=10
        )

        assert isinstance(ranked_results, list)

        if ranked_results:
            # Check that results have ranking information
            first_result = ranked_results[0]
            assert hasattr(first_result, "relevance_score")
            assert hasattr(first_result, "ranking_factors")
            assert isinstance(first_result.ranking_factors, dict)

            # Check that results are sorted by relevance
            if len(ranked_results) > 1:
                assert ranked_results[0].relevance_score >= ranked_results[1].relevance_score

    # Advanced Filtering Tests
    @pytest.mark.asyncio
    async def test_search_with_filters(self, mcp_server, temp_test_files):
        """Test search with advanced filtering."""
        search_filter = SearchFilter(
            file_extensions=[".py"], min_file_size=100, max_file_size=10000, languages=["python"]
        )

        result = await mcp_server.search_with_filters(
            pattern="def", search_filter=search_filter, paths=[temp_test_files], context=1
        )

        assert isinstance(result.items, list)
        assert "filters_applied" in result.query_info
        assert "filtered_out" in result.stats

        # Check that only Python files are included
        for item in result.items:
            assert item["file"].endswith(".py")

    # File Statistics Tests
    @pytest.mark.asyncio
    async def test_file_statistics(self, mcp_server, temp_test_files):
        """Test comprehensive file statistics."""
        stats = await mcp_server.get_file_statistics(paths=[temp_test_files], include_analysis=True)

        assert isinstance(stats, dict)
        assert "total_files" in stats
        assert "languages" in stats
        assert "file_extensions" in stats
        assert "size_distribution" in stats
        assert stats["total_files"] > 0

        # Check language detection
        assert "python" in stats["languages"] or "javascript" in stats["languages"]

    # Resource Management Tests
    @pytest.mark.asyncio
    async def test_resource_management(self, resource_manager):
        """Test MCP resource management functionality."""
        # Test getting available resources
        resources = resource_manager.get_available_resources()
        assert isinstance(resources, list)
        assert len(resources) > 0

        # Check resource structure
        first_resource = resources[0]
        assert "uri" in first_resource
        assert "name" in first_resource
        assert "description" in first_resource
        assert "mimeType" in first_resource

        # Test getting specific resource content
        config_content = await resource_manager.get_resource_content("pysearch://config/current")
        assert isinstance(config_content, dict)
        assert "resource_type" in config_content
        assert config_content["resource_type"] == "configuration"

    # Progress Reporting Tests
    @pytest.mark.asyncio
    async def test_progress_reporting(self, progress_server, temp_test_files):
        """Test progress reporting functionality."""
        progress_updates = []

        async for update in progress_server.search_with_progress(
            pattern="function", paths=[temp_test_files], context=1
        ):
            progress_updates.append(update)

        assert len(progress_updates) > 0

        # Check progress update structure
        first_update = progress_updates[0]
        assert hasattr(first_update, "operation_id")
        assert hasattr(first_update, "status")
        assert hasattr(first_update, "progress")
        assert hasattr(first_update, "current_step")

        # Check that progress increases
        if len(progress_updates) > 1:
            assert progress_updates[-1].progress >= progress_updates[0].progress

    # Prompt Template Tests
    @pytest.mark.asyncio
    async def test_prompt_templates(self, prompt_manager):
        """Test MCP prompt template functionality."""
        # Test getting available prompts
        prompts = prompt_manager.get_available_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

        # Check prompt structure
        first_prompt = prompts[0]
        assert "name" in first_prompt
        assert "id" in first_prompt
        assert "category" in first_prompt
        assert "description" in first_prompt

        # Test getting specific prompt
        security_prompt = prompt_manager.get_prompt_by_id("security_vulnerabilities")
        assert security_prompt is not None
        assert security_prompt.category == PromptCategory.SECURITY

        # Test prompt text generation
        prompt_text = prompt_manager.generate_prompt_text("security_vulnerabilities")
        assert isinstance(prompt_text, str)
        assert len(prompt_text) > 0

        # Test getting prompts by category
        security_prompts = prompt_manager.get_prompts_by_category(PromptCategory.SECURITY)
        assert len(security_prompts) > 0

    # Composition Tests
    @pytest.mark.asyncio
    async def test_search_composition(self, search_composer, temp_test_files):
        """Test search composition functionality."""
        # Test getting predefined pipelines
        pipelines = search_composer.get_predefined_pipelines()
        assert isinstance(pipelines, list)
        assert len(pipelines) > 0

        # Check pipeline structure
        first_pipeline = pipelines[0]
        assert "id" in first_pipeline
        assert "name" in first_pipeline
        assert "description" in first_pipeline
        assert "operation_count" in first_pipeline

        # Test executing a simple pipeline
        security_pipeline = search_composer.get_pipeline_by_id("security_analysis")
        if security_pipeline:
            # Modify pipeline to use test files
            for operation in security_pipeline.operations:
                operation.parameters["paths"] = [temp_test_files]

            result = await search_composer.execute_pipeline(security_pipeline)

            assert isinstance(result.final_results, list)
            assert result.total_operations > 0
            assert result.execution_time_ms > 0
            assert isinstance(result.composition_stats, dict)

    # Integration Tests
    @pytest.mark.asyncio
    async def test_session_management(self, mcp_server):
        """Test search session management."""
        # Create a session
        session = mcp_server._get_or_create_session()
        assert session.session_id is not None
        assert len(session.queries) == 0

        # Perform a search with session
        await mcp_server.search_text(pattern="test", paths=["./src"], session_id=session.session_id)

        # Check that session was updated
        updated_session = mcp_server.search_sessions[session.session_id]
        assert len(updated_session.queries) > 0

    @pytest.mark.asyncio
    async def test_cache_management(self, mcp_server, temp_test_files):
        """Test file analysis cache management."""
        python_file = Path(temp_test_files) / "test_file.py"

        # First analysis should populate cache
        analysis1 = await mcp_server.analyze_file_content(str(python_file))
        cache_size_before = len(mcp_server.file_analysis_cache)

        # Second analysis should use cache
        analysis2 = await mcp_server.analyze_file_content(str(python_file))
        cache_size_after = len(mcp_server.file_analysis_cache)

        # Results should be identical
        assert analysis1.complexity_score == analysis2.complexity_score
        assert analysis1.code_quality_score == analysis2.code_quality_score

        # Cache should not have grown
        assert cache_size_after == cache_size_before

    # Performance Tests
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self, mcp_server):
        """Test performance of enhanced features."""
        import time

        # Test basic search performance
        start_time = time.time()
        result = await mcp_server.search_text(pattern="def", paths=["./src"])
        search_time = time.time() - start_time

        assert search_time < 5.0  # Should complete within 5 seconds
        assert result.execution_time_ms > 0

        # Test ranking performance
        start_time = time.time()
        await mcp_server.search_with_ranking(pattern="class", paths=["./src"], max_results=20)
        ranking_time = time.time() - start_time

        assert ranking_time < 10.0  # Should complete within 10 seconds


# Async test runner for manual execution
async def run_enhanced_tests():
    """Run enhanced feature tests manually without pytest."""
    print("Running Enhanced PySearch MCP Server Tests")
    print("=" * 60)

    mcp_server = PySearchMCPServer()
    ProgressAwareSearchServer()
    resource_manager = MCPResourceManager(mcp_server)
    prompt_manager = MCPPromptManager()
    search_composer = SearchComposer(mcp_server)

    test_instance = TestMCPFeatures()

    tests = [
        ("Multi-pattern Search OR", test_instance.test_multi_pattern_search_or),
        ("Multi-pattern Search AND", test_instance.test_multi_pattern_search_and),
        ("File Statistics", test_instance.test_file_statistics),
        ("Resource Management", test_instance.test_resource_management),
        ("Prompt Templates", test_instance.test_prompt_templates),
        ("Search Composition", test_instance.test_search_composition),
        ("Session Management", test_instance.test_session_management),
        ("Performance Benchmarks", test_instance.test_performance_benchmarks),
    ]

    passed = 0
    failed = 0

    # Create temporary test files
    import tempfile

    temp_dir = tempfile.mkdtemp()

    try:
        for test_name, test_func in tests:
            try:
                print(f"\n{test_name}...")

                # Prepare arguments based on test function
                if "temp_test_files" in test_func.__code__.co_varnames:
                    await test_func(mcp_server, temp_dir)
                elif "resource_manager" in test_func.__code__.co_varnames:
                    await test_func(resource_manager)
                elif "prompt_manager" in test_func.__code__.co_varnames:
                    await test_func(prompt_manager)
                elif "search_composer" in test_func.__code__.co_varnames:
                    await test_func(search_composer, temp_dir)
                else:
                    await test_func(mcp_server)

                print(f"✅ {test_name} PASSED")
                passed += 1
            except Exception as e:
                print(f"❌ {test_name} FAILED: {e}")
                failed += 1

    finally:
        # Cleanup
        import shutil

        shutil.rmtree(temp_dir)

    print("\n" + "=" * 60)
    print(f"Enhanced Test Results: {passed} passed, {failed} failed")
    print(f"Success Rate: {passed/(passed+failed)*100:.1f}%")


if __name__ == "__main__":
    # Run tests manually
    asyncio.run(run_enhanced_tests())
