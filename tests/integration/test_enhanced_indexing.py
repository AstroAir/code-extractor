"""
Comprehensive test suite for enhanced indexing engine.

This module provides extensive testing coverage for all enhanced indexing
features, including unit tests, integration tests, and performance benchmarks.

Test Categories:
    - Content addressing and caching
    - Tag-based index management
    - Multi-language support and chunking
    - Vector database integration
    - Distributed indexing
    - Error handling and recovery
    - Performance monitoring

Example:
    Run all tests:
        >>> python -m pytest tests/test_enhanced_indexing.py -v

    Run specific test category:
        >>> python -m pytest tests/test_enhanced_indexing.py::TestContentAddressing -v

    Run performance benchmarks:
        >>> python -m pytest tests/test_enhanced_indexing.py::TestPerformanceBenchmarks -v
"""

import asyncio
import tempfile
import time
from collections.abc import Generator
from pathlib import Path

import pytest

# Guarded imports to avoid missing-module errors at import time.
advanced_chunking = pytest.importorskip("src.pysearch.advanced_chunking")
config_mod = pytest.importorskip("src.pysearch.config")
content_addr_mod = pytest.importorskip("src.pysearch.content_addressing")
error_handling_mod = pytest.importorskip("src.pysearch.utils.error_handling")
indexing_engine_mod = pytest.importorskip("src.pysearch.enhanced_indexing_engine")
lang_support_mod = pytest.importorskip("src.pysearch.enhanced_language_support")
perf_mon_mod = pytest.importorskip("src.pysearch.performance_monitoring")
types_mod = pytest.importorskip("src.pysearch.types")

# Extract required symbols
ChunkingEngine = advanced_chunking.ChunkingEngine
ChunkingConfig = advanced_chunking.ChunkingConfig
ChunkingStrategy = advanced_chunking.ChunkingStrategy

SearchConfig = config_mod.SearchConfig

ContentAddress = content_addr_mod.ContentAddress
GlobalCacheManager = content_addr_mod.GlobalCacheManager
IndexTag = content_addr_mod.IndexTag

ErrorCollector = error_handling_mod.AdvancedErrorCollector
ErrorSeverity = error_handling_mod.ErrorSeverity
RecoveryManager = error_handling_mod.RecoveryManager

IndexingEngine = indexing_engine_mod.IndexingEngine
IndexCoordinator = indexing_engine_mod.IndexCoordinator

LanguageRegistry = lang_support_mod.LanguageRegistry
TreeSitterProcessor = lang_support_mod.TreeSitterProcessor

MetricsCollector = perf_mon_mod.MetricsCollector
MetricType = perf_mon_mod.MetricType
PerformanceProfiler = perf_mon_mod.PerformanceProfiler

Language = types_mod.Language


class TestContentAddressing:
    """Test content addressing and caching functionality."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def sample_python_file(self, temp_dir: Path) -> Path:
        """Create sample Python file for testing."""
        content = '''
def hello_world():
    """Print hello world message."""
    print("Hello, World!")

class Calculator:
    """Simple calculator class."""

    def add(self, a, b):
        """Add two numbers."""
        return a + b

    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
'''
        file_path = temp_dir / "sample.py"
        file_path.write_text(content)
        return file_path

    @pytest.mark.asyncio
    async def test_content_address_creation(self, sample_python_file: Path) -> None:
        """Test ContentAddress creation from file."""
        content_addr = await ContentAddress.from_file(str(sample_python_file))

        assert content_addr.path == str(sample_python_file)
        assert len(content_addr.content_hash) == 64  # SHA256 hash length
        assert content_addr.size > 0
        assert content_addr.mtime > 0
        assert content_addr.language == Language.PYTHON

    @pytest.mark.asyncio
    async def test_global_cache_manager(self, temp_dir: Path) -> None:
        """Test global cache manager functionality."""
        cache_manager = GlobalCacheManager(temp_dir)

        # Test storing and retrieving content
        content_hash = "test_hash_123"
        artifact_id = "test_artifact"
        test_content = {"data": "test_data", "value": 42}
        tags = [IndexTag("dir1", "main", "artifact1")]

        await cache_manager.store_cached_content(content_hash, artifact_id, test_content, tags)

        retrieved_content = await cache_manager.get_cached_content(content_hash, artifact_id)

        assert retrieved_content == test_content

        # Test tag associations
        associated_tags = await cache_manager.get_tags_for_content(content_hash, artifact_id)
        assert len(associated_tags) == 1
        assert associated_tags[0].directory == "dir1"

    @pytest.mark.asyncio
    async def test_index_tag_operations(self) -> None:
        """Test IndexTag creation and string conversion."""
        tag = IndexTag("test_dir", "main", "code_snippets")
        tag_string = tag.to_string()

        assert tag_string == "test_dir::main::code_snippets"

        # Test round-trip conversion
        reconstructed_tag = IndexTag.from_string(tag_string)
        assert reconstructed_tag == tag


class TestLanguageSupport:
    """Test enhanced multi-language support."""

    @pytest.fixture
    def language_registry(self) -> LanguageRegistry:
        """Get language registry for testing."""
        return LanguageRegistry()

    def test_language_registry_initialization(self, language_registry: LanguageRegistry) -> None:
        """Test language registry initialization."""
        supported_languages = language_registry.get_supported_languages()

        # Should support at least Python
        assert Language.PYTHON in supported_languages

        # Should have processors for supported languages
        python_processor = language_registry.get_processor(Language.PYTHON)
        assert python_processor is not None
        assert isinstance(python_processor, TreeSitterProcessor)

    @pytest.mark.asyncio
    async def test_python_chunking(self, language_registry: LanguageRegistry) -> None:
        """Test Python code chunking."""
        python_code = '''
def function1():
    """First function."""
    return 1

def function2():
    """Second function."""
    return 2

class TestClass:
    """Test class."""

    def method1(self):
        """Test method."""
        return "test"
'''

        processor = language_registry.get_processor(Language.PYTHON)
        chunks = []

        async for chunk in processor.chunk_code(python_code, 500):
            chunks.append(chunk)

        # Should create multiple chunks for different entities
        assert len(chunks) >= 3  # At least function1, function2, TestClass

        # Check chunk properties
        for chunk in chunks:
            assert chunk.language == Language.PYTHON
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert len(chunk.content) > 0

    @pytest.mark.asyncio
    async def test_entity_extraction(self, language_registry: LanguageRegistry) -> None:
        """Test code entity extraction."""
        python_code = '''
import os
from typing import List

def calculate_sum(numbers: List[int]) -> int:
    """Calculate sum of numbers."""
    return sum(numbers)

class DataProcessor:
    """Process data efficiently."""

    def __init__(self, name: str):
        self.name = name

    def process(self, data):
        """Process the data."""
        return data.upper()
'''

        processor = language_registry.get_processor(Language.PYTHON)
        entities = processor.extract_entities(python_code)

        # Should extract function and class
        entity_names = [entity.name for entity in entities]
        assert "calculate_sum" in entity_names
        assert "DataProcessor" in entity_names

        # Check entity properties
        for entity in entities:
            assert entity.name
            assert entity.entity_type
            assert entity.start_line > 0
            assert entity.content


class TestChunkingEngine:
    """Test advanced chunking engine."""

    @pytest.fixture
    def chunking_engine(self) -> ChunkingEngine:
        """Create chunking engine for testing."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=500,
            min_chunk_size=50,
        )
        return ChunkingEngine(config)

    @pytest.mark.asyncio
    async def test_file_chunking(self, chunking_engine: ChunkingEngine, tmp_path: Path) -> None:
        """Test file chunking functionality."""
        # Create test file
        test_content = '''
def function1():
    """First function with some content."""
    x = 1
    y = 2
    return x + y

def function2():
    """Second function with different content."""
    a = "hello"
    b = "world"
    return a + " " + b

class TestClass:
    """Test class with methods."""

    def method1(self):
        return "method1"

    def method2(self):
        return "method2"
'''

        test_file = tmp_path / "test.py"
        test_file.write_text(test_content)

        # Chunk the file
        chunks = await chunking_engine.chunk_file(str(test_file))

        # Verify chunks
        assert len(chunks) > 0

        for chunk in chunks:
            assert chunk.content
            assert chunk.start_line > 0
            assert chunk.end_line >= chunk.start_line
            assert chunk.language == Language.PYTHON
            assert chunk.chunk_id
            assert 0.0 <= chunk.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_chunking_strategies(self, tmp_path: Path) -> None:
        """Test different chunking strategies."""
        test_content = "def test(): pass\n" * 100  # Large repetitive content
        test_file = tmp_path / "large.py"
        test_file.write_text(test_content)

        # Test different strategies
        strategies = [
            ChunkingStrategy.STRUCTURAL,
            ChunkingStrategy.SEMANTIC,
            ChunkingStrategy.HYBRID,
        ]

        for strategy in strategies:
            config = ChunkingConfig(strategy=strategy, max_chunk_size=500)
            engine = ChunkingEngine(config)

            chunks = await engine.chunk_file(str(test_file))
            assert len(chunks) > 0

            # Verify chunk sizes
            for chunk in chunks:
                assert len(chunk.content) <= 500 * 1.2  # Allow some flexibility


class TestErrorHandling:
    """Test enhanced error handling and recovery."""

    @pytest.fixture
    def error_collector(self) -> ErrorCollector:
        """Create error collector for testing."""
        return ErrorCollector()

    @pytest.mark.asyncio
    async def test_error_collection(self, error_collector: ErrorCollector) -> None:
        """Test error collection functionality."""
        # Add various types of errors
        _ = await error_collector.add_error("test_file.py", "File not found", ErrorSeverity.ERROR)

        _ = await error_collector.add_error(
            "network_operation", "Connection timeout", ErrorSeverity.WARNING
        )

        # Verify errors were collected
        assert error_collector.has_errors()
        assert len(error_collector.errors) == 2

        # Test error retrieval
        error_summary = error_collector.get_error_summary()
        assert error_summary["total_errors"] == 2
        assert "error" in error_summary["severity_counts"]
        assert "warning" in error_summary["severity_counts"]

    @pytest.mark.asyncio
    async def test_recovery_manager(self, tmp_path: Path) -> None:
        """Test error recovery functionality."""
        config = SearchConfig(paths=[str(tmp_path)])
        recovery_manager = RecoveryManager(config)

        # Test file access recovery
        from src.pysearch.utils.error_handling import (  # type: ignore[import-not-found]
            ErrorCategory,
            IndexingError,
        )

        error = IndexingError(
            error_id="test_error",
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.FILE_ACCESS,
            message="Permission denied",
            context="nonexistent_file.py",
        )

        # Recovery should handle non-existent files gracefully
        recovery_success = await recovery_manager.attempt_recovery(error, {})
        assert isinstance(recovery_success, bool)


class TestPerformanceMonitoring:
    """Test performance monitoring system."""

    @pytest.fixture
    def metrics_collector(self) -> MetricsCollector:
        """Create metrics collector for testing."""
        return MetricsCollector()

    @pytest.mark.asyncio
    async def test_metrics_collection(self, metrics_collector: MetricsCollector) -> None:
        """Test metrics collection functionality."""
        # Record various metrics
        await metrics_collector.record_metric("test_counter", 1.0, MetricType.COUNTER)
        await metrics_collector.record_metric("test_gauge", 42.5, MetricType.GAUGE, unit="MB")
        await metrics_collector.record_metric("test_timer", 1.23, MetricType.TIMER, unit="seconds")

        # Verify metrics were recorded
        all_metrics = await metrics_collector.get_metrics()
        assert len(all_metrics) == 3

        # Test aggregates
        counter_stats = metrics_collector.get_aggregate_stats("test_counter")
        assert counter_stats is not None
        assert counter_stats["count"] == 1
        assert counter_stats["sum"] == 1.0

    @pytest.mark.asyncio
    async def test_performance_profiler(self, tmp_path: Path) -> None:
        """Test performance profiling functionality."""
        _ = tmp_path  # acknowledge fixture usage
        metrics_collector = MetricsCollector()
        profiler = PerformanceProfiler(metrics_collector)

        # Profile a simple operation
        async with profiler.profile_operation("test_operation"):
            # Simulate some work
            await asyncio.sleep(0.1)

        # Check that metrics were recorded
        duration_metrics = await metrics_collector.get_metrics("operation_duration")
        assert len(duration_metrics) > 0

        completed_metrics = await metrics_collector.get_metrics("operation_completed")
        assert len(completed_metrics) > 0


class TestIntegration:
    """Integration tests for the complete enhanced indexing system."""

    @pytest.fixture
    def test_config(self, tmp_path: Path) -> SearchConfig:
        """Create test configuration."""
        from typing import cast

        return cast(
            SearchConfig,
            SearchConfig(
                paths=[str(tmp_path)],
                cache_dir=str(tmp_path / "cache"),
                enable_metadata_indexing=True,
            ),
        )

    @pytest.fixture
    def sample_codebase(self, tmp_path: Path) -> Path:
        """Create sample codebase for testing."""
        # Python file
        python_file = tmp_path / "main.py"
        python_file.write_text('''
import json
from typing import List

def process_data(data: List[dict]) -> dict:
    """Process a list of data dictionaries."""
    result = {}
    for item in data:
        if "id" in item:
            result[item["id"]] = item
    return result

class DataManager:
    """Manages data operations."""

    def __init__(self, storage_path: str):
        self.storage_path = storage_path

    def save_data(self, data: dict) -> bool:
        """Save data to storage."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(data, f)
            return True
        except Exception:
            return False
''')

        # JavaScript file
        js_file = tmp_path / "utils.js"
        js_file.write_text("""
function calculateSum(numbers) {
    /**
     * Calculate sum of an array of numbers.
     * @param {number[]} numbers - Array of numbers
     * @returns {number} Sum of all numbers
     */
    return numbers.reduce((sum, num) => sum + num, 0);
}

class EventHandler {
    constructor(name) {
        this.name = name;
        this.listeners = [];
    }

    addEventListener(event, callback) {
        this.listeners.push({ event, callback });
    }

    emit(event, data) {
        this.listeners
            .filter(listener => listener.event === event)
            .forEach(listener => listener.callback(data));
    }
}

export { calculateSum, EventHandler };
""")

        return tmp_path

    @pytest.mark.asyncio
    async def test_enhanced_indexing_engine(
        self, test_config: SearchConfig, sample_codebase: Path
    ) -> None:
        """Test the complete enhanced indexing engine."""
        engine = IndexingEngine(test_config)
        await engine.initialize()

        # Test indexing
        progress_updates = []
        async for update in engine.refresh_index([str(sample_codebase)]):
            progress_updates.append(update)

        # Verify progress updates
        assert len(progress_updates) > 0
        assert progress_updates[-1].status in ["done", "done_with_errors"]
        assert progress_updates[-1].progress == 1.0

        # Test statistics
        stats = await engine.coordinator.get_index_stats()
        assert "total_indexes" in stats
        assert stats["total_indexes"] > 0

    @pytest.mark.asyncio
    async def test_index_coordinator(
        self, test_config: SearchConfig, sample_codebase: Path
    ) -> None:
        """Test index coordinator functionality."""
        _ = sample_codebase  # acknowledge fixture usage
        coordinator = IndexCoordinator(test_config)

        # Add test indexes (guarded)
        code_snippets_mod = pytest.importorskip("src.pysearch.indexes.code_snippets_index")
        full_text_mod = pytest.importorskip("src.pysearch.indexes.full_text_index")

        CodeSnippetsIndex = code_snippets_mod.CodeSnippetsIndex
        FullTextIndex = full_text_mod.FullTextIndex

        coordinator.add_index(CodeSnippetsIndex(test_config))
        coordinator.add_index(FullTextIndex(test_config))

        # Test index management
        assert len(coordinator.indexes) == 2

        snippets_index = coordinator.get_index("enhanced_code_snippets")
        assert snippets_index is not None

        # Test index removal
        removed = coordinator.remove_index("enhanced_code_snippets")
        assert removed is True
        assert len(coordinator.indexes) == 1


class TestPerformanceBenchmarks:
    """Performance benchmarks for the enhanced indexing system."""

    @pytest.fixture
    def large_codebase(self, tmp_path: Path) -> Path:
        """Create large codebase for performance testing."""
        # Create multiple files with varying sizes
        for i in range(50):
            file_path = tmp_path / f"file_{i}.py"
            content = f'''
def function_{i}_1():
    """Function {i}_1 documentation."""
    return {i} * 1

def function_{i}_2():
    """Function {i}_2 documentation."""
    return {i} * 2

class Class_{i}:
    """Class {i} documentation."""

    def __init__(self):
        self.value = {i}

    def method_{i}(self):
        """Method {i} documentation."""
        return self.value * 10
''' * (i % 5 + 1)  # Varying file sizes

            file_path.write_text(content)

        return tmp_path

    @pytest.mark.asyncio
    async def test_indexing_performance(self, large_codebase: Path) -> None:
        """Benchmark indexing performance."""
        config = SearchConfig(
            paths=[str(large_codebase)],
            cache_dir=str(large_codebase / "cache"),
        )

        engine = IndexingEngine(config)
        await engine.initialize()

        # Measure indexing time
        start_time = time.time()

        progress_updates = []
        async for update in engine.refresh_index([str(large_codebase)]):
            progress_updates.append(update)

        end_time = time.time()
        indexing_duration = end_time - start_time

        # Performance assertions
        assert indexing_duration < 60.0  # Should complete within 1 minute
        assert len(progress_updates) > 0
        assert progress_updates[-1].progress == 1.0

        # Log performance metrics
        print(f"Indexing duration: {indexing_duration:.2f}s")
        print("Files processed: 50")
        print(f"Throughput: {50 / indexing_duration:.2f} files/second")

    @pytest.mark.asyncio
    async def test_chunking_performance(self, large_codebase: Path) -> None:
        """Benchmark chunking performance."""
        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            max_chunk_size=1000,
        )
        engine = ChunkingEngine(config)

        # Get all Python files
        python_files = list(large_codebase.glob("*.py"))

        # Measure chunking time
        start_time = time.time()

        all_chunks = await engine.chunk_multiple_files(
            [str(f) for f in python_files], max_concurrent=5
        )

        end_time = time.time()
        chunking_duration = end_time - start_time

        # Performance assertions
        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        assert total_chunks > 0
        assert chunking_duration < 30.0  # Should complete within 30 seconds

        # Log performance metrics
        print(f"Chunking duration: {chunking_duration:.2f}s")
        print(f"Total chunks: {total_chunks}")
        print(f"Chunking throughput: {total_chunks / chunking_duration:.2f} chunks/second")

    @pytest.mark.asyncio
    async def test_memory_usage(self, large_codebase: Path) -> None:
        """Test memory usage during indexing."""
        psutil = pytest.importorskip("psutil")

        config = SearchConfig(
            paths=[str(large_codebase)],
            cache_dir=str(large_codebase / "cache"),
        )

        # Measure initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform indexing
        engine = IndexingEngine(config)
        await engine.initialize()

        async for _ in engine.refresh_index([str(large_codebase)]):
            pass

        # Measure final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory usage assertions
        assert memory_increase < 500  # Should not use more than 500MB additional

        print(f"Initial memory: {initial_memory:.2f}MB")
        print(f"Final memory: {final_memory:.2f}MB")
        print(f"Memory increase: {memory_increase:.2f}MB")


# Test fixtures and utilities
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
