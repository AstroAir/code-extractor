"""
Tests for metadata indexing functionality.

This module contains tests for the metadata indexing system including
metadata indexing, entity-level indexing, and comprehensive querying.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
from typing import Generator

import pytest

from pysearch import SearchConfig
from pysearch.indexing.metadata import (
    MetadataIndexer, EntityMetadata, FileMetadata, IndexQuery, IndexStats, MetadataIndex
)
from pysearch import EntityType, Language


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config(temp_dir: Path) -> SearchConfig:
    """Create a test configuration."""
    return SearchConfig(
        paths=[str(temp_dir)],
        include=["**/*.py"],
        enable_enhanced_indexing=True
    )


@pytest.fixture
def sample_file_metadata() -> FileMetadata:
    """Sample file metadata for testing."""
    return FileMetadata(
        file_path="/test/example.py",
        size=1024,
        mtime=1234567890.0,
        language="python",
        line_count=50,
        entity_count=5,
        complexity_score=15.5,
        semantic_summary="A test module for demonstration",
        imports=["import os", "import sys"],
        dependencies=["requests", "numpy"]
    )


@pytest.fixture
def sample_entity_metadata() -> EntityMetadata:
    """Sample entity metadata for testing."""
    return EntityMetadata(
        entity_id="func_test_123",
        name="test_function",
        entity_type="function",
        file_path="/test/example.py",
        start_line=10,
        end_line=15,
        signature="def test_function(arg1: str) -> bool:",
        docstring="Test function for demonstration",
        language="python",
        scope="global",
        complexity_score=5.0,
        properties={"args": ["arg1"], "returns": "bool"}
    )


class TestEntityMetadata:
    """Test EntityMetadata functionality."""

    def test_init(self, sample_entity_metadata: EntityMetadata) -> None:
        """Test EntityMetadata initialization."""
        metadata = sample_entity_metadata

        assert metadata.entity_id == "func_test_123"
        assert metadata.name == "test_function"
        assert metadata.entity_type == "function"
        assert metadata.file_path == "/test/example.py"
        assert metadata.start_line == 10
        assert metadata.end_line == 15
        assert metadata.signature == "def test_function(arg1: str) -> bool:"
        assert metadata.docstring == "Test function for demonstration"
        assert metadata.language == "python"
        assert metadata.scope == "global"
        assert metadata.complexity_score == 5.0
        assert metadata.properties == {"args": ["arg1"], "returns": "bool"}


class TestFileMetadata:
    """Test FileMetadata functionality."""

    def test_init(self, sample_file_metadata: FileMetadata) -> None:
        """Test FileMetadata initialization."""
        metadata = sample_file_metadata

        assert metadata.file_path == "/test/example.py"
        assert metadata.size == 1024
        assert metadata.mtime == 1234567890.0
        assert metadata.language == "python"
        assert metadata.line_count == 50
        assert metadata.entity_count == 5
        assert metadata.complexity_score == 15.5
        assert metadata.semantic_summary == "A test module for demonstration"
        assert metadata.imports == ["import os", "import sys"]
        assert metadata.dependencies == ["requests", "numpy"]


class TestIndexQuery:
    """Test IndexQuery functionality."""

    def test_default_query(self) -> None:
        """Test default IndexQuery values."""
        query = IndexQuery()

        assert query.file_patterns is None
        assert query.languages is None
        assert query.min_size is None
        assert query.max_size is None
        assert query.entity_types is None
        assert query.semantic_query is None
        assert query.similarity_threshold == 0.7
        assert query.include_entities is True
        assert query.limit is None
        assert query.offset == 0

    def test_custom_query(self) -> None:
        """Test custom IndexQuery values."""
        query = IndexQuery(
            file_patterns=["*.py"],
            languages=["python"],
            min_size=100,
            max_size=10000,
            entity_types=["function", "class"],
            semantic_query="database operations",
            similarity_threshold=0.8,
            include_entities=False,
            limit=50,
            offset=10
        )

        assert query.file_patterns == ["*.py"]
        assert query.languages == ["python"]
        assert query.min_size == 100
        assert query.max_size == 10000
        assert query.entity_types == ["function", "class"]
        assert query.semantic_query == "database operations"
        assert query.similarity_threshold == 0.8
        assert query.include_entities is False
        assert query.limit == 50
        assert query.offset == 10


class TestMetadataIndex:
    """Test MetadataIndex functionality."""

    @pytest.mark.asyncio
    async def test_init(self, temp_dir: Path) -> None:
        """Test MetadataIndex initialization."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)

        assert index.db_path == db_path
        assert index._connection is None

        await index.initialize()

        assert index._connection is not None
        assert db_path.exists()

        await index.close()

    @pytest.mark.asyncio
    async def test_create_tables(self, temp_dir: Path) -> None:
        """Test database table creation."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        # Check that tables exist
        assert index._connection is not None
        conn = index._connection
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        assert "files" in tables
        assert "entities" in tables

        await index.close()

    @pytest.mark.asyncio
    async def test_add_file_metadata(self, temp_dir: Path, sample_file_metadata: FileMetadata) -> None:
        """Test adding file metadata."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        await index.add_file_metadata(sample_file_metadata)

        # Verify data was inserted
        assert index._connection is not None
        conn = index._connection
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM files WHERE file_path = ?",
                       (sample_file_metadata.file_path,))
        row = cursor.fetchone()

        assert row is not None
        assert row["file_path"] == sample_file_metadata.file_path
        assert row["size"] == sample_file_metadata.size
        assert row["language"] == sample_file_metadata.language

        await index.close()

    @pytest.mark.asyncio
    async def test_add_entity_metadata(self, temp_dir: Path, sample_entity_metadata: EntityMetadata) -> None:
        """Test adding entity metadata."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        await index.add_entity_metadata(sample_entity_metadata)

        # Verify data was inserted
        assert index._connection is not None
        conn = index._connection
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE entity_id = ?",
                       (sample_entity_metadata.entity_id,))
        row = cursor.fetchone()

        assert row is not None
        assert row["entity_id"] == sample_entity_metadata.entity_id
        assert row["name"] == sample_entity_metadata.name
        assert row["entity_type"] == sample_entity_metadata.entity_type

        await index.close()

    @pytest.mark.asyncio
    async def test_query_files(self, temp_dir: Path, sample_file_metadata: FileMetadata) -> None:
        """Test querying files."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        # Add test data
        await index.add_file_metadata(sample_file_metadata)

        # Query with language filter
        query = IndexQuery(languages=["python"])
        files = await index.query_files(query)

        assert len(files) == 1
        assert files[0].file_path == sample_file_metadata.file_path
        assert files[0].language == "python"

        # Query with size filter
        query = IndexQuery(min_size=500, max_size=2000)
        files = await index.query_files(query)

        assert len(files) == 1

        # Query with no matches
        query = IndexQuery(languages=["javascript"])
        files = await index.query_files(query)

        assert len(files) == 0

        await index.close()

    @pytest.mark.asyncio
    async def test_query_entities(self, temp_dir: Path, sample_entity_metadata: EntityMetadata) -> None:
        """Test querying entities."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        # Add test data
        await index.add_entity_metadata(sample_entity_metadata)

        # Query with entity type filter
        query = IndexQuery(entity_types=["function"])
        entities = await index.query_entities(query)

        assert len(entities) == 1
        assert entities[0].entity_id == sample_entity_metadata.entity_id
        assert entities[0].entity_type == "function"

        # Query with name filter
        query = IndexQuery(entity_names=["test_function"])
        entities = await index.query_entities(query)

        assert len(entities) == 1

        # Query with no matches
        query = IndexQuery(entity_types=["class"])
        entities = await index.query_entities(query)

        assert len(entities) == 0

        await index.close()

    @pytest.mark.asyncio
    async def test_get_stats(
        self,
        temp_dir: Path,
        sample_file_metadata: FileMetadata,
        sample_entity_metadata: EntityMetadata
    ) -> None:
        """Test getting index statistics."""
        db_path = temp_dir / "test.db"
        index = MetadataIndex(db_path)
        await index.initialize()

        # Add test data
        await index.add_file_metadata(sample_file_metadata)
        await index.add_entity_metadata(sample_entity_metadata)

        stats = await index.get_stats()

        assert stats.total_files == 1
        assert stats.total_entities == 1
        assert "python" in stats.languages
        assert "function" in stats.entity_types

        await index.close()


class TestMetadataIndexer:
    """Test MetadataIndexer functionality."""

    def test_init(self, config: SearchConfig) -> None:
        """Test MetadataIndexer initialization."""
        indexer = MetadataIndexer(config)

        assert indexer.config == config
        assert indexer.base_indexer is not None
        assert indexer.entity_extractor is not None
        assert indexer.semantic_embedding is not None
        assert indexer.metadata_index is not None
        assert indexer._initialized is False

    @pytest.mark.asyncio
    async def test_initialize(self, config: SearchConfig) -> None:
        """Test indexer initialization."""
        indexer = MetadataIndexer(config)

        with patch.object(indexer.metadata_index, 'initialize', new_callable=AsyncMock) as mock_init:
            await indexer.initialize()
            assert indexer._initialized is True
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_calculate_file_complexity(self, config: SearchConfig) -> None:
        """Test file complexity calculation."""
        indexer = MetadataIndexer(config)

        simple_code = "print('hello')"
        complex_code = """
def complex_function():
    if condition:
        for item in items:
            try:
                if another_condition:
                    while loop_condition:
                        process_item(item)
            except Exception:
                handle_error()
"""

        simple_complexity = indexer._calculate_file_complexity(
            simple_code, Language.PYTHON)
        complex_complexity = indexer._calculate_file_complexity(
            complex_code, Language.PYTHON)

        assert simple_complexity < complex_complexity
        assert simple_complexity >= 0
        assert complex_complexity >= 0

    @pytest.mark.asyncio
    async def test_extract_imports(self, config: SearchConfig) -> None:
        """Test import extraction."""
        indexer = MetadataIndexer(config)

        python_code = """
import os
import sys
from typing import List, Dict
from pathlib import Path
"""

        imports = indexer._extract_imports(python_code, Language.PYTHON)

        assert len(imports) >= 3
        assert any("import os" in imp for imp in imports)
        assert any("import sys" in imp for imp in imports)
        assert any("from typing import" in imp for imp in imports)

    @pytest.mark.asyncio
    async def test_create_entity_text(self, config: SearchConfig) -> None:
        """Test entity text creation for embeddings."""
        indexer = MetadataIndexer(config)

        from pysearch import CodeEntity
        entity = CodeEntity(
            id="test_entity",
            name="test_function",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            signature="def test_function(arg: str) -> bool:",
            docstring="Test function for demonstration",
            properties={"args": ["arg"], "returns": "bool"}
        )

        text = indexer._create_entity_text(entity)

        assert "test_function" in text
        assert "def test_function" in text
        assert "Test function for demonstration" in text
        assert "args: ['arg']" in text

    @pytest.mark.asyncio
    async def test_query_index(self, config: SearchConfig) -> None:
        """Test index querying."""
        indexer = MetadataIndexer(config)

        # Mock the metadata index
        mock_files = [
            FileMetadata(
                file_path="test.py",
                size=1000,
                mtime=123456789.0,
                language="python",
                line_count=50,
                entity_count=3,
                complexity_score=10.0
            )
        ]

        mock_entities = [
            EntityMetadata(
                entity_id="test_func",
                name="test_function",
                entity_type="function",
                file_path="test.py",
                start_line=10,
                end_line=15,
                complexity_score=5.0
            )
        ]

        with patch.object(indexer, '_initialized', True):
            with patch.object(indexer.metadata_index, 'query_files', new_callable=AsyncMock) as mock_query_files:
                with patch.object(indexer.metadata_index, 'query_entities', new_callable=AsyncMock) as mock_query_entities:
                    with patch.object(indexer.metadata_index, 'get_stats', new_callable=AsyncMock) as mock_get_stats:
                        mock_query_files.return_value = mock_files
                        mock_query_entities.return_value = mock_entities
                        mock_get_stats.return_value = IndexStats(
                            total_files=1, total_entities=1)

                        query = IndexQuery(include_entities=True)
                        results = await indexer.query_index(query)

                        assert "files" in results
                        assert "entities" in results
                        assert "stats" in results
                        assert len(results["files"]) == 1
                        assert len(results["entities"]) == 1
                        assert results["stats"]["total_files"] == 1

    @pytest.mark.asyncio
    async def test_close(self, config: SearchConfig) -> None:
        """Test indexer cleanup."""
        indexer = MetadataIndexer(config)

        with patch.object(indexer.metadata_index, 'close', new_callable=AsyncMock) as mock_close:
            await indexer.close()
            mock_close.assert_called_once()
