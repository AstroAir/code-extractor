"""Tests for pysearch.indexing.metadata.database module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.indexing.metadata.database import MetadataIndex
from pysearch.indexing.metadata.models import (
    EntityMetadata,
    FileMetadata,
    IndexQuery,
    IndexStats,
)


def _make_file_metadata(path: str = "test.py", **kwargs) -> FileMetadata:
    defaults = dict(
        file_path=path, size=1024, mtime=time.time(), language="python",
        line_count=50, entity_count=3, complexity_score=5.0,
    )
    defaults.update(kwargs)
    return FileMetadata(**defaults)


def _make_entity_metadata(entity_id: str = "e1", file_path: str = "test.py", **kwargs) -> EntityMetadata:
    defaults = dict(
        entity_id=entity_id, name="main", entity_type="function",
        file_path=file_path, start_line=1, end_line=10,
        signature="def main():", language="python",
    )
    defaults.update(kwargs)
    return EntityMetadata(**defaults)


class TestMetadataIndex:
    """Tests for MetadataIndex class."""

    def test_init(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        assert idx is not None
        assert idx.db_path == tmp_path / "metadata.db"

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        assert idx._connection is not None
        await idx.close()

    @pytest.mark.asyncio
    async def test_close(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.close()
        assert idx._connection is None

    @pytest.mark.asyncio
    async def test_close_without_init(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.close()  # Should not raise

    # --- add / query file metadata ---

    @pytest.mark.asyncio
    async def test_add_file_metadata(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        fm = _make_file_metadata("src/app.py")
        await idx.add_file_metadata(fm)
        files = await idx.query_files(IndexQuery())
        assert len(files) == 1
        assert files[0].file_path == "src/app.py"
        assert files[0].language == "python"
        await idx.close()

    @pytest.mark.asyncio
    async def test_add_file_metadata_upsert(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("a.py", size=100))
        await idx.add_file_metadata(_make_file_metadata("a.py", size=200))
        files = await idx.query_files(IndexQuery())
        assert len(files) == 1
        assert files[0].size == 200
        await idx.close()

    @pytest.mark.asyncio
    async def test_add_file_metadata_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.add_file_metadata(_make_file_metadata())  # Should not raise

    @pytest.mark.asyncio
    async def test_query_files_language_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("a.py", language="python"))
        await idx.add_file_metadata(_make_file_metadata("b.js", language="javascript"))
        files = await idx.query_files(IndexQuery(languages=["python"]))
        assert len(files) == 1
        assert files[0].file_path == "a.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_size_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("small.py", size=100))
        await idx.add_file_metadata(_make_file_metadata("big.py", size=10000))
        files = await idx.query_files(IndexQuery(min_size=5000))
        assert len(files) == 1
        assert files[0].file_path == "big.py"
        files2 = await idx.query_files(IndexQuery(max_size=500))
        assert len(files2) == 1
        assert files2[0].file_path == "small.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_lines_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("short.py", line_count=10))
        await idx.add_file_metadata(_make_file_metadata("long.py", line_count=500))
        files = await idx.query_files(IndexQuery(min_lines=100))
        assert len(files) == 1
        assert files[0].file_path == "long.py"
        files2 = await idx.query_files(IndexQuery(max_lines=50))
        assert len(files2) == 1
        assert files2[0].file_path == "short.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_modified_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("old.py", mtime=1000.0))
        await idx.add_file_metadata(_make_file_metadata("new.py", mtime=9999.0))
        files = await idx.query_files(IndexQuery(modified_after=5000.0))
        assert len(files) == 1
        assert files[0].file_path == "new.py"
        files2 = await idx.query_files(IndexQuery(modified_before=5000.0))
        assert len(files2) == 1
        assert files2[0].file_path == "old.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_file_patterns(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("src/app.py"))
        await idx.add_file_metadata(_make_file_metadata("tests/test_app.py"))
        files = await idx.query_files(IndexQuery(file_patterns=["src/*"]))
        assert len(files) == 1
        assert files[0].file_path == "src/app.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_limit_offset(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        for i in range(5):
            await idx.add_file_metadata(_make_file_metadata(f"f{i}.py"))
        files = await idx.query_files(IndexQuery(limit=2))
        assert len(files) == 2
        files2 = await idx.query_files(IndexQuery(limit=2, offset=3))
        assert len(files2) == 2
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_files_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        files = await idx.query_files(IndexQuery())
        assert files == []

    # --- add / query entity metadata ---

    @pytest.mark.asyncio
    async def test_add_entity_metadata(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("test.py"))
        em = _make_entity_metadata("e1", "test.py")
        await idx.add_entity_metadata(em)
        entities = await idx.query_entities(IndexQuery())
        assert len(entities) == 1
        assert entities[0].name == "main"
        await idx.close()

    @pytest.mark.asyncio
    async def test_add_entity_metadata_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.add_entity_metadata(_make_entity_metadata())  # Should not raise

    @pytest.mark.asyncio
    async def test_query_entities_type_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1", entity_type="function"))
        await idx.add_entity_metadata(_make_entity_metadata("e2", entity_type="class", name="Foo"))
        entities = await idx.query_entities(IndexQuery(entity_types=["class"]))
        assert len(entities) == 1
        assert entities[0].entity_type == "class"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_entities_name_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1", name="connect_db"))
        await idx.add_entity_metadata(_make_entity_metadata("e2", name="process_data"))
        entities = await idx.query_entities(IndexQuery(entity_names=["connect"]))
        assert len(entities) == 1
        assert "connect" in entities[0].name
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_entities_docstring_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1", docstring="Does something"))
        await idx.add_entity_metadata(_make_entity_metadata("e2", name="no_doc"))
        entities = await idx.query_entities(IndexQuery(has_docstring=True))
        assert len(entities) == 1
        assert entities[0].entity_id == "e1"
        entities2 = await idx.query_entities(IndexQuery(has_docstring=False))
        assert len(entities2) == 1
        assert entities2[0].entity_id == "e2"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_entities_complexity_filter(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1", complexity_score=2.0))
        await idx.add_entity_metadata(_make_entity_metadata("e2", complexity_score=20.0, name="complex"))
        entities = await idx.query_entities(IndexQuery(min_complexity=10.0))
        assert len(entities) == 1
        assert entities[0].entity_id == "e2"
        entities2 = await idx.query_entities(IndexQuery(max_complexity=5.0))
        assert len(entities2) == 1
        assert entities2[0].entity_id == "e1"
        await idx.close()

    @pytest.mark.asyncio
    async def test_query_entities_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        entities = await idx.query_entities(IndexQuery())
        assert entities == []

    # --- get_stats ---

    @pytest.mark.asyncio
    async def test_get_stats_empty(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        stats = await idx.get_stats()
        assert isinstance(stats, IndexStats)
        assert stats.total_files == 0
        assert stats.total_entities == 0
        await idx.close()

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("a.py"))
        await idx.add_entity_metadata(_make_entity_metadata("e1"))
        stats = await idx.get_stats()
        assert stats.total_files == 1
        assert stats.total_entities == 1
        assert "python" in stats.languages
        assert "function" in stats.entity_types
        await idx.close()

    @pytest.mark.asyncio
    async def test_get_stats_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        stats = await idx.get_stats()
        assert stats.total_files == 0

    # --- delete operations ---

    @pytest.mark.asyncio
    async def test_delete_file_metadata(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("a.py"))
        await idx.add_entity_metadata(_make_entity_metadata("e1", "a.py"))
        await idx.delete_file_metadata("a.py")
        files = await idx.query_files(IndexQuery())
        assert len(files) == 0
        entities = await idx.query_entities(IndexQuery())
        assert len(entities) == 0  # cascade delete
        await idx.close()

    @pytest.mark.asyncio
    async def test_delete_file_metadata_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.delete_file_metadata("a.py")  # Should not raise

    @pytest.mark.asyncio
    async def test_delete_entity_metadata(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1"))
        await idx.add_entity_metadata(_make_entity_metadata("e2", name="other"))
        await idx.delete_entity_metadata("e1")
        entities = await idx.query_entities(IndexQuery())
        assert len(entities) == 1
        assert entities[0].entity_id == "e2"
        await idx.close()

    @pytest.mark.asyncio
    async def test_delete_entity_metadata_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.delete_entity_metadata("e1")  # Should not raise

    @pytest.mark.asyncio
    async def test_delete_entities_by_file(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_entity_metadata(_make_entity_metadata("e1", "a.py"))
        await idx.add_entity_metadata(_make_entity_metadata("e2", "a.py", name="other"))
        await idx.add_entity_metadata(_make_entity_metadata("e3", "b.py", name="keep"))
        deleted = await idx.delete_entities_by_file("a.py")
        assert deleted == 2
        entities = await idx.query_entities(IndexQuery())
        assert len(entities) == 1
        assert entities[0].file_path == "b.py"
        await idx.close()

    @pytest.mark.asyncio
    async def test_delete_entities_by_file_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        deleted = await idx.delete_entities_by_file("a.py")
        assert deleted == 0

    # --- file_exists ---

    @pytest.mark.asyncio
    async def test_file_exists_true(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        await idx.add_file_metadata(_make_file_metadata("a.py"))
        assert await idx.file_exists("a.py") is True
        await idx.close()

    @pytest.mark.asyncio
    async def test_file_exists_false(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        await idx.initialize()
        assert await idx.file_exists("nonexistent.py") is False
        await idx.close()

    @pytest.mark.asyncio
    async def test_file_exists_no_connection(self, tmp_path: Path):
        idx = MetadataIndex(tmp_path / "metadata.db")
        assert await idx.file_exists("a.py") is False
