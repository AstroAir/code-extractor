"""Tests for pysearch.indexing.metadata.models module."""

from __future__ import annotations

import time

import pytest

from pysearch.indexing.metadata.models import (
    EntityMetadata,
    FileMetadata,
    IndexQuery,
    IndexStats,
)


class TestEntityMetadata:
    """Tests for EntityMetadata dataclass."""

    def test_creation(self):
        em = EntityMetadata(
            entity_id="e1",
            name="main",
            entity_type="function",
            file_path="test.py",
            start_line=1,
            end_line=10,
        )
        assert em.entity_id == "e1"
        assert em.name == "main"
        assert em.entity_type == "function"

    def test_defaults(self):
        em = EntityMetadata(
            entity_id="e1", name="x", entity_type="var",
            file_path="a.py", start_line=1, end_line=1,
        )
        assert em.signature is None
        assert em.docstring is None
        assert em.language == "unknown"
        assert em.scope is None
        assert em.complexity_score == 0.0
        assert em.semantic_embedding is None
        assert em.properties == {}

    def test_with_optional_fields(self):
        em = EntityMetadata(
            entity_id="e2", name="MyClass", entity_type="class",
            file_path="a.py", start_line=1, end_line=20,
            signature="class MyClass:", docstring="A class.",
            language="python", scope="module",
            complexity_score=5.0, semantic_embedding=[0.1, 0.2],
            properties={"decorators": ["dataclass"]},
        )
        assert em.signature == "class MyClass:"
        assert em.docstring == "A class."
        assert em.language == "python"
        assert em.scope == "module"
        assert em.complexity_score == 5.0
        assert em.semantic_embedding == [0.1, 0.2]
        assert em.properties == {"decorators": ["dataclass"]}

    def test_last_updated_default(self):
        before = time.time()
        em = EntityMetadata(
            entity_id="e1", name="x", entity_type="var",
            file_path="a.py", start_line=1, end_line=1,
        )
        after = time.time()
        assert before <= em.last_updated <= after


class TestFileMetadata:
    """Tests for FileMetadata dataclass."""

    def test_creation(self):
        fm = FileMetadata(
            file_path="test.py",
            language="python",
            size=1024,
            mtime=1.0,
        )
        assert fm.file_path == "test.py"
        assert fm.language == "python"
        assert fm.size == 1024
        assert fm.mtime == 1.0

    def test_defaults(self):
        fm = FileMetadata(file_path="a.py", size=0, mtime=0.0)
        assert fm.sha1 is None
        assert fm.language == "unknown"
        assert fm.line_count == 0
        assert fm.entity_count == 0
        assert fm.complexity_score == 0.0
        assert fm.semantic_summary is None
        assert fm.imports == []
        assert fm.exports == []
        assert fm.dependencies == []
        assert fm.access_count == 0

    def test_with_optional_fields(self):
        fm = FileMetadata(
            file_path="b.py", size=2048, mtime=1.0,
            sha1="abc123", language="python", line_count=100,
            entity_count=5, complexity_score=10.0,
            semantic_summary="A module for data processing",
            imports=["import os"], exports=["main"],
            dependencies=["os", "sys"],
        )
        assert fm.sha1 == "abc123"
        assert fm.line_count == 100
        assert fm.entity_count == 5
        assert fm.imports == ["import os"]
        assert fm.exports == ["main"]
        assert fm.dependencies == ["os", "sys"]


class TestIndexQuery:
    """Tests for IndexQuery dataclass."""

    def test_defaults(self):
        q = IndexQuery()
        assert q.file_patterns is None
        assert q.languages is None
        assert q.min_size is None
        assert q.max_size is None
        assert q.min_lines is None
        assert q.max_lines is None
        assert q.modified_after is None
        assert q.modified_before is None
        assert q.entity_types is None
        assert q.entity_names is None
        assert q.has_docstring is None
        assert q.min_complexity is None
        assert q.max_complexity is None
        assert q.semantic_query is None
        assert q.similarity_threshold == 0.7
        assert q.include_entities is True
        assert q.include_file_content is False
        assert q.limit is None
        assert q.offset == 0

    def test_with_all_filters(self):
        q = IndexQuery(
            file_patterns=["src/*.py"],
            languages=["python"],
            min_size=100,
            max_size=10000,
            min_lines=10,
            max_lines=500,
            modified_after=1000.0,
            modified_before=9999.0,
            entity_types=["function", "class"],
            entity_names=["main"],
            has_docstring=True,
            min_complexity=1.0,
            max_complexity=50.0,
            semantic_query="database",
            similarity_threshold=0.8,
            include_entities=False,
            include_file_content=True,
            limit=25,
            offset=10,
        )
        assert q.file_patterns == ["src/*.py"]
        assert q.languages == ["python"]
        assert q.min_size == 100
        assert q.max_size == 10000
        assert q.entity_types == ["function", "class"]
        assert q.has_docstring is True
        assert q.similarity_threshold == 0.8
        assert q.limit == 25
        assert q.offset == 10


class TestIndexStats:
    """Tests for IndexStats dataclass."""

    def test_defaults(self):
        s = IndexStats()
        assert s.total_files == 0
        assert s.total_entities == 0
        assert s.languages == {}
        assert s.entity_types == {}
        assert s.avg_file_size == 0.0
        assert s.avg_entities_per_file == 0.0
        assert s.index_size_mb == 0.0
        assert s.last_build_time == 0.0
        assert s.build_duration == 0.0

    def test_with_values(self):
        s = IndexStats(
            total_files=50,
            total_entities=200,
            languages={"python": 30, "javascript": 20},
            entity_types={"function": 100, "class": 50, "variable": 50},
            avg_file_size=2048.0,
            avg_entities_per_file=4.0,
            index_size_mb=1.5,
            last_build_time=1000.0,
            build_duration=5.5,
        )
        assert s.total_files == 50
        assert s.total_entities == 200
        assert s.languages["python"] == 30
        assert s.entity_types["function"] == 100
        assert s.build_duration == 5.5
