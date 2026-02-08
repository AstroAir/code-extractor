"""Tests for pysearch.indexing.metadata.models module."""

from __future__ import annotations

import time

import pytest

from pysearch.indexing.metadata.models import EntityMetadata, FileMetadata


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
