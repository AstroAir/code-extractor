"""Tests for pysearch.analysis.graphrag.core module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.analysis.graphrag.core import (
    EntityExtractor,
    RelationshipMapper,
)
from pysearch.core.types import EntityType, Language


class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    def test_init(self):
        extractor = EntityExtractor()
        assert extractor is not None
        assert isinstance(extractor.language_extractors, dict)
        assert Language.PYTHON in extractor.language_extractors

    @pytest.mark.asyncio
    async def test_extract_from_python_file(self, tmp_path: Path):
        py_file = tmp_path / "test.py"
        py_file.write_text("def hello():\n    pass\n\nclass World:\n    pass\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        assert isinstance(entities, list)
        names = [e.name for e in entities]
        assert "hello" in names
        assert "World" in names

    @pytest.mark.asyncio
    async def test_extract_empty_file(self, tmp_path: Path):
        py_file = tmp_path / "empty.py"
        py_file.write_text("", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        assert entities == []

    @pytest.mark.asyncio
    async def test_extract_functions(self, tmp_path: Path):
        py_file = tmp_path / "funcs.py"
        py_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        funcs = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(funcs) == 2

    @pytest.mark.asyncio
    async def test_extract_classes(self, tmp_path: Path):
        py_file = tmp_path / "cls.py"
        py_file.write_text("class MyClass:\n    def method(self):\n        pass\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 1

    @pytest.mark.asyncio
    async def test_extract_nonexistent_file(self, tmp_path: Path):
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(tmp_path / "nonexistent.py")
        assert entities == []


class TestRelationshipMapper:
    """Tests for RelationshipMapper class."""

    def test_init(self):
        mapper = RelationshipMapper()
        assert mapper is not None

    @pytest.mark.asyncio
    async def test_map_empty(self):
        mapper = RelationshipMapper()
        relationships = await mapper.map_relationships([], {})
        assert relationships == []
