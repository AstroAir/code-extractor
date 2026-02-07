"""Tests for pysearch.indexing.advanced.integration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.types import Language
from pysearch.indexing.advanced.integration import (
    IndexSearchEngine,
    IndexSearchResult,
    IndexingOrchestrator,
)


class TestIndexSearchResult:
    """Tests for IndexSearchResult dataclass."""

    def test_creation(self):
        r = IndexSearchResult(
            path="test.py",
            content="def hello(): pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            score=0.95,
        )
        assert r.path == "test.py"
        assert r.language == Language.PYTHON
        assert r.score == 0.95

    def test_defaults(self):
        r = IndexSearchResult(
            path="x.py", content="", start_line=1, end_line=1,
            language=Language.UNKNOWN, score=0.0,
        )
        assert r.score == 0.0
        assert r.entity_name is None


class TestIndexSearchEngine:
    """Tests for IndexSearchEngine class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = IndexSearchEngine(cfg)
        assert engine is not None

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = IndexSearchEngine(cfg)
        results = await engine.search("test")
        assert isinstance(results, list)


class TestIndexingOrchestrator:
    """Tests for IndexingOrchestrator class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        orch = IndexingOrchestrator(cfg)
        assert orch is not None
