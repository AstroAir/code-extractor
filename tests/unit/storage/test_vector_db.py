"""Tests for pysearch.storage.vector_db module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.storage.vector_db import (
    EmbeddingConfig,
    VectorDatabase,
    VectorIndexManager,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig dataclass."""

    def test_defaults(self):
        cfg = EmbeddingConfig()
        assert cfg.dimensions > 0
        assert cfg.provider == "openai"

    def test_custom(self):
        cfg = EmbeddingConfig(dimensions=768, provider="huggingface")
        assert cfg.dimensions == 768
        assert cfg.provider == "huggingface"


class TestVectorDatabase:
    """Tests for VectorDatabase abstract base class."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            VectorDatabase()


def _lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _lancedb_available(), reason="LanceDB not installed")
class TestVectorIndexManager:
    """Tests for VectorIndexManager class."""

    def test_init(self, tmp_path: Path):
        cfg = EmbeddingConfig()
        mgr = VectorIndexManager(tmp_path, cfg)
        assert mgr is not None

    @pytest.mark.asyncio
    async def test_search_not_initialized(self, tmp_path: Path):
        cfg = EmbeddingConfig()
        mgr = VectorIndexManager(tmp_path, cfg)
        results = await mgr.search("test query", "collection")
        assert isinstance(results, list)
        assert len(results) == 0
