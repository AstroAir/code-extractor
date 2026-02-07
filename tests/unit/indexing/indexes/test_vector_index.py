"""Tests for pysearch.indexing.indexes.vector_index module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from pysearch.indexing.indexes.vector_index import VectorIndex


def _lancedb_available() -> bool:
    try:
        import lancedb  # noqa: F401
        return True
    except ImportError:
        return False


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.resolve_cache_dir.return_value = tmp_path / "cache"
    cfg.chunk_size = 1000
    cfg.embedding_provider = "huggingface"
    cfg.embedding_model = "all-MiniLM-L6-v2"
    cfg.embedding_batch_size = 100
    cfg.openai_api_key = None
    cfg.vector_db_provider = "lancedb"
    return cfg


class TestVectorIndex:
    """Tests for VectorIndex class."""

    @pytest.mark.skipif(
        not _lancedb_available(),
        reason="LanceDB not installed",
    )
    def test_init(self, tmp_path: Path):
        idx = VectorIndex(config=_make_config(tmp_path))
        assert idx is not None

    @pytest.mark.skipif(
        not _lancedb_available(),
        reason="LanceDB not installed",
    )
    def test_artifact_id(self, tmp_path: Path):
        idx = VectorIndex(config=_make_config(tmp_path))
        assert idx.artifact_id == "enhanced_vectors"

    @pytest.mark.skipif(
        not _lancedb_available(),
        reason="LanceDB not installed",
    )
    def test_relative_expected_time(self, tmp_path: Path):
        idx = VectorIndex(config=_make_config(tmp_path))
        assert idx.relative_expected_time == 3.0
