"""Tests for pysearch.indexing.advanced.chunking module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.indexing.advanced.chunking import (
    ChunkingConfig,
    ChunkingEngine,
    ChunkingStrategy,
)


class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_values(self):
        assert ChunkingStrategy.BASIC == "basic"
        assert ChunkingStrategy.STRUCTURAL == "structural"
        assert ChunkingStrategy.SEMANTIC == "semantic"
        assert ChunkingStrategy.HYBRID == "hybrid"


class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.max_chunk_size > 0
        assert cfg.overlap_size >= 0
        assert cfg.strategy == ChunkingStrategy.HYBRID

    def test_custom(self):
        cfg = ChunkingConfig(max_chunk_size=500, overlap_size=50)
        assert cfg.max_chunk_size == 500
        assert cfg.overlap_size == 50


class TestChunkingEngine:
    """Tests for ChunkingEngine class."""

    def test_init(self):
        engine = ChunkingEngine()
        assert engine is not None

    def test_init_with_config(self):
        cfg = ChunkingConfig(max_chunk_size=1000)
        engine = ChunkingEngine(config=cfg)
        assert engine.config.max_chunk_size == 1000

    @pytest.mark.asyncio
    async def test_chunk_empty_file(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        chunks = await engine.chunk_file(f)
        assert isinstance(chunks, list)

    @pytest.mark.asyncio
    async def test_chunk_simple_file(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "funcs.py"
        f.write_text("def hello():\n    pass\n\ndef world():\n    pass\n", encoding="utf-8")
        chunks = await engine.chunk_file(f)
        assert isinstance(chunks, list)
