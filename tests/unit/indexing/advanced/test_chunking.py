"""Tests for pysearch.indexing.advanced.chunking module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pysearch.core.types import Language
from pysearch.indexing.advanced.chunking import (
    BasicChunker,
    ChunkingConfig,
    ChunkingEngine,
    ChunkingStrategy,
    ChunkingStrategyBase,
    ChunkMetadata,
    HybridChunker,
    MetadataCodeChunk,
    SemanticChunker,
    StructuralChunker,
)


# ---------------------------------------------------------------------------
# ChunkingStrategy enum
# ---------------------------------------------------------------------------
class TestChunkingStrategy:
    """Tests for ChunkingStrategy enum."""

    def test_values(self):
        assert ChunkingStrategy.BASIC == "basic"
        assert ChunkingStrategy.STRUCTURAL == "structural"
        assert ChunkingStrategy.SEMANTIC == "semantic"
        assert ChunkingStrategy.HYBRID == "hybrid"

    def test_is_str_enum(self):
        assert isinstance(ChunkingStrategy.BASIC, str)


# ---------------------------------------------------------------------------
# ChunkingConfig dataclass
# ---------------------------------------------------------------------------
class TestChunkingConfig:
    """Tests for ChunkingConfig dataclass."""

    def test_defaults(self):
        cfg = ChunkingConfig()
        assert cfg.max_chunk_size == 1000
        assert cfg.min_chunk_size == 50
        assert cfg.overlap_size == 100
        assert cfg.respect_boundaries is True
        assert cfg.include_context is True
        assert cfg.quality_threshold == 0.7
        assert cfg.max_chunks_per_file == 100
        assert cfg.strategy == ChunkingStrategy.HYBRID

    def test_custom(self):
        cfg = ChunkingConfig(max_chunk_size=500, overlap_size=50)
        assert cfg.max_chunk_size == 500
        assert cfg.overlap_size == 50


# ---------------------------------------------------------------------------
# ChunkMetadata dataclass
# ---------------------------------------------------------------------------
class TestChunkMetadata:
    """Tests for ChunkMetadata dataclass."""

    def test_creation(self):
        meta = ChunkMetadata(
            chunk_id="file.py:1:10",
            quality_score=0.8,
            boundary_type="function",
            contains_entities=["foo"],
            dependencies=["os"],
            semantic_tags=["utility"],
            complexity_score=3.0,
        )
        assert meta.chunk_id == "file.py:1:10"
        assert meta.quality_score == 0.8
        assert meta.boundary_type == "function"
        assert meta.contains_entities == ["foo"]
        assert meta.dependencies == ["os"]


# ---------------------------------------------------------------------------
# MetadataCodeChunk dataclass
# ---------------------------------------------------------------------------
class TestMetadataCodeChunk:
    """Tests for MetadataCodeChunk dataclass."""

    def test_defaults(self):
        chunk = MetadataCodeChunk(
            content="pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        assert chunk.chunk_id == ""
        assert chunk.metadata is None
        assert chunk.overlap_with == []
        assert chunk.quality_score == 0.0

    def test_with_metadata(self):
        meta = ChunkMetadata(
            chunk_id="x",
            quality_score=0.9,
            boundary_type="class",
            contains_entities=[],
            dependencies=[],
            semantic_tags=[],
            complexity_score=0.0,
        )
        chunk = MetadataCodeChunk(
            content="class Foo: pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="class",
            chunk_id="file:1:1",
            metadata=meta,
            quality_score=0.9,
        )
        assert chunk.metadata is meta
        assert chunk.quality_score == 0.9


# ---------------------------------------------------------------------------
# BasicChunker
# ---------------------------------------------------------------------------
class TestBasicChunker:
    """Tests for BasicChunker class."""

    async def test_chunk_small_content(self):
        cfg = ChunkingConfig(max_chunk_size=5000)
        chunker = BasicChunker(cfg)
        content = "line1\nline2\nline3\n"
        chunks = [c async for c in chunker.chunk_content(content, Language.PYTHON, "f.py")]
        assert len(chunks) >= 1
        assert all(isinstance(c, MetadataCodeChunk) for c in chunks)
        assert chunks[0].chunk_type == "basic"
        assert chunks[0].quality_score == 0.3

    async def test_chunk_large_content_splits(self):
        cfg = ChunkingConfig(max_chunk_size=50)
        chunker = BasicChunker(cfg)
        # Create content that exceeds max_chunk_size
        content = "\n".join(f"line_{i} = {i}" for i in range(50))
        chunks = [c async for c in chunker.chunk_content(content, Language.PYTHON, "big.py")]
        assert len(chunks) > 1

    async def test_chunk_empty_content(self):
        cfg = ChunkingConfig()
        chunker = BasicChunker(cfg)
        chunks = [c async for c in chunker.chunk_content("", Language.PYTHON)]
        assert chunks == []

    async def test_chunk_whitespace_only_content(self):
        cfg = ChunkingConfig()
        chunker = BasicChunker(cfg)
        chunks = [c async for c in chunker.chunk_content("   \n  \n  ", Language.PYTHON)]
        assert chunks == []

    async def test_chunk_id_format(self):
        cfg = ChunkingConfig(max_chunk_size=5000)
        chunker = BasicChunker(cfg)
        content = "x = 1\ny = 2\n"
        chunks = [c async for c in chunker.chunk_content(content, Language.PYTHON, "test.py")]
        assert len(chunks) >= 1
        assert chunks[0].chunk_id.startswith("test.py:")


# ---------------------------------------------------------------------------
# ChunkingStrategyBase.calculate_chunk_quality
# ---------------------------------------------------------------------------
class TestCalculateChunkQuality:
    """Tests for ChunkingStrategyBase.calculate_chunk_quality."""

    def _make_chunker(self, max_chunk_size=1000):
        cfg = ChunkingConfig(max_chunk_size=max_chunk_size)
        return StructuralChunker(cfg)

    def test_function_type_high_quality(self):
        chunker = self._make_chunker(max_chunk_size=1000)
        chunk = MetadataCodeChunk(
            content="x" * 500,
            start_line=1,
            end_line=10,
            language=Language.PYTHON,
            chunk_type="function",
            entity_name="foo",
            dependencies=["os"],
        )
        quality = chunker.calculate_chunk_quality(chunk)
        # 0.3 (size) + 0.4 (function) + 0.2 (entity_name) + 0.1 (deps) = 1.0
        assert quality == pytest.approx(1.0)

    def test_block_type_medium_quality(self):
        chunker = self._make_chunker(max_chunk_size=1000)
        chunk = MetadataCodeChunk(
            content="x" * 500,
            start_line=1,
            end_line=5,
            language=Language.PYTHON,
            chunk_type="block",
        )
        quality = chunker.calculate_chunk_quality(chunk)
        # 0.3 (size) + 0.2 (block) = 0.5
        assert quality == pytest.approx(0.5)

    def test_oversized_chunk_reduced_quality(self):
        chunker = self._make_chunker(max_chunk_size=100)
        chunk = MetadataCodeChunk(
            content="x" * 200,
            start_line=1,
            end_line=5,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        quality = chunker.calculate_chunk_quality(chunk)
        # size_ratio=2.0 -> max(0.0, 0.3 - 1.0*0.2) = 0.1
        assert quality == pytest.approx(0.1)

    def test_very_small_chunk_no_size_bonus(self):
        chunker = self._make_chunker(max_chunk_size=1000)
        chunk = MetadataCodeChunk(
            content="x",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        quality = chunker.calculate_chunk_quality(chunk)
        # size_ratio ~0.001 -> no size bonus, no type bonus
        assert quality == 0.0

    def test_quality_capped_at_one(self):
        chunker = self._make_chunker(max_chunk_size=1000)
        chunk = MetadataCodeChunk(
            content="x" * 500,
            start_line=1,
            end_line=10,
            language=Language.PYTHON,
            chunk_type="class",
            entity_name="Bar",
            dependencies=["sys", "os"],
        )
        quality = chunker.calculate_chunk_quality(chunk)
        assert quality <= 1.0


# ---------------------------------------------------------------------------
# StructuralChunker
# ---------------------------------------------------------------------------
class TestStructuralChunker:
    """Tests for StructuralChunker class."""

    def test_build_language_config(self):
        cfg = ChunkingConfig(max_chunk_size=2000, min_chunk_size=100, respect_boundaries=False)
        chunker = StructuralChunker(cfg)
        lc = chunker._build_language_config()
        assert lc.max_chunk_size == 2000
        assert lc.min_chunk_size == 100
        assert lc.respect_boundaries is False

    async def test_basic_structural_chunk_python(self):
        cfg = ChunkingConfig(max_chunk_size=60, respect_boundaries=True)
        chunker = StructuralChunker(cfg)
        content = "import os\n\ndef foo():\n    pass\n\ndef bar():\n    pass\n"
        chunks = [
            c async for c in chunker._basic_structural_chunk(
                content, Language.PYTHON, "test.py"
            )
        ]
        assert len(chunks) >= 1
        assert all(c.chunk_type == "structural" for c in chunks)
        assert all(c.quality_score == 0.5 for c in chunks)

    async def test_basic_structural_chunk_javascript(self):
        cfg = ChunkingConfig(max_chunk_size=80)
        chunker = StructuralChunker(cfg)
        content = "function hello() {\n  return 1;\n}\n\nclass Foo {\n}\n"
        chunks = [
            c async for c in chunker._basic_structural_chunk(
                content, Language.JAVASCRIPT, "test.js"
            )
        ]
        assert len(chunks) >= 1

    async def test_basic_structural_chunk_java(self):
        cfg = ChunkingConfig(max_chunk_size=80)
        chunker = StructuralChunker(cfg)
        content = "public class Foo {\n  private int x;\n}\n"
        chunks = [
            c async for c in chunker._basic_structural_chunk(
                content, Language.JAVA, "Test.java"
            )
        ]
        assert len(chunks) >= 1

    async def test_basic_structural_chunk_default_language(self):
        cfg = ChunkingConfig(max_chunk_size=60)
        chunker = StructuralChunker(cfg)
        content = "fn main() {\n  println!(\"hello\");\n}\n"
        chunks = [
            c async for c in chunker._basic_structural_chunk(
                content, Language.RUST, "main.rs"
            )
        ]
        assert len(chunks) >= 1

    async def test_chunk_content_fallback_when_no_processor(self):
        cfg = ChunkingConfig(max_chunk_size=5000)
        chunker = StructuralChunker(cfg)
        content = "x = 1\ny = 2\n"
        with patch(
            "pysearch.indexing.advanced.chunking.language_registry"
        ) as mock_registry:
            mock_registry.get_processor.return_value = None
            chunks = [
                c async for c in chunker.chunk_content(
                    content, Language.PYTHON, "test.py"
                )
            ]
        assert len(chunks) >= 1
        assert chunks[0].chunk_type == "structural"


# ---------------------------------------------------------------------------
# SemanticChunker
# ---------------------------------------------------------------------------
class TestSemanticChunker:
    """Tests for SemanticChunker class."""

    async def test_calculate_semantic_similarity_identical(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        sim = await chunker._calculate_semantic_similarity("def foo(): pass", "def foo(): pass")
        assert sim == pytest.approx(1.0)

    async def test_calculate_semantic_similarity_different(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        sim = await chunker._calculate_semantic_similarity(
            "import os\nimport sys", "class Foo:\n    def bar(self): pass"
        )
        assert 0.0 <= sim <= 1.0

    async def test_calculate_semantic_similarity_empty(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        assert await chunker._calculate_semantic_similarity("", "hello") == 0.0
        assert await chunker._calculate_semantic_similarity("hello", "") == 0.0
        assert await chunker._calculate_semantic_similarity("", "") == 0.0

    async def test_group_chunks_semantically_empty(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        groups = await chunker._group_chunks_semantically([])
        assert groups == []

    async def test_group_chunks_semantically_single(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        chunk = MetadataCodeChunk(
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="function",
        )
        groups = await chunker._group_chunks_semantically([chunk])
        assert len(groups) == 1
        assert groups[0] == [chunk]

    async def test_merge_chunks_empty(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        result = await chunker._merge_chunks([], "file.py")
        assert result is None

    async def test_merge_chunks_single(self):
        cfg = ChunkingConfig()
        chunker = SemanticChunker(cfg)
        chunk = MetadataCodeChunk(
            content="x = 1",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        result = await chunker._merge_chunks([chunk], "file.py")
        assert result is not None
        assert result.content == "x = 1"

    async def test_merge_chunks_too_large_returns_first(self):
        cfg = ChunkingConfig(max_chunk_size=10)
        chunker = SemanticChunker(cfg)
        c1 = MetadataCodeChunk(
            content="a" * 10,
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        c2 = MetadataCodeChunk(
            content="b" * 10,
            start_line=2,
            end_line=2,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        result = await chunker._merge_chunks([c1, c2], "file.py")
        # Merged size = 21 > 10 * 1.5 = 15 -> returns first chunk
        assert result is c1

    async def test_merge_chunks_preserves_metadata(self):
        cfg = ChunkingConfig(max_chunk_size=5000)
        chunker = SemanticChunker(cfg)
        c1 = MetadataCodeChunk(
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="function",
            entity_name="foo",
            dependencies=["os"],
            complexity_score=2.0,
        )
        c2 = MetadataCodeChunk(
            content="def bar(): pass",
            start_line=3,
            end_line=3,
            language=Language.PYTHON,
            chunk_type="function",
            entity_name="bar",
            dependencies=["sys"],
            complexity_score=4.0,
        )
        result = await chunker._merge_chunks([c1, c2], "file.py")
        assert result is not None
        assert result.start_line == 1
        assert result.end_line == 3
        assert result.chunk_type == "semantic_group"
        assert set(result.dependencies) == {"os", "sys"}
        assert result.complexity_score == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# HybridChunker
# ---------------------------------------------------------------------------
class TestHybridChunker:
    """Tests for HybridChunker class."""

    def test_init(self):
        cfg = ChunkingConfig()
        chunker = HybridChunker(cfg)
        assert isinstance(chunker.structural_chunker, StructuralChunker)
        assert isinstance(chunker.semantic_chunker, SemanticChunker)

    async def test_optimize_chunks_high_quality_unchanged(self):
        cfg = ChunkingConfig(quality_threshold=0.5)
        chunker = HybridChunker(cfg)
        chunk = MetadataCodeChunk(
            content="x" * 100,
            start_line=1,
            end_line=5,
            language=Language.PYTHON,
            chunk_type="function",
            quality_score=0.9,
        )
        result = await chunker.optimize_chunks([chunk])
        assert len(result) == 1
        assert result[0] is chunk

    async def test_optimize_chunks_low_quality_attempted(self):
        cfg = ChunkingConfig(quality_threshold=0.8)
        chunker = HybridChunker(cfg)
        chunk = MetadataCodeChunk(
            content="x",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
            quality_score=0.1,
        )
        result = await chunker.optimize_chunks([chunk])
        assert len(result) == 1
        # _improve_chunk returns the same chunk for now
        assert result[0] is chunk

    async def test_improve_chunk_returns_original(self):
        cfg = ChunkingConfig()
        chunker = HybridChunker(cfg)
        chunk = MetadataCodeChunk(
            content="pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
        )
        improved = await chunker._improve_chunk(chunk)
        assert improved is chunk


# ---------------------------------------------------------------------------
# ChunkingEngine
# ---------------------------------------------------------------------------
class TestChunkingEngine:
    """Tests for ChunkingEngine class."""

    def test_init_default_config(self):
        engine = ChunkingEngine()
        assert engine.config.strategy == ChunkingStrategy.HYBRID
        assert len(engine.chunkers) == 4

    def test_init_with_config(self):
        cfg = ChunkingConfig(max_chunk_size=2000)
        engine = ChunkingEngine(config=cfg)
        assert engine.config.max_chunk_size == 2000

    def test_all_strategies_registered(self):
        engine = ChunkingEngine()
        for strategy in ChunkingStrategy:
            assert strategy in engine.chunkers

    # -- _select_strategy --
    def test_select_strategy_small_file(self):
        engine = ChunkingEngine()
        result = engine._select_strategy("x" * 100, Language.PYTHON)
        assert result == ChunkingStrategy.STRUCTURAL

    def test_select_strategy_large_python_file(self):
        engine = ChunkingEngine()
        result = engine._select_strategy("x" * 3000, Language.PYTHON)
        assert result == ChunkingStrategy.HYBRID

    def test_select_strategy_large_javascript_file(self):
        engine = ChunkingEngine()
        result = engine._select_strategy("x" * 3000, Language.JAVASCRIPT)
        assert result == ChunkingStrategy.HYBRID

    def test_select_strategy_large_unknown_language(self):
        engine = ChunkingEngine()
        result = engine._select_strategy("x" * 3000, Language.UNKNOWN)
        assert result == ChunkingStrategy.STRUCTURAL

    # -- chunk_file --
    async def test_chunk_file_with_content(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "test.py"
        f.write_text("def hello():\n    pass\n", encoding="utf-8")
        chunks = await engine.chunk_file(str(f), content="def hello():\n    pass\n")
        assert isinstance(chunks, list)

    async def test_chunk_file_reads_from_disk(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "funcs.py"
        f.write_text("def hello():\n    pass\n\ndef world():\n    pass\n", encoding="utf-8")
        chunks = await engine.chunk_file(str(f))
        assert isinstance(chunks, list)

    async def test_chunk_file_nonexistent_returns_empty(self):
        engine = ChunkingEngine()
        chunks = await engine.chunk_file("/nonexistent/file.py")
        assert chunks == []

    async def test_chunk_file_with_explicit_strategy(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "test.py"
        f.write_text("x = 1\n", encoding="utf-8")
        chunks = await engine.chunk_file(str(f), strategy=ChunkingStrategy.BASIC)
        assert isinstance(chunks, list)

    async def test_chunk_empty_file(self, tmp_path: Path):
        engine = ChunkingEngine()
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        chunks = await engine.chunk_file(str(f))
        assert isinstance(chunks, list)

    # -- get_chunking_stats --
    def test_get_chunking_stats_empty(self):
        engine = ChunkingEngine()
        stats = engine.get_chunking_stats([])
        assert stats == {"total_chunks": 0}

    def test_get_chunking_stats_populated(self):
        engine = ChunkingEngine()
        chunks = [
            MetadataCodeChunk(
                content="abc",
                start_line=1,
                end_line=1,
                language=Language.PYTHON,
                chunk_type="basic",
                quality_score=0.5,
            ),
            MetadataCodeChunk(
                content="defgh",
                start_line=2,
                end_line=3,
                language=Language.PYTHON,
                chunk_type="function",
                quality_score=0.9,
            ),
        ]
        stats = engine.get_chunking_stats(chunks)
        assert stats["total_chunks"] == 2
        assert stats["total_size"] == 8  # 3 + 5
        assert stats["average_size"] == 4.0
        assert stats["average_quality"] == pytest.approx(0.7)
        assert "basic" in stats["chunk_types"]
        assert "function" in stats["chunk_types"]
        assert stats["size_distribution"]["min"] == 3
        assert stats["size_distribution"]["max"] == 5

    # -- _post_process_chunks --
    async def test_post_process_chunks_empty(self):
        engine = ChunkingEngine()
        result = await engine._post_process_chunks([])
        assert result == []

    async def test_post_process_chunks_filters_low_quality(self):
        cfg = ChunkingConfig(quality_threshold=0.5, overlap_size=0)
        engine = ChunkingEngine(config=cfg)
        low = MetadataCodeChunk(
            content="x",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
            quality_score=0.1,
        )
        high = MetadataCodeChunk(
            content="y" * 100,
            start_line=2,
            end_line=5,
            language=Language.PYTHON,
            chunk_type="function",
            quality_score=0.8,
        )
        result = await engine._post_process_chunks([low, high])
        assert len(result) == 1
        assert result[0] is high

    async def test_post_process_single_chunk_always_kept(self):
        cfg = ChunkingConfig(quality_threshold=0.9, overlap_size=0)
        engine = ChunkingEngine(config=cfg)
        low = MetadataCodeChunk(
            content="x",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="basic",
            quality_score=0.1,
        )
        result = await engine._post_process_chunks([low])
        assert len(result) == 1

    async def test_post_process_limits_max_chunks(self):
        cfg = ChunkingConfig(max_chunks_per_file=2, quality_threshold=0.0, overlap_size=0)
        engine = ChunkingEngine(config=cfg)
        chunks = [
            MetadataCodeChunk(
                content=f"chunk_{i}",
                start_line=i,
                end_line=i,
                language=Language.PYTHON,
                chunk_type="basic",
                quality_score=float(i) / 10,
            )
            for i in range(5)
        ]
        result = await engine._post_process_chunks(chunks)
        assert len(result) == 2
        # Should keep highest quality, re-sorted by line number
        assert result[0].start_line < result[1].start_line

    # -- chunk_multiple_files --
    async def test_chunk_multiple_files(self, tmp_path: Path):
        engine = ChunkingEngine()
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1\n", encoding="utf-8")
        f2.write_text("y = 2\n", encoding="utf-8")
        result = await engine.chunk_multiple_files([str(f1), str(f2)])
        assert isinstance(result, dict)
        assert str(f1) in result
        assert str(f2) in result

    async def test_chunk_multiple_files_handles_errors(self, tmp_path: Path):
        engine = ChunkingEngine()
        f1 = tmp_path / "a.py"
        f1.write_text("x = 1\n", encoding="utf-8")
        result = await engine.chunk_multiple_files(
            [str(f1), "/nonexistent/z.py"]
        )
        assert isinstance(result, dict)
        assert str(f1) in result
