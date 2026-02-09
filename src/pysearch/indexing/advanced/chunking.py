"""
Advanced code-aware chunking engine for enhanced indexing.

This module implements sophisticated chunking strategies that understand code
structure and semantics, going beyond simple line-based chunking to create
meaningful code segments optimized for search and retrieval.

Classes:
    ChunkingStrategy: Abstract base for chunking strategies
    SemanticChunker: Semantic-aware chunking
    StructuralChunker: Structure-aware chunking using AST
    HybridChunker: Combines multiple chunking approaches
    ChunkingEngine: Main chunking coordination engine

Features:
    - Multiple chunking strategies (semantic, structural, hybrid)
    - Language-aware boundary detection
    - Function/class boundary preservation
    - Context-aware chunk sizing
    - Overlap management for better retrieval
    - Chunk quality scoring and optimization
    - Memory-efficient streaming processing

Example:
    Basic chunking:
        >>> from pysearch.advanced_chunking import ChunkingEngine
        >>> engine = ChunkingEngine()
        >>> chunks = await engine.chunk_file("example.py", content)

    Advanced chunking with custom strategy:
        >>> from pysearch.advanced_chunking import HybridChunker
        >>> chunker = HybridChunker(max_chunk_size=1500, overlap_size=100)
        >>> async for chunk in chunker.chunk_content(content, "python"):
        ...     print(f"Chunk: {chunk.start_line}-{chunk.end_line}")
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ...analysis.language_detection import detect_language
from ...analysis.language_support import CodeChunk, LanguageConfig, language_registry
from ...core.types import Language
from ...utils.logging_config import get_logger

logger = get_logger()


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    BASIC = "basic"  # Simple line-based chunking
    STRUCTURAL = "structural"  # AST/tree-sitter based chunking
    SEMANTIC = "semantic"  # Semantic similarity based chunking
    HYBRID = "hybrid"  # Combination of multiple strategies


@dataclass
class ChunkingConfig:
    """Configuration for chunking operations."""

    strategy: ChunkingStrategy = ChunkingStrategy.HYBRID
    max_chunk_size: int = 1000
    min_chunk_size: int = 50
    overlap_size: int = 100
    respect_boundaries: bool = True
    include_context: bool = True
    quality_threshold: float = 0.7
    max_chunks_per_file: int = 100


@dataclass
class ChunkMetadata:
    """Metadata for a code chunk."""

    chunk_id: str
    quality_score: float
    boundary_type: str  # "function", "class", "block", "line"
    contains_entities: list[str]
    dependencies: list[str]
    semantic_tags: list[str]
    complexity_score: float


@dataclass
class MetadataCodeChunk(CodeChunk):
    """Enhanced code chunk with additional metadata."""

    chunk_id: str = ""
    metadata: ChunkMetadata | None = None
    overlap_with: list[str] = field(default_factory=list)
    quality_score: float = 0.0


class BasicChunker:
    """Simple line-based chunking without structural awareness."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    async def chunk_content(
        self,
        content: str,
        language: Language,
        file_path: str = "",
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Chunk content by fixed line count, splitting at blank lines when possible."""
        lines = content.split("\n")
        max_lines = max(1, self.config.max_chunk_size // 40)  # ~40 chars per line estimate
        current_chunk: list[str] = []
        start_line = 1

        for _i, line in enumerate(lines, 1):
            current_chunk.append(line)

            # Check if we've hit the size limit
            chunk_text = "\n".join(current_chunk)
            if len(chunk_text) >= self.config.max_chunk_size or len(current_chunk) >= max_lines:
                # Try to split at a blank line boundary
                split_idx = len(current_chunk)
                for j in range(len(current_chunk) - 1, max(0, len(current_chunk) // 2), -1):
                    if current_chunk[j].strip() == "":
                        split_idx = j + 1
                        break

                emit_lines = current_chunk[:split_idx]
                remainder = current_chunk[split_idx:]

                if emit_lines:
                    yield MetadataCodeChunk(
                        content="\n".join(emit_lines),
                        start_line=start_line,
                        end_line=start_line + len(emit_lines) - 1,
                        language=language,
                        chunk_type="basic",
                        chunk_id=f"{file_path}:{start_line}:{start_line + len(emit_lines) - 1}",
                        quality_score=0.3,
                    )

                start_line = start_line + len(emit_lines)
                current_chunk = remainder

        # Emit final chunk
        if current_chunk:
            chunk_text = "\n".join(current_chunk)
            if chunk_text.strip():
                yield MetadataCodeChunk(
                    content=chunk_text,
                    start_line=start_line,
                    end_line=start_line + len(current_chunk) - 1,
                    language=language,
                    chunk_type="basic",
                    chunk_id=f"{file_path}:{start_line}:{start_line + len(current_chunk) - 1}",
                    quality_score=0.3,
                )


class ChunkingStrategyBase(ABC):
    """Abstract base for chunking strategies."""

    def __init__(self, config: ChunkingConfig):
        self.config = config

    @abstractmethod
    async def chunk_content(
        self,
        content: str,
        language: Language,
        file_path: str = "",
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Chunk content according to this strategy."""
        # This is an abstract async generator method
        # Subclasses should implement this as an async generator using yield
        if False:  # pragma: no cover
            yield

    def calculate_chunk_quality(self, chunk: MetadataCodeChunk) -> float:
        """Calculate quality score for a chunk."""
        quality = 0.0

        # Size quality (prefer chunks near target size)
        size_ratio = len(chunk.content) / self.config.max_chunk_size
        if 0.3 <= size_ratio <= 1.0:
            quality += 0.3
        elif size_ratio > 1.0:
            quality += max(0.0, 0.3 - (size_ratio - 1.0) * 0.2)

        # Boundary quality (prefer complete entities)
        if chunk.chunk_type in ["function", "class", "method"]:
            quality += 0.4
        elif chunk.chunk_type in ["block", "statement"]:
            quality += 0.2

        # Content quality (prefer meaningful code)
        if chunk.entity_name:
            quality += 0.2
        if chunk.dependencies:
            quality += 0.1

        return min(1.0, quality)


class StructuralChunker(ChunkingStrategyBase):
    """Structure-aware chunking using AST/tree-sitter."""

    def _build_language_config(self) -> LanguageConfig:
        """Build a LanguageConfig from the current ChunkingConfig."""
        return LanguageConfig(
            max_chunk_size=self.config.max_chunk_size,
            min_chunk_size=self.config.min_chunk_size,
            respect_boundaries=self.config.respect_boundaries,
            include_comments=True,
            include_docstrings=True,
            include_imports=True,
        )

    async def chunk_content(
        self,
        content: str,
        language: Language,
        file_path: str = "",
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Chunk content based on code structure."""
        # Apply LanguageConfig to synchronize chunking settings with the processor
        lang_config = self._build_language_config()
        processor = language_registry.get_processor(language)
        if processor:
            processor.config = lang_config
            # Use tree-sitter based chunking
            async for chunk in processor.chunk_code(content, self.config.max_chunk_size):
                enhanced_chunk = MetadataCodeChunk(
                    content=chunk.content,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    language=chunk.language,
                    chunk_type=chunk.chunk_type,
                    entity_name=chunk.entity_name,
                    entity_type=chunk.entity_type,
                    complexity_score=chunk.complexity_score,
                    dependencies=chunk.dependencies,
                    chunk_id=f"{file_path}:{chunk.start_line}:{chunk.end_line}",
                )
                enhanced_chunk.quality_score = self.calculate_chunk_quality(enhanced_chunk)
                yield enhanced_chunk
        else:
            # Fallback to basic structural chunking
            async for chunk in self._basic_structural_chunk(content, language, file_path):
                yield chunk

    async def _basic_structural_chunk(
        self,
        content: str,
        language: Language,
        file_path: str,
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Basic structural chunking without tree-sitter."""
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_size = 0
        start_line = 1

        # Language-specific patterns for boundaries
        if language == Language.PYTHON:
            boundary_patterns = [r"^\s*def\s+", r"^\s*class\s+", r"^\s*async\s+def\s+"]
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            boundary_patterns = [r"^\s*function\s+", r"^\s*class\s+", r"^\s*const\s+\w+\s*=\s*\("]
        elif language == Language.JAVA:
            boundary_patterns = [r"^\s*public\s+", r"^\s*private\s+", r"^\s*protected\s+"]
        else:
            boundary_patterns = [r"^\s*\w+.*{", r"^\s*}"]

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1

            # Check if this line starts a new boundary
            is_boundary = any(re.match(pattern, line) for pattern in boundary_patterns)

            if (
                current_size + line_size > self.config.max_chunk_size
                and current_chunk
                and (is_boundary or not self.config.respect_boundaries)
            ):

                # Yield current chunk
                chunk_content = "\n".join(current_chunk)
                yield MetadataCodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1,
                    language=language,
                    chunk_type="structural",
                    chunk_id=f"{file_path}:{start_line}:{i-1}",
                    quality_score=0.5,  # Default quality for basic chunks
                )

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
                start_line = i
            else:
                current_chunk.append(line)
                current_size += line_size

        # Yield final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            yield MetadataCodeChunk(
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines),
                language=language,
                chunk_type="structural",
                chunk_id=f"{file_path}:{start_line}:{len(lines)}",
                quality_score=0.5,
            )


class SemanticChunker(ChunkingStrategyBase):
    """Semantic-aware chunking based on content similarity."""

    async def chunk_content(
        self,
        content: str,
        language: Language,
        file_path: str = "",
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Chunk content based on semantic similarity."""
        # First get structural chunks as base
        structural_chunker = StructuralChunker(self.config)
        structural_chunks = []

        async for chunk in structural_chunker.chunk_content(content, language, file_path):
            structural_chunks.append(chunk)

        # Group semantically similar chunks
        semantic_groups = await self._group_chunks_semantically(structural_chunks)

        for group in semantic_groups:
            if len(group) == 1:
                yield group[0]
            else:
                # Merge semantically similar chunks
                merged_chunk = await self._merge_chunks(group, file_path)
                if merged_chunk:
                    yield merged_chunk

    async def _group_chunks_semantically(
        self,
        chunks: list[MetadataCodeChunk],
    ) -> list[list[MetadataCodeChunk]]:
        """Group chunks by semantic similarity."""
        if not chunks:
            return []

        groups = []
        remaining_chunks = chunks.copy()

        while remaining_chunks:
            current_group = [remaining_chunks.pop(0)]
            current_content = current_group[0].content

            # Find similar chunks
            i = 0
            while i < len(remaining_chunks):
                chunk = remaining_chunks[i]
                similarity = await self._calculate_semantic_similarity(
                    current_content, chunk.content
                )

                if similarity > 0.7:  # High similarity threshold
                    current_group.append(remaining_chunks.pop(i))
                    current_content += "\n" + chunk.content
                else:
                    i += 1

            groups.append(current_group)

        return groups

    async def _calculate_semantic_similarity(
        self,
        content1: str,
        content2: str,
    ) -> float:
        """Calculate semantic similarity between two code chunks."""
        # Simple keyword-based similarity for now
        # In a full implementation, this could use embeddings

        words1 = set(re.findall(r"\w+", content1.lower()))
        words2 = set(re.findall(r"\w+", content2.lower()))

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union) if union else 0.0

    async def _merge_chunks(
        self,
        chunks: list[MetadataCodeChunk],
        file_path: str,
    ) -> MetadataCodeChunk | None:
        """Merge semantically similar chunks."""
        if not chunks:
            return None

        # Sort by line number
        chunks.sort(key=lambda c: c.start_line)

        # Merge content
        merged_content = "\n".join(chunk.content for chunk in chunks)

        # Check size limit
        if len(merged_content) > self.config.max_chunk_size * 1.5:
            # Too large to merge, return first chunk
            return chunks[0]

        # Create merged chunk
        merged_chunk = MetadataCodeChunk(
            content=merged_content,
            start_line=chunks[0].start_line,
            end_line=chunks[-1].end_line,
            language=chunks[0].language,
            chunk_type="semantic_group",
            chunk_id=f"{file_path}:{chunks[0].start_line}:{chunks[-1].end_line}",
        )

        # Merge metadata
        all_entities = []
        all_dependencies = []
        total_complexity = 0.0

        for chunk in chunks:
            if chunk.entity_name:
                all_entities.append(chunk.entity_name)
            all_dependencies.extend(chunk.dependencies)
            total_complexity += chunk.complexity_score

        merged_chunk.dependencies = list(set(all_dependencies))
        merged_chunk.complexity_score = total_complexity / len(chunks)
        merged_chunk.quality_score = self.calculate_chunk_quality(merged_chunk)

        return merged_chunk


class HybridChunker(ChunkingStrategyBase):
    """Hybrid chunking combining structural and semantic approaches."""

    def __init__(self, config: ChunkingConfig):
        super().__init__(config)
        self.structural_chunker = StructuralChunker(config)
        self.semantic_chunker = SemanticChunker(config)

    async def chunk_content(
        self,
        content: str,
        language: Language,
        file_path: str = "",
    ) -> AsyncGenerator[MetadataCodeChunk, None]:
        """Hybrid chunking using both structural and semantic approaches."""
        # Start with structural chunking
        structural_chunks = []
        async for chunk in self.structural_chunker.chunk_content(content, language, file_path):
            structural_chunks.append(chunk)

        # Apply semantic grouping to improve chunk quality
        if len(structural_chunks) > 1:
            semantic_groups = await self.semantic_chunker._group_chunks_semantically(
                structural_chunks
            )

            for group in semantic_groups:
                if len(group) == 1:
                    yield group[0]
                else:
                    # Try to merge if it improves quality
                    merged = await self.semantic_chunker._merge_chunks(group, file_path)
                    if merged and merged.quality_score > group[0].quality_score:
                        yield merged
                    else:
                        # Keep original chunks
                        for chunk in group:
                            yield chunk
        else:
            # Single chunk or no chunks
            for chunk in structural_chunks:
                yield chunk

    async def optimize_chunks(
        self,
        chunks: list[MetadataCodeChunk],
    ) -> list[MetadataCodeChunk]:
        """Optimize chunk boundaries and sizes."""
        optimized = []

        for chunk in chunks:
            # Check if chunk can be improved
            if chunk.quality_score < self.config.quality_threshold:
                # Try to improve chunk
                improved_chunk = await self._improve_chunk(chunk)
                optimized.append(improved_chunk or chunk)
            else:
                optimized.append(chunk)

        return optimized

    async def _improve_chunk(
        self,
        chunk: MetadataCodeChunk,
    ) -> MetadataCodeChunk | None:
        """Try to improve a low-quality chunk."""
        # Implementation would analyze the chunk and try to improve it
        # For now, return the original chunk
        return chunk


class ChunkingEngine:
    """
    Main chunking engine that coordinates different chunking strategies.

    This engine provides the high-level interface for code chunking operations,
    automatically selecting the best strategy based on content and language.
    """

    def __init__(self, config: ChunkingConfig | None = None):
        self.config = config or ChunkingConfig()
        self.chunkers: dict[ChunkingStrategy, ChunkingStrategyBase | BasicChunker] = {
            ChunkingStrategy.BASIC: BasicChunker(self.config),
            ChunkingStrategy.STRUCTURAL: StructuralChunker(self.config),
            ChunkingStrategy.SEMANTIC: SemanticChunker(self.config),
            ChunkingStrategy.HYBRID: HybridChunker(self.config),
        }

    async def chunk_file(
        self,
        file_path: str,
        content: str | None = None,
        strategy: ChunkingStrategy | None = None,
    ) -> list[MetadataCodeChunk]:
        """
        Chunk a file using the specified or auto-selected strategy.

        Args:
            file_path: Path to the file
            content: File content (will be read if None)
            strategy: Chunking strategy (auto-selected if None)

        Returns:
            List of enhanced code chunks
        """
        if content is None:
            try:
                from ...utils.helpers import read_text_safely

                content = read_text_safely(Path(file_path))
                if content is None:
                    logger.error(f"Could not read file {file_path}")
                    return []
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")
                return []

        # Detect language
        language = detect_language(Path(file_path), content)

        # Select strategy if not provided
        if strategy is None:
            strategy = self._select_strategy(content, language)

        # Get chunker
        chunker = self.chunkers.get(strategy)
        if not chunker:
            logger.warning(f"Unknown chunking strategy: {strategy}")
            chunker = self.chunkers[ChunkingStrategy.STRUCTURAL]

        # Perform chunking
        chunks = []
        async for chunk in chunker.chunk_content(content, language, file_path):
            chunks.append(chunk)

        # Post-process chunks
        chunks = await self._post_process_chunks(chunks)

        return chunks

    def _select_strategy(self, content: str, language: Language) -> ChunkingStrategy:
        """Auto-select the best chunking strategy."""
        content_size = len(content)

        # For small files, use structural chunking
        if content_size < 2000:
            return ChunkingStrategy.STRUCTURAL

        # For large files with good language support, use hybrid
        if language in [Language.PYTHON, Language.JAVASCRIPT, Language.TYPESCRIPT, Language.JAVA]:
            return ChunkingStrategy.HYBRID

        # For other languages, use structural
        return ChunkingStrategy.STRUCTURAL

    async def _post_process_chunks(
        self,
        chunks: list[MetadataCodeChunk],
    ) -> list[MetadataCodeChunk]:
        """Post-process chunks to improve quality."""
        if not chunks:
            return chunks

        # Filter out low-quality chunks
        quality_threshold = self.config.quality_threshold
        filtered_chunks = [
            chunk
            for chunk in chunks
            if chunk.quality_score >= quality_threshold or len(chunks) == 1
        ]

        # Add overlap if configured
        if self.config.overlap_size > 0:
            filtered_chunks = await self._add_overlap(filtered_chunks)

        # Limit number of chunks per file
        if len(filtered_chunks) > self.config.max_chunks_per_file:
            # Keep highest quality chunks
            filtered_chunks.sort(key=lambda c: c.quality_score, reverse=True)
            filtered_chunks = filtered_chunks[: self.config.max_chunks_per_file]
            # Re-sort by line number
            filtered_chunks.sort(key=lambda c: c.start_line)

        return filtered_chunks

    async def _add_overlap(
        self,
        chunks: list[MetadataCodeChunk],
    ) -> list[MetadataCodeChunk]:
        """Add overlap between adjacent chunks."""
        if len(chunks) <= 1:
            return chunks

        overlapped_chunks = []

        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk - no previous overlap
                overlapped_chunks.append(chunk)
            else:
                # Add overlap with previous chunk
                prev_chunk = chunks[i - 1]
                overlap_lines = min(
                    self.config.overlap_size, (chunk.start_line - prev_chunk.end_line) // 2
                )

                if overlap_lines > 0:
                    # Extend chunk backward
                    new_start_line = max(1, chunk.start_line - overlap_lines)

                    # Update chunk content to include overlap
                    # This would require re-reading the content
                    # For now, just track the overlap relationship
                    chunk.overlap_with.append(prev_chunk.chunk_id)
                    chunk.start_line = new_start_line

                overlapped_chunks.append(chunk)

        return overlapped_chunks

    async def chunk_multiple_files(
        self,
        file_paths: list[str],
        strategy: ChunkingStrategy | None = None,
        max_concurrent: int = 10,
    ) -> dict[str, list[MetadataCodeChunk]]:
        """
        Chunk multiple files concurrently.

        Args:
            file_paths: List of file paths to chunk
            strategy: Chunking strategy to use
            max_concurrent: Maximum concurrent chunking operations

        Returns:
            Dictionary mapping file paths to their chunks
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def chunk_single_file(file_path: str) -> tuple[str, list[MetadataCodeChunk]]:
            async with semaphore:
                chunks = await self.chunk_file(file_path, strategy=strategy)
                return file_path, chunks

        # Process files concurrently
        tasks = [chunk_single_file(fp) for fp in file_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results
        file_chunks = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Error chunking file: {result}")
            elif isinstance(result, tuple) and len(result) == 2:
                file_path, chunks = result
                file_chunks[file_path] = chunks
            else:
                logger.warning(f"Unexpected result format: {result}")

        return file_chunks

    def get_chunking_stats(self, chunks: list[MetadataCodeChunk]) -> dict[str, Any]:
        """Get statistics for a set of chunks."""
        if not chunks:
            return {"total_chunks": 0}

        total_size = sum(len(chunk.content) for chunk in chunks)
        avg_quality = sum(chunk.quality_score for chunk in chunks) / len(chunks)

        chunk_types: dict[str, int] = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        return {
            "total_chunks": len(chunks),
            "total_size": total_size,
            "average_size": total_size / len(chunks),
            "average_quality": avg_quality,
            "chunk_types": chunk_types,
            "size_distribution": {
                "min": min(len(chunk.content) for chunk in chunks),
                "max": max(len(chunk.content) for chunk in chunks),
                "median": sorted([len(chunk.content) for chunk in chunks])[len(chunks) // 2],
            },
        }
