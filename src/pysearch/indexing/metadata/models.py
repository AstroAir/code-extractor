"""
Metadata data models for pysearch.

This module contains the core data structures used by the metadata indexing system,
including entity metadata, file metadata, query specifications, and statistics.

Classes:
    EntityMetadata: Metadata for a code entity in the index
    FileMetadata: Enhanced metadata for a file in the index
    IndexQuery: Query specification for the enhanced index
    IndexStats: Statistics for the enhanced index

Features:
    - Comprehensive metadata structures
    - Query specification with multiple filter criteria
    - Performance statistics and analytics
    - Semantic metadata support
    - Dependency tracking
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EntityMetadata:
    """Metadata for a code entity in the index."""

    entity_id: str
    name: str
    entity_type: str
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    language: str = "unknown"
    scope: str | None = None
    complexity_score: float = 0.0
    semantic_embedding: list[float] | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class FileMetadata:
    """Enhanced metadata for a file in the index."""

    file_path: str
    size: int
    mtime: float
    sha1: str | None = None
    language: str = "unknown"
    line_count: int = 0
    entity_count: int = 0
    complexity_score: float = 0.0
    semantic_summary: str | None = None
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    last_indexed: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class IndexQuery:
    """Query specification for the enhanced index."""

    # File-level filters
    file_patterns: list[str] | None = None
    languages: list[str] | None = None
    min_size: int | None = None
    max_size: int | None = None
    min_lines: int | None = None
    max_lines: int | None = None
    modified_after: float | None = None
    modified_before: float | None = None

    # Entity-level filters
    entity_types: list[str] | None = None
    entity_names: list[str] | None = None
    has_docstring: bool | None = None
    min_complexity: float | None = None
    max_complexity: float | None = None

    # Semantic filters
    semantic_query: str | None = None
    similarity_threshold: float = 0.7

    # Result options
    include_entities: bool = True
    include_file_content: bool = False
    limit: int | None = None
    offset: int = 0


@dataclass
class IndexStats:
    """Statistics for the enhanced index."""

    total_files: int = 0
    total_entities: int = 0
    languages: dict[str, int] = field(default_factory=dict)
    entity_types: dict[str, int] = field(default_factory=dict)
    avg_file_size: float = 0.0
    avg_entities_per_file: float = 0.0
    index_size_mb: float = 0.0
    last_build_time: float = 0.0
    build_duration: float = 0.0
