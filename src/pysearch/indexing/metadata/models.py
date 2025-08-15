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
from typing import Any, Dict, List, Optional


@dataclass
class EntityMetadata:
    """Metadata for a code entity in the index."""

    entity_id: str
    name: str
    entity_type: str
    file_path: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    language: str = "unknown"
    scope: Optional[str] = None
    complexity_score: float = 0.0
    semantic_embedding: Optional[List[float]] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)


@dataclass
class FileMetadata:
    """Enhanced metadata for a file in the index."""

    file_path: str
    size: int
    mtime: float
    sha1: Optional[str] = None
    language: str = "unknown"
    line_count: int = 0
    entity_count: int = 0
    complexity_score: float = 0.0
    semantic_summary: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    last_indexed: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)


@dataclass
class IndexQuery:
    """Query specification for the enhanced index."""

    # File-level filters
    file_patterns: Optional[List[str]] = None
    languages: Optional[List[str]] = None
    min_size: Optional[int] = None
    max_size: Optional[int] = None
    min_lines: Optional[int] = None
    max_lines: Optional[int] = None
    modified_after: Optional[float] = None
    modified_before: Optional[float] = None

    # Entity-level filters
    entity_types: Optional[List[str]] = None
    entity_names: Optional[List[str]] = None
    has_docstring: Optional[bool] = None
    min_complexity: Optional[float] = None
    max_complexity: Optional[float] = None

    # Semantic filters
    semantic_query: Optional[str] = None
    similarity_threshold: float = 0.7

    # Result options
    include_entities: bool = True
    include_file_content: bool = False
    limit: Optional[int] = None
    offset: int = 0


@dataclass
class IndexStats:
    """Statistics for the enhanced index."""

    total_files: int = 0
    total_entities: int = 0
    languages: Dict[str, int] = field(default_factory=dict)
    entity_types: Dict[str, int] = field(default_factory=dict)
    avg_file_size: float = 0.0
    avg_entities_per_file: float = 0.0
    index_size_mb: float = 0.0
    last_build_time: float = 0.0
    build_duration: float = 0.0
