"""
Re-export types commonly used in integrations.

This module provides convenient access to core types that are frequently
used in integration modules, avoiding the need for long import paths.
"""

# Re-export core types
from ..core.types import (
    ASTFilters,
    FileMetadata,
    Language,
    MatchSpan,
    MetadataFilters,
    OutputFormat,
    Query,
    SearchItem,
    SearchResult,
    SearchStats,
)

# Re-export integration-specific types
from .multi_repo import MultiRepoSearchResult, RepositoryInfo

__all__ = [
    # Core types
    "ASTFilters",
    "FileMetadata",
    "Language",
    "MatchSpan",
    "MetadataFilters",
    "OutputFormat",
    "Query",
    "SearchItem",
    "SearchResult",
    "SearchStats",
    # Integration types
    "MultiRepoSearchResult",
    "RepositoryInfo",
]
