"""
Core functionality for the pysearch package.

This module contains the fundamental components of the search engine:
- Main API classes and interfaces
- Configuration management
- Core data types and structures
- Search history tracking

The core module provides the essential building blocks that other modules
depend on, ensuring a clean separation of concerns and maintainable architecture.
"""

from .api import PySearch
from .config import SearchConfig
from .history import SearchHistory
from .types import (
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

__all__ = [
    # Main classes
    "PySearch",
    "SearchConfig",
    "SearchHistory",
    # Data types
    "OutputFormat",
    "SearchItem",
    "SearchStats",
    "SearchResult",
    "Query",
    "ASTFilters",
    "MatchSpan",
    "Language",
    "FileMetadata",
    "MetadataFilters",
]
