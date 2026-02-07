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
    # Multi-repo types
    "MultiRepoSearchResult",
    "RepositoryInfo",
    # Distributed indexing types (lazy)
    "WorkItem",
    "WorkItemType",
    "WorkerStats",
    # IDE hook types (lazy)
    "HookType",
    "DefinitionLocation",
    "ReferenceLocation",
    "CompletionItem",
    "HoverInfo",
    "DocumentSymbol",
    "Diagnostic",
]

# Lazy imports to avoid circular dependencies
_DISTRIBUTED_INDEXING_NAMES = {"WorkItem", "WorkItemType", "WorkerStats"}
_IDE_HOOKS_NAMES = {
    "CompletionItem", "DefinitionLocation", "Diagnostic",
    "DocumentSymbol", "HookType", "HoverInfo", "ReferenceLocation",
}


def __getattr__(name: str):  # noqa: N807
    if name in _DISTRIBUTED_INDEXING_NAMES:
        from . import distributed_indexing
        return getattr(distributed_indexing, name)
    if name in _IDE_HOOKS_NAMES:
        from . import ide_hooks
        return getattr(ide_hooks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
