"""
External integrations and third-party connections.

This module handles integration with external systems and services:
- Multi-repository search capabilities
- Distributed indexing across multiple nodes
- IDE integrations and hooks
- Third-party service connections

The integrations module provides extensible interfaces for connecting
pysearch with external tools and services.
"""

from typing import Any

from .multi_repo import MultiRepoSearchEngine, MultiRepoSearchResult, RepositoryInfo

__all__ = [
    # Multi-repository support
    "MultiRepoSearchEngine",
    "MultiRepoSearchResult",
    "RepositoryInfo",
    # Distributed indexing (lazy)
    "DistributedIndexingEngine",
    "IndexingWorker",
    "WorkItem",
    "WorkItemType",
    "WorkQueue",
    "WorkerStats",
    # IDE integration (lazy)
    "IDEHooks",
    "IDEIntegration",
    "HookType",
    "DefinitionLocation",
    "ReferenceLocation",
    "CompletionItem",
    "HoverInfo",
    "DocumentSymbol",
    "Diagnostic",
    "ide_query",
]

# Lazy imports for distributed_indexing and ide_hooks to avoid circular
# import chains (core.api → integrations → distributed_indexing → analysis).
_DISTRIBUTED_INDEXING_NAMES = {
    "DistributedIndexingEngine",
    "IndexingWorker",
    "WorkItem",
    "WorkItemType",
    "WorkQueue",
    "WorkerStats",
}

_IDE_HOOKS_NAMES = {
    "CompletionItem",
    "DefinitionLocation",
    "Diagnostic",
    "DocumentSymbol",
    "HookType",
    "HoverInfo",
    "IDEHooks",
    "IDEIntegration",
    "ReferenceLocation",
    "ide_query",
}


def __getattr__(name: str) -> Any:  # noqa: N807
    if name in _DISTRIBUTED_INDEXING_NAMES:
        from . import distributed_indexing

        return getattr(distributed_indexing, name)
    if name in _IDE_HOOKS_NAMES:
        from . import ide_hooks

        return getattr(ide_hooks, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
