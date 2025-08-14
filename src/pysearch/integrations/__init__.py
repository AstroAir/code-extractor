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

from .multi_repo import MultiRepoSearchEngine, MultiRepoSearchResult, RepositoryInfo

__all__ = [
    # Multi-repository support
    "MultiRepoSearchEngine",
    "MultiRepoSearchResult",
    "RepositoryInfo",
]
