"""
Base classes and interfaces for the enhanced indexing system.

This module provides the abstract base classes and interfaces that all
enhanced index implementations must follow to participate in the
enhanced indexing system.

Classes:
    CodebaseIndex: Abstract base class for all enhanced index types

Features:
    - Standard interface for all index types
    - Progress tracking support
    - Async generator pattern for updates
    - Flexible retrieval interface
    - Time estimation for progress calculation
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any

from ...analysis.content_addressing import (
    IndexingProgressUpdate,
    IndexTag,
    MarkCompleteCallback,
    RefreshIndexResults,
)


class CodebaseIndex(ABC):
    """
    Abstract base class for all enhanced index types.

    This interface defines the contract that all index implementations must follow
    to participate in the enhanced indexing system.
    """

    @property
    @abstractmethod
    def artifact_id(self) -> str:
        """Unique identifier for this index type."""
        pass

    @property
    @abstractmethod
    def relative_expected_time(self) -> float:
        """Relative time cost for this index type (1.0 = baseline)."""
        pass

    @abstractmethod
    async def update(
        self,
        tag: IndexTag,
        results: RefreshIndexResults,
        mark_complete: MarkCompleteCallback,
        repo_name: str | None = None,
    ) -> AsyncGenerator[IndexingProgressUpdate, None]:
        """
        Update the index with new/changed/deleted files.

        Args:
            tag: Index tag identifying the specific index instance
            results: Files to compute/delete/add_tag/remove_tag
            mark_complete: Callback to mark operations as complete
            repo_name: Optional repository name for context

        Yields:
            Progress updates during the indexing operation
        """
        # This is an abstract async generator method
        if False:  # pragma: no cover
            yield

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        tag: IndexTag,
        limit: int = 50,
        **kwargs: Any,
    ) -> list[Any]:
        """
        Retrieve results from this index.

        Args:
            query: Search query
            tag: Index tag to search within
            limit: Maximum number of results
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        pass
