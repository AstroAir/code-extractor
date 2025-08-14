"""
Search strategies and pattern matching implementations.

This module contains all the different search approaches and matching algorithms:
- Text and regex pattern matching
- Fuzzy search capabilities
- Semantic search (basic and advanced)
- Result scoring and ranking

The search module encapsulates the various ways to find and match content,
providing a clean interface for different search strategies.
"""

from .matchers import search_in_file
from .scorer import (
    RankingStrategy,
    cluster_results_by_similarity,
    score_item,
    sort_items,
)
from .semantic import semantic_similarity_score
from .semantic_advanced import SemanticSearchEngine

__all__ = [
    # Pattern matching
    "search_in_file",
    # Scoring and ranking
    "RankingStrategy",
    "cluster_results_by_similarity",
    "score_item",
    "sort_items",
    # Semantic search
    "semantic_similarity_score",
    "SemanticSearchEngine",
]
