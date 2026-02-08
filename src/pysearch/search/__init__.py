"""
Search strategies and pattern matching implementations.

This module contains all the different search approaches and matching algorithms:
- Text and regex pattern matching
- Boolean query parsing and evaluation
- Fuzzy search capabilities (multiple algorithms)
- Semantic search (basic and advanced)
- Result scoring and ranking

The search module encapsulates the various ways to find and match content,
providing a clean interface for different search strategies.
"""

from .boolean import (
    BooleanQueryEvaluator,
    BooleanQueryParser,
    evaluate_boolean_query,
    evaluate_boolean_query_with_items,
    extract_terms,
    parse_boolean_query,
)
from .fuzzy import (
    FuzzyAlgorithm,
    FuzzyMatch,
    calculate_similarity,
    fuzzy_match,
    fuzzy_regex_pattern,
    fuzzy_search_advanced,
    suggest_corrections,
)
from .matchers import search_in_file
from .scorer import (
    RankingStrategy,
    cluster_results_by_similarity,
    deduplicate_overlapping_results,
    group_results_by_file,
    score_item,
    sort_items,
)
from .semantic import concept_to_patterns, expand_semantic_query, semantic_similarity_score
from .semantic_advanced import SemanticSearchEngine

__all__ = [
    # Pattern matching
    "search_in_file",
    # Boolean query
    "BooleanQueryParser",
    "BooleanQueryEvaluator",
    "parse_boolean_query",
    "evaluate_boolean_query",
    "evaluate_boolean_query_with_items",
    "extract_terms",
    # Fuzzy search
    "FuzzyAlgorithm",
    "FuzzyMatch",
    "calculate_similarity",
    "fuzzy_match",
    "fuzzy_regex_pattern",
    "fuzzy_search_advanced",
    "suggest_corrections",
    # Scoring and ranking
    "RankingStrategy",
    "cluster_results_by_similarity",
    "deduplicate_overlapping_results",
    "group_results_by_file",
    "score_item",
    "sort_items",
    # Semantic search
    "concept_to_patterns",
    "expand_semantic_query",
    "semantic_similarity_score",
    "SemanticSearchEngine",
]
