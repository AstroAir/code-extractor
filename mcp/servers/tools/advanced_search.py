"""Advanced Search Tools — fuzzy, multi-pattern, corrections, word-level fuzzy."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_advanced_search_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register advanced search tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def search_fuzzy(
        pattern: str,
        similarity_threshold: float = 0.6,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Perform fuzzy search with approximate string matching.

        Args:
            pattern: Text pattern to search for with fuzzy matching
            similarity_threshold: Minimum similarity score (0.0 to 1.0, default: 0.6)
            max_results: Maximum number of results to return (default: 100)
            paths: Optional list of paths to search
            context: Number of context lines around matches
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Search results with approximate matches
        """
        _validate(
            pattern=pattern,
            paths=paths,
            context=context,
            similarity_threshold=similarity_threshold,
            max_results=max_results,
        )
        try:
            resp = engine.search_fuzzy(
                pattern,
                similarity_threshold,
                max_results,
                paths,
                context,
                session_id=session_id,
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Fuzzy search failed: {e}") from e

    @mcp.tool
    def search_multi_pattern(
        patterns: list[str],
        operator: str = "OR",
        use_regex: bool = False,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for multiple patterns with logical operators.

        Args:
            patterns: List of patterns to search for
            operator: Logical operator — "AND" (all must match in same file) or "OR" (any match)
            use_regex: Whether patterns are regular expressions
            paths: Optional list of paths to search
            context: Number of context lines around matches
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Combined search results from multiple patterns
        """
        if not patterns:
            raise ToolError("At least one pattern is required")
        _validate(paths=paths, context=context)
        try:
            resp = engine.search_multi_pattern(
                patterns,
                operator,
                use_regex,
                paths,
                context,
                session_id=session_id,
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Multi-pattern search failed: {e}") from e

    @mcp.tool
    def suggest_corrections(
        word: str,
        max_suggestions: int = 10,
        algorithm: str = "damerau_levenshtein",
        paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Suggest spelling corrections for a word based on codebase identifiers.

        Scans indexed files, extracts identifiers, and returns the most similar
        ones using fuzzy matching algorithms.

        Args:
            word: Word to find corrections for
            max_suggestions: Maximum number of suggestions (default: 10)
            algorithm: Similarity algorithm — "levenshtein", "damerau_levenshtein",
                       "jaro_winkler", "soundex", "metaphone" (default: "damerau_levenshtein")
            paths: Optional list of paths to scan for identifiers

        Returns:
            List of suggestions with identifier name and similarity score
        """
        _validate(pattern=word, paths=paths)
        try:
            return engine.suggest_corrections(word, max_suggestions, algorithm, paths)
        except Exception as e:
            raise ToolError(f"Suggestion generation failed: {e}") from e

    @mcp.tool
    def search_word_fuzzy(
        pattern: str,
        max_distance: int = 2,
        min_similarity: float = 0.6,
        algorithms: list[str] | None = None,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> dict[str, Any]:
        """
        Word-level fuzzy search using actual similarity algorithms.

        Unlike regex-based fuzzy search, this compares individual words in file
        content against the pattern using real edit-distance and similarity
        algorithms, returning matches with precise similarity scores.

        Args:
            pattern: Word or short phrase to search for
            max_distance: Maximum edit distance for distance-based algorithms (default: 2)
            min_similarity: Minimum similarity score 0.0-1.0 (default: 0.6)
            algorithms: List of algorithm names — "levenshtein", "damerau_levenshtein",
                        "jaro_winkler", "soundex", "metaphone" (default: all three distance-based)
            max_results: Maximum number of results (default: 100)
            paths: Optional list of paths to search
            context: Number of context lines around matches (default: 3)

        Returns:
            Search results with word-level fuzzy matches
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.word_level_fuzzy_search(
                pattern, max_distance, min_similarity, algorithms, max_results, paths, context
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Word-level fuzzy search failed: {e}") from e
