"""Core Search Tools — text, regex, AST, and semantic search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_core_search_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register core search tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def search_text(
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for text patterns in files.

        Args:
            pattern: Text pattern to search for
            paths: Optional list of paths to search (uses configured paths if None)
            context: Number of context lines around matches (default: 3)
            case_sensitive: Whether search should be case sensitive (default: False)
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Search results with matching text, file locations, and statistics
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_text(
                pattern, paths, context, case_sensitive, session_id=session_id
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Text search failed: {e}") from e

    @mcp.tool
    def search_regex(
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for regex patterns in files.

        Args:
            pattern: Regular expression pattern to search for
            paths: Optional list of paths to search
            context: Number of context lines around matches
            case_sensitive: Whether search should be case sensitive
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Search results with regex matches
        """
        _validate(is_regex=True, pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_regex(
                pattern, paths, context, case_sensitive, session_id=session_id
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Regex search failed: {e}") from e

    @mcp.tool
    def search_ast(
        pattern: str,
        func_name: str | None = None,
        class_name: str | None = None,
        decorator: str | None = None,
        imported: str | None = None,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search using Abstract Syntax Tree analysis with structural filters.

        Args:
            pattern: Base pattern to search for
            func_name: Regex pattern to match function names
            class_name: Regex pattern to match class names
            decorator: Regex pattern to match decorator names
            imported: Regex pattern to match imported symbols
            paths: Optional list of paths to search
            context: Number of context lines around matches
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Search results with AST-matched items
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_ast(
                pattern,
                func_name,
                class_name,
                decorator,
                imported,
                paths,
                context,
                session_id=session_id,
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"AST search failed: {e}") from e

    @mcp.tool
    def search_semantic(
        concept: str,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search for semantic concepts in code.

        Expands the concept into related patterns and performs a combined search.

        Args:
            concept: Semantic concept to search for (e.g., "database", "authentication", "testing")
            paths: Optional list of paths to search
            context: Number of context lines around matches
            session_id: Optional session ID for context-aware search tracking

        Returns:
            Search results with semantically related matches
        """
        _validate(pattern=concept, paths=paths, context=context)
        try:
            resp = engine.search_semantic(concept, paths, context, session_id=session_id)
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Semantic search failed: {e}") from e
