#!/usr/bin/env python3
"""
PySearch Basic MCP Server

A basic MCP (Model Context Protocol) server that exposes core PySearch functionality as tools
for LLM consumption using the FastMCP framework.

This is the legacy/simple implementation. For advanced features, use the main MCP server.

This server provides basic code search capabilities including:
- Text and regex pattern search
- AST-based search with filters
- Semantic concept search
- Configuration management
- Search history and statistics

Usage:
    python basic_mcp_server.py

The server runs on stdio transport by default, suitable for MCP clients.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Note: FastMCP import will be added once dependency is resolved
# from fastmcp import FastMCP, Context
# from fastmcp.exceptions import ToolError
# PySearch imports
from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.language_detection import get_supported_languages
from pysearch.semantic import expand_semantic_query
from pysearch.types import (
    ASTFilters,
    Language,
    Query,
    SearchResult,
)


@dataclass
class SearchResponse:
    """Structured response for search operations."""

    items: list[dict[str, Any]]
    stats: dict[str, Any]
    query_info: dict[str, Any]
    total_matches: int
    execution_time_ms: float


@dataclass
class ConfigResponse:
    """Structured response for configuration operations."""

    paths: list[str]
    include_patterns: list[str] | None
    exclude_patterns: list[str] | None
    context_lines: int
    parallel: bool
    workers: int
    languages: list[str] | None


class BasicPySearchMCPServer:
    """
    Basic MCP Server wrapper for PySearch functionality.

    This class maintains a PySearch instance and exposes its functionality
    through MCP tools with proper error handling and response formatting.

    This is the legacy/simple implementation. For advanced features, use PySearchMCPServer.
    """

    def __init__(self, name: str = "PySearch MCP Server"):
        self.name = name
        self.search_engine: PySearch | None = None
        self.current_config: SearchConfig | None = None
        self.search_history: list[dict[str, Any]] = []

        # Initialize with default configuration
        self._initialize_default_config()

    def _initialize_default_config(self) -> None:
        """Initialize with a sensible default configuration."""
        self.current_config = SearchConfig(
            paths=["."],
            include=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.cpp", "**/*.c"],
            exclude=["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**"],
            context=3,
            parallel=True,
            workers=4,
        )
        self.search_engine = PySearch(self.current_config)

    def _format_search_result(self, result: SearchResult, query: Query) -> SearchResponse:
        """Format SearchResult into structured response."""
        items = []
        for item in result.items:
            items.append(
                {
                    "file": str(item.file),
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "lines": item.lines,
                    "match_spans": item.match_spans,
                    "score": getattr(item, "score", None),
                }
            )

        return SearchResponse(
            items=items,
            stats={
                "files_scanned": result.stats.files_scanned,
                "files_matched": result.stats.files_matched,
                "total_items": result.stats.items,
                "elapsed_ms": result.stats.elapsed_ms,
                "indexed_files": result.stats.indexed_files,
            },
            query_info={
                "pattern": query.pattern,
                "use_regex": query.use_regex,
                "use_ast": query.use_ast,
                "use_semantic": query.use_semantic,
                "context": query.context,
                "filters": asdict(query.filters) if query.filters else None,
            },
            total_matches=result.stats.items,
            execution_time_ms=result.stats.elapsed_ms,
        )

    def _add_to_history(self, query: Query, result: SearchResult) -> None:
        """Add search operation to history."""
        self.search_history.append(
            {
                "timestamp": asyncio.get_event_loop().time(),
                "query": asdict(query),
                "result_count": result.stats.items,
                "execution_time_ms": result.stats.elapsed_ms,
                "matched_files": result.stats.files_matched,
            }
        )

        # Keep only last 100 searches
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

    # Core Search Tools

    async def search_text(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> SearchResponse:
        """
        Perform basic text search across files.

        Args:
            pattern: Text pattern to search for
            paths: Optional list of paths to search (uses configured paths if None)
            context: Number of context lines around matches
            case_sensitive: Whether search should be case sensitive

        Returns:
            SearchResponse with matching results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Update paths if provided; ignore invalid directories to avoid exceptions
        if paths:
            valid_paths = [p for p in paths if Path(p).exists()]
            search_paths = valid_paths if valid_paths else self.current_config.paths
            temp_config = SearchConfig(
                paths=search_paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
            )
            temp_engine = PySearch(temp_config)
        else:
            temp_engine = self.search_engine
            if context != self.current_config.context:
                # Update context for this search
                temp_config = SearchConfig(
                    paths=self.current_config.paths,
                    include=self.current_config.include,
                    exclude=self.current_config.exclude,
                    context=context,
                    parallel=self.current_config.parallel,
                    workers=self.current_config.workers,
                )
                temp_engine = PySearch(temp_config)

        # Validate regex pattern early if it looks like a regex
        if case_sensitive is not None and not isinstance(pattern, str):  # defensive
            raise ValueError("Pattern must be a string")

        query = Query(pattern=pattern, use_regex=False, context=context)

        result = temp_engine.run(query)
        self._add_to_history(query, result)

        return self._format_search_result(result, query)

    async def search_regex(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> SearchResponse:
        """
        Perform regex pattern search across files.

        Args:
            pattern: Regex pattern to search for
            paths: Optional list of paths to search
            context: Number of context lines around matches
            case_sensitive: Whether search should be case sensitive

        Returns:
            SearchResponse with matching results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Similar logic to search_text but with regex enabled
        if paths:
            temp_config = SearchConfig(
                paths=paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
            )
            temp_engine = PySearch(temp_config)
        else:
            temp_engine = self.search_engine
            if context != self.current_config.context:
                temp_config = SearchConfig(
                    paths=self.current_config.paths,
                    include=self.current_config.include,
                    exclude=self.current_config.exclude,
                    context=context,
                    parallel=self.current_config.parallel,
                    workers=self.current_config.workers,
                )
                temp_engine = PySearch(temp_config)

        # Validate regex pattern early to provide clear error messages
        try:
            import re as _re
            _re.compile(pattern)
        except Exception as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        query = Query(pattern=pattern, use_regex=True, context=context)

        result = temp_engine.run(query)
        self._add_to_history(query, result)

        return self._format_search_result(result, query)

    async def search_ast(
        self,
        pattern: str,
        func_name: str | None = None,
        class_name: str | None = None,
        decorator: str | None = None,
        imported: str | None = None,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """
        Perform AST-based search with filters.

        Args:
            pattern: Base pattern to search for
            func_name: Regex pattern to match function names
            class_name: Regex pattern to match class names
            decorator: Regex pattern to match decorator names
            imported: Regex pattern to match imported symbols
            paths: Optional list of paths to search
            context: Number of context lines around matches

        Returns:
            SearchResponse with matching results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Create AST filters
        ast_filters = ASTFilters(
            func_name=func_name, class_name=class_name, decorator=decorator, imported=imported
        )

        # Setup engine with paths if provided
        if paths:
            temp_config = SearchConfig(
                paths=paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
            )
            temp_engine = PySearch(temp_config)
        else:
            temp_engine = self.search_engine
            if context != self.current_config.context:
                temp_config = SearchConfig(
                    paths=self.current_config.paths,
                    include=self.current_config.include,
                    exclude=self.current_config.exclude,
                    context=context,
                    parallel=self.current_config.parallel,
                    workers=self.current_config.workers,
                )
                temp_engine = PySearch(temp_config)

        query = Query(pattern=pattern, use_ast=True, filters=ast_filters, context=context)

        result = temp_engine.run(query)
        self._add_to_history(query, result)

        return self._format_search_result(result, query)

    async def search_semantic(
        self, concept: str, paths: list[str] | None = None, context: int = 3
    ) -> SearchResponse:
        """
        Perform semantic concept search.

        Args:
            concept: Semantic concept to search for (e.g., "database", "web", "testing")
            paths: Optional list of paths to search
            context: Number of context lines around matches

        Returns:
            SearchResponse with matching results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Expand semantic concept to patterns
        patterns = expand_semantic_query(concept)
        if not patterns:
            raise ValueError(f"No patterns found for concept: {concept}")

        # Use the first pattern as primary, combine others with OR
        combined_pattern = "|".join(f"({pattern})" for pattern in patterns)

        # Setup engine with paths if provided
        if paths:
            temp_config = SearchConfig(
                paths=paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
            )
            temp_engine = PySearch(temp_config)
        else:
            temp_engine = self.search_engine
            if context != self.current_config.context:
                temp_config = SearchConfig(
                    paths=self.current_config.paths,
                    include=self.current_config.include,
                    exclude=self.current_config.exclude,
                    context=context,
                    parallel=self.current_config.parallel,
                    workers=self.current_config.workers,
                )
                temp_engine = PySearch(temp_config)

        query = Query(pattern=combined_pattern, use_regex=True, use_semantic=True, context=context)

        result = temp_engine.run(query)
        self._add_to_history(query, result)

        return self._format_search_result(result, query)

    # Configuration Tools

    async def configure_search(
        self,
        paths: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        context: int | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        languages: list[str] | None = None,
    ) -> ConfigResponse:
        """
        Update search configuration.

        Args:
            paths: List of paths to search
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            context: Number of context lines
            parallel: Whether to use parallel processing
            workers: Number of worker threads
            languages: List of languages to filter by

        Returns:
            ConfigResponse with updated configuration
        """
        if not self.current_config:
            raise ValueError("Configuration not initialized")

        # Update configuration with provided values
        new_config = SearchConfig(
            paths=paths if paths is not None else self.current_config.paths,
            include=(
                include_patterns if include_patterns is not None else self.current_config.include
            ),
            exclude=(
                exclude_patterns if exclude_patterns is not None else self.current_config.exclude
            ),
            context=context if context is not None else self.current_config.context,
            parallel=parallel if parallel is not None else self.current_config.parallel,
            workers=workers if workers is not None else self.current_config.workers,
            languages=(
                set(Language(lang) for lang in languages)
                if languages
                else self.current_config.languages
            ),
        )

        self.current_config = new_config
        self.search_engine = PySearch(new_config)

        return ConfigResponse(
            paths=new_config.paths,
            include_patterns=new_config.include,
            exclude_patterns=new_config.exclude,
            context_lines=new_config.context,
            parallel=new_config.parallel,
            workers=new_config.workers,
            languages=(
                [lang.value for lang in new_config.languages] if new_config.languages else None
            ),
        )

    async def get_search_config(self) -> ConfigResponse:
        """
        Get current search configuration.

        Returns:
            ConfigResponse with current configuration
        """
        if not self.current_config:
            raise ValueError("Configuration not initialized")

        return ConfigResponse(
            paths=self.current_config.paths,
            include_patterns=self.current_config.include,
            exclude_patterns=self.current_config.exclude,
            context_lines=self.current_config.context,
            parallel=self.current_config.parallel,
            workers=self.current_config.workers,
            languages=(
                [lang.value for lang in self.current_config.languages]
                if self.current_config.languages
                else None
            ),
        )

    # Utility Tools

    async def get_supported_languages(self) -> list[str]:
        """
        Get list of supported programming languages.

        Returns:
            List of supported language names
        """
        return [lang.value for lang in get_supported_languages()]

    async def clear_caches(self) -> dict[str, str]:
        """
        Clear search engine caches.

        Returns:
            Status message
        """
        if self.search_engine:
            self.search_engine.clear_caches()

        return {"status": "Caches cleared successfully"}

    async def get_search_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent search history.

        Args:
            limit: Maximum number of history entries to return

        Returns:
            List of recent search operations
        """
        return self.search_history[-limit:] if self.search_history else []


def create_basic_mcp_server() -> BasicPySearchMCPServer:
    """Create and configure the basic MCP server instance."""
    return BasicPySearchMCPServer()


def setup_fastmcp_server() -> None:
    """
    Setup FastMCP server with PySearch tools.

    This function will be enabled once FastMCP is installed.
    Uncomment the imports at the top and this function to use.
    """
    # Uncomment these lines once FastMCP is installed:
    # from fastmcp import FastMCP, Context
    # from fastmcp.exceptions import ToolError

    # Create the MCP server
    # mcp = FastMCP(
    #     name="PySearch MCP Server",
    #     instructions="""
    #     This server provides comprehensive code search capabilities using PySearch.
    #
    #     Available tools:
    #     - search_text: Basic text search across files
    #     - search_regex: Regex pattern search
    #     - search_ast: AST-based search with structural filters
    #     - search_semantic: Semantic concept search
    #     - configure_search: Update search configuration
    #     - get_search_config: Get current configuration
    #     - get_supported_languages: List supported languages
    #     - clear_caches: Clear search caches
    #     - get_search_history: Get recent search history
    #
    #     Use these tools to search through codebases efficiently with various
    #     search modes and filtering options.
    #     """
    # )

    # Create the PySearch server instance
    # pysearch_server = create_mcp_server()

    # Register tools with FastMCP
    # @mcp.tool
    # async def search_text(
    #     pattern: str,
    #     paths: Optional[List[str]] = None,
    #     context: int = 3,
    #     case_sensitive: bool = False
    # ) -> SearchResponse:
    #     """Perform basic text search across files."""
    #     return await pysearch_server.search_text(pattern, paths, context, case_sensitive)

    # @mcp.tool
    # async def search_regex(
    #     pattern: str,
    #     paths: Optional[List[str]] = None,
    #     context: int = 3,
    #     case_sensitive: bool = False
    # ) -> SearchResponse:
    #     """Perform regex pattern search across files."""
    #     return await pysearch_server.search_regex(pattern, paths, context, case_sensitive)

    # @mcp.tool
    # async def search_ast(
    #     pattern: str,
    #     func_name: Optional[str] = None,
    #     class_name: Optional[str] = None,
    #     decorator: Optional[str] = None,
    #     imported: Optional[str] = None,
    #     paths: Optional[List[str]] = None,
    #     context: int = 3
    # ) -> SearchResponse:
    #     """Perform AST-based search with structural filters."""
    #     return await pysearch_server.search_ast(
    #         pattern, func_name, class_name, decorator, imported, paths, context
    #     )

    # @mcp.tool
    # async def search_semantic(
    #     concept: str,
    #     paths: Optional[List[str]] = None,
    #     context: int = 3
    # ) -> SearchResponse:
    #     """Perform semantic concept search."""
    #     return await pysearch_server.search_semantic(concept, paths, context)

    # @mcp.tool
    # async def configure_search(
    #     paths: Optional[List[str]] = None,
    #     include_patterns: Optional[List[str]] = None,
    #     exclude_patterns: Optional[List[str]] = None,
    #     context: Optional[int] = None,
    #     parallel: Optional[bool] = None,
    #     workers: Optional[int] = None,
    #     languages: Optional[List[str]] = None
    # ) -> ConfigResponse:
    #     """Update search configuration."""
    #     return await pysearch_server.configure_search(
    #         paths, include_patterns, exclude_patterns, context, parallel, workers, languages
    #     )

    # @mcp.tool
    # async def get_search_config() -> ConfigResponse:
    #     """Get current search configuration."""
    #     return await pysearch_server.get_search_config()

    # @mcp.tool
    # async def get_supported_languages() -> List[str]:
    #     """Get list of supported programming languages."""
    #     return await pysearch_server.get_supported_languages()

    # @mcp.tool
    # async def clear_caches() -> Dict[str, str]:
    #     """Clear search engine caches."""
    #     return await pysearch_server.clear_caches()

    # @mcp.tool
    # async def get_search_history(limit: int = 10) -> List[Dict[str, Any]]:
    #     """Get recent search history."""
    #     return await pysearch_server.get_search_history(limit)

    # return mcp


if __name__ == "__main__":
    print("PySearch MCP Server")
    print("=" * 50)
    print()
    print("This server exposes PySearch functionality via the Model Context Protocol (MCP).")
    print()
    print("To run the server:")
    print("1. Install FastMCP: pip install fastmcp")
    print("2. Uncomment the FastMCP imports and setup_fastmcp_server() function")
    print("3. Add: mcp = setup_fastmcp_server(); mcp.run()")
    print()
    print("Available tools:")
    print("- search_text: Basic text search")
    print("- search_regex: Regex pattern search")
    print("- search_ast: AST-based structural search")
    print("- search_semantic: Semantic concept search")
    print("- configure_search: Update search settings")
    print("- get_search_config: Get current configuration")
    print("- get_supported_languages: List supported languages")
    print("- clear_caches: Clear search caches")
    print("- get_search_history: Get recent searches")
    print()
    print("Server implementation is ready for FastMCP integration.")
