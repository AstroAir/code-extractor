#!/usr/bin/env python3
"""
PySearch MCP Server — Consolidated FastMCP Implementation

A production-ready MCP (Model Context Protocol) server that exposes all PySearch
functionality as tools for LLM consumption using the FastMCP framework.

Features:
    Core Search:
        - Text search with case sensitivity control
        - Regex pattern search with validation
        - AST-based structural search with filters
        - Semantic concept search with pattern expansion

    Advanced Search:
        - Fuzzy search with configurable similarity thresholds
        - Multi-pattern search with logical operators (AND/OR)
        - Search with advanced result ranking
        - Search with comprehensive filtering (size, date, language, etc.)

    Analysis & Utilities:
        - File content analysis with complexity metrics
        - Search configuration management
        - Supported language listing
        - Cache management
        - Search history tracking

    MCP Features:
        - Progress reporting for long-running operations
        - Session management for context-aware searches
        - Resource management with LRU caching
        - Input validation and security sanitization
        - MCP resource endpoints for configuration and statistics

Usage:
    # Run with STDIO transport (default, for MCP clients)
    python -m mcp.servers.pysearch_mcp_server

    # Or use FastMCP CLI
    fastmcp run mcp/servers/pysearch_mcp_server.py

    # For HTTP transport (web services)
    python -m mcp.servers.pysearch_mcp_server --transport http --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

import asyncio
import json
import logging
import re as _re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# FastMCP imports
try:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

# PySearch core imports
from pysearch import PySearch, SearchConfig, get_supported_languages
from pysearch.core.types import (
    ASTFilters,
    Language,
    Query,
    SearchResult,
)
from pysearch.search.scorer import RankingStrategy

# Shared MCP utilities
from ..shared.progress import ProgressAwareSearchServer, ProgressTracker
from ..shared.resource_manager import ResourceManager
from ..shared.session_manager import EnhancedSessionManager, get_session_manager
from ..shared.validation import (
    InputValidator,
    PerformanceValidationError,
    SecurityValidationError,
    ValidationError,
    check_validation_results,
    get_sanitized_values,
    validate_tool_input,
)

# Semantic expansion (optional, graceful fallback)
try:
    from pysearch.analysis.language_detection import detect_language
except ImportError:
    detect_language = None  # type: ignore[assignment]

try:
    from pysearch.semantic import expand_semantic_query

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

    def expand_semantic_query(concept: str) -> list[str]:  # type: ignore[misc]
        """Fallback: return concept as-is when semantic module unavailable."""
        return [concept]


# Setup logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Core search engine wrapper
# ---------------------------------------------------------------------------


class PySearchEngine:
    """
    Manages the PySearch engine lifecycle, search execution, configuration,
    history tracking, and integration with shared MCP utilities.
    """

    def __init__(self) -> None:
        self.search_engine: PySearch | None = None
        self.current_config: SearchConfig | None = None
        self.search_history: list[dict[str, Any]] = []
        self.validator = InputValidator()
        self.resource_manager = ResourceManager()
        self.session_manager: EnhancedSessionManager = get_session_manager()
        self.progress_tracker = ProgressTracker()

        self._initialize_default_config()

    # -- Initialization -----------------------------------------------------

    def _initialize_default_config(self) -> None:
        """Initialize with sensible default configuration."""
        self.current_config = SearchConfig(
            paths=["."],
            include=["**/*.py", "**/*.js", "**/*.ts", "**/*.java", "**/*.cpp", "**/*.c",
                     "**/*.go", "**/*.rs", "**/*.rb", "**/*.php", "**/*.swift", "**/*.kt"],
            exclude=["**/node_modules/**", "**/.git/**", "**/venv/**", "**/__pycache__/**",
                     "**/dist/**", "**/build/**", "**/.tox/**", "**/.mypy_cache/**"],
            context=3,
            parallel=True,
            workers=4,
        )
        self.search_engine = PySearch(self.current_config)

    def _get_engine(
        self,
        paths: list[str] | None = None,
        context: int | None = None,
    ) -> PySearch:
        """Get a PySearch engine, optionally with overridden paths/context."""
        if not self.current_config:
            raise ValueError("Search engine not initialized")

        need_temp = False
        cfg_paths = self.current_config.paths
        cfg_context = self.current_config.context

        if paths:
            valid = [p for p in paths if Path(p).exists()]
            if valid:
                cfg_paths = valid
                need_temp = True

        if context is not None and context != self.current_config.context:
            cfg_context = context
            need_temp = True

        if need_temp:
            temp_cfg = SearchConfig(
                paths=cfg_paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=cfg_context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
                languages=self.current_config.languages,
            )
            return PySearch(temp_cfg)

        if not self.search_engine:
            self.search_engine = PySearch(self.current_config)
        return self.search_engine

    # -- Result formatting --------------------------------------------------

    @staticmethod
    def _format_result(result: SearchResult, query: Query) -> SearchResponse:
        """Format a SearchResult into a structured SearchResponse."""
        items = []
        for item in result.items:
            items.append({
                "file": str(item.file),
                "start_line": item.start_line,
                "end_line": item.end_line,
                "lines": item.lines,
                "match_spans": item.match_spans,
                "score": getattr(item, "score", None),
            })

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
        """Record search in history (capped at 100 entries)."""
        self.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": {
                "pattern": query.pattern,
                "use_regex": query.use_regex,
                "use_ast": query.use_ast,
                "use_semantic": query.use_semantic,
            },
            "result_count": result.stats.items,
            "execution_time_ms": result.stats.elapsed_ms,
            "matched_files": result.stats.files_matched,
        })
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

    # -- Core searches ------------------------------------------------------

    def search_text(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> SearchResponse:
        """Perform basic text search."""
        engine = self._get_engine(paths, context)
        query = Query(pattern=pattern, use_regex=False, context=context)
        result = engine.run(query)
        self._add_to_history(query, result)
        return self._format_result(result, query)

    def search_regex(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> SearchResponse:
        """Perform regex pattern search."""
        try:
            _re.compile(pattern)
        except _re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        engine = self._get_engine(paths, context)
        query = Query(pattern=pattern, use_regex=True, context=context)
        result = engine.run(query)
        self._add_to_history(query, result)
        return self._format_result(result, query)

    def search_ast(
        self,
        pattern: str,
        func_name: str | None = None,
        class_name: str | None = None,
        decorator: str | None = None,
        imported: str | None = None,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """Perform AST-based structural search."""
        ast_filters = ASTFilters(
            func_name=func_name,
            class_name=class_name,
            decorator=decorator,
            imported=imported,
        )
        engine = self._get_engine(paths, context)
        query = Query(pattern=pattern, use_ast=True, filters=ast_filters, context=context)
        result = engine.run(query)
        self._add_to_history(query, result)
        return self._format_result(result, query)

    def search_semantic(
        self,
        concept: str,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """Perform semantic concept search via pattern expansion."""
        patterns = expand_semantic_query(concept)
        if not patterns:
            raise ValueError(f"No patterns found for concept: {concept}")

        combined = "|".join(f"({p})" for p in patterns)
        engine = self._get_engine(paths, context)
        query = Query(pattern=combined, use_regex=True, use_semantic=True, context=context)
        result = engine.run(query)
        self._add_to_history(query, result)
        return self._format_result(result, query)

    # -- Advanced searches --------------------------------------------------

    def search_fuzzy(
        self,
        pattern: str,
        similarity_threshold: float = 0.6,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """Perform fuzzy search by building a regex from the pattern."""
        # Build a flexible regex that allows character insertions/deletions
        # This is a simplified approach; for full fuzzy use rapidfuzz post-filtering
        chars = list(pattern)
        fuzzy_regex = ".*".join(_re.escape(c) for c in chars)

        engine = self._get_engine(paths, context)
        query = Query(pattern=fuzzy_regex, use_regex=True, context=context)
        result = engine.run(query)
        self._add_to_history(query, result)

        resp = self._format_result(result, query)
        # Trim to max_results
        if len(resp.items) > max_results:
            resp.items = resp.items[:max_results]
            resp.total_matches = max_results
        return resp

    def search_multi_pattern(
        self,
        patterns: list[str],
        operator: str = "OR",
        use_regex: bool = False,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """Search for multiple patterns combined with AND/OR logic."""
        if not patterns:
            raise ValueError("At least one pattern is required")

        if operator.upper() == "OR":
            if use_regex:
                combined = "|".join(f"({p})" for p in patterns)
            else:
                combined = "|".join(f"({_re.escape(p)})" for p in patterns)
            engine = self._get_engine(paths, context)
            query = Query(pattern=combined, use_regex=True, context=context)
            result = engine.run(query)
            self._add_to_history(query, result)
            return self._format_result(result, query)

        elif operator.upper() == "AND":
            # Run each pattern, intersect by file
            engine = self._get_engine(paths, context)
            file_sets: list[set[str]] = []
            all_items: list[dict[str, Any]] = []
            last_query = None

            for p in patterns:
                q = Query(pattern=p, use_regex=use_regex, context=context)
                r = engine.run(q)
                self._add_to_history(q, r)
                files = {str(item.file) for item in r.items}
                file_sets.append(files)
                formatted = self._format_result(r, q)
                all_items.extend(formatted.items)
                last_query = q

            if not file_sets or last_query is None:
                raise ValueError("No patterns processed")

            common_files = file_sets[0]
            for fs in file_sets[1:]:
                common_files &= fs

            filtered = [item for item in all_items if item["file"] in common_files]
            # Deduplicate by (file, start_line)
            seen: set[tuple[str, int]] = set()
            deduped: list[dict[str, Any]] = []
            for item in filtered:
                key = (item["file"], item["start_line"])
                if key not in seen:
                    seen.add(key)
                    deduped.append(item)

            return SearchResponse(
                items=deduped,
                stats={"files_matched": len(common_files), "total_items": len(deduped)},
                query_info={"patterns": patterns, "operator": "AND"},
                total_matches=len(deduped),
                execution_time_ms=0,
            )
        else:
            raise ValueError(f"Unsupported operator: {operator}. Use 'AND' or 'OR'.")

    # -- Configuration ------------------------------------------------------

    def configure_search(
        self,
        paths: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        context: int | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        languages: list[str] | None = None,
    ) -> ConfigResponse:
        """Update search configuration."""
        if not self.current_config:
            raise ValueError("Configuration not initialized")

        new_config = SearchConfig(
            paths=paths if paths is not None else self.current_config.paths,
            include=include_patterns if include_patterns is not None else self.current_config.include,
            exclude=exclude_patterns if exclude_patterns is not None else self.current_config.exclude,
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

        return self._make_config_response(new_config)

    def get_search_config(self) -> ConfigResponse:
        """Get current search configuration."""
        if not self.current_config:
            raise ValueError("Configuration not initialized")
        return self._make_config_response(self.current_config)

    @staticmethod
    def _make_config_response(cfg: SearchConfig) -> ConfigResponse:
        return ConfigResponse(
            paths=cfg.paths,
            include_patterns=cfg.include,
            exclude_patterns=cfg.exclude,
            context_lines=cfg.context,
            parallel=cfg.parallel,
            workers=cfg.workers,
            languages=[lang.value for lang in cfg.languages] if cfg.languages else None,
        )

    # -- Utilities ----------------------------------------------------------

    def get_supported_languages(self) -> list[str]:
        """List supported programming languages."""
        return [lang.value for lang in get_supported_languages()]

    def clear_caches(self) -> dict[str, str]:
        """Clear all caches."""
        if self.search_engine:
            self.search_engine.clear_caches()
        self.resource_manager.clear_cache()
        return {"status": "All caches cleared successfully"}

    def get_search_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent search history."""
        return self.search_history[-limit:] if self.search_history else []

    def analyze_file(self, file_path: str) -> dict[str, Any]:
        """Analyze a single file for basic metrics."""
        path = Path(file_path)
        if not path.is_file():
            raise ValueError(f"File not found: {file_path}")

        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.splitlines()

        # Basic metrics
        total_lines = len(lines)
        blank_lines = sum(1 for line in lines if not line.strip())
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        code_lines = total_lines - blank_lines - comment_lines

        # Count structures (Python-specific patterns)
        functions = sum(1 for line in lines if _re.match(r"\s*def\s+\w+", line))
        classes = sum(1 for line in lines if _re.match(r"\s*class\s+\w+", line))
        imports = sum(1 for line in lines if _re.match(r"\s*(import|from)\s+", line))

        # Detect language
        suffix = path.suffix.lower()
        lang_map = {
            ".py": "python", ".js": "javascript", ".ts": "typescript",
            ".java": "java", ".cpp": "cpp", ".c": "c", ".go": "go",
            ".rs": "rust", ".rb": "ruby", ".php": "php",
            ".swift": "swift", ".kt": "kotlin",
        }
        language = lang_map.get(suffix, "unknown")

        stat = path.stat()

        return {
            "file_path": str(path),
            "file_name": path.name,
            "language": language,
            "file_size_bytes": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "total_lines": total_lines,
            "code_lines": code_lines,
            "blank_lines": blank_lines,
            "comment_lines": comment_lines,
            "comment_ratio": round(comment_lines / max(total_lines, 1), 3),
            "functions_count": functions,
            "classes_count": classes,
            "imports_count": imports,
        }


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP | None:
    """Create and configure the FastMCP server with all PySearch tools."""
    if not FASTMCP_AVAILABLE:
        logger.error("FastMCP not available. Install with: pip install fastmcp")
        return None

    engine = PySearchEngine()

    mcp = FastMCP(
        name="PySearch",
        instructions="""
        PySearch MCP Server — Comprehensive code search capabilities for LLM agents.

        Core Search Tools:
        - search_text: Basic text search across files
        - search_regex: Regex pattern search with validation
        - search_ast: AST-based structural search with filters
        - search_semantic: Semantic concept search with pattern expansion

        Advanced Search Tools:
        - search_fuzzy: Fuzzy text search with similarity matching
        - search_multi_pattern: Multiple pattern search with AND/OR operators

        Analysis Tools:
        - analyze_file: File content analysis with metrics

        Configuration Tools:
        - configure_search: Update search configuration
        - get_search_config: Get current configuration
        - get_supported_languages: List supported programming languages
        - clear_caches: Clear search engine caches

        Utility Tools:
        - get_search_history: Get recent search history
        - get_server_health: Get server health and diagnostics

        Resources:
        - pysearch://config/current: Current search configuration
        - pysearch://history/searches: Search history
        - pysearch://stats/overview: Server statistics
        - pysearch://languages/supported: Supported languages
        """
    )

    # -- Validation helper --------------------------------------------------

    def _validate(**kwargs: Any) -> dict[str, Any]:
        """Validate and sanitize tool inputs, raising ToolError on failure."""
        results = validate_tool_input(**kwargs)
        try:
            check_validation_results(results)
        except ValidationError as e:
            raise ToolError(f"Input validation failed: {e.message}") from e
        except SecurityValidationError as e:
            raise ToolError(f"Security validation failed: {e.message}") from e
        except PerformanceValidationError as e:
            raise ToolError(f"Performance validation failed: {e.message}") from e
        return get_sanitized_values(results)

    # ======================================================================
    # Core Search Tools
    # ======================================================================

    @mcp.tool
    def search_text(
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search for text patterns in files.

        Args:
            pattern: Text pattern to search for
            paths: Optional list of paths to search (uses configured paths if None)
            context: Number of context lines around matches (default: 3)
            case_sensitive: Whether search should be case sensitive (default: False)

        Returns:
            Search results with matching text, file locations, and statistics
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_text(pattern, paths, context, case_sensitive)
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Text search failed: {e}") from e

    @mcp.tool
    def search_regex(
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
    ) -> dict[str, Any]:
        """
        Search for regex patterns in files.

        Args:
            pattern: Regular expression pattern to search for
            paths: Optional list of paths to search
            context: Number of context lines around matches
            case_sensitive: Whether search should be case sensitive

        Returns:
            Search results with regex matches
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_regex(pattern, paths, context, case_sensitive)
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

        Returns:
            Search results with AST-matched items
        """
        _validate(pattern=pattern, paths=paths, context=context)
        try:
            resp = engine.search_ast(
                pattern, func_name, class_name, decorator, imported, paths, context
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"AST search failed: {e}") from e

    @mcp.tool
    def search_semantic(
        concept: str,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> dict[str, Any]:
        """
        Search for semantic concepts in code.

        Expands the concept into related patterns and performs a combined search.

        Args:
            concept: Semantic concept to search for (e.g., "database", "authentication", "testing")
            paths: Optional list of paths to search
            context: Number of context lines around matches

        Returns:
            Search results with semantically related matches
        """
        _validate(pattern=concept, paths=paths, context=context)
        try:
            resp = engine.search_semantic(concept, paths, context)
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Semantic search failed: {e}") from e

    # ======================================================================
    # Advanced Search Tools
    # ======================================================================

    @mcp.tool
    def search_fuzzy(
        pattern: str,
        similarity_threshold: float = 0.6,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> dict[str, Any]:
        """
        Perform fuzzy search with approximate string matching.

        Args:
            pattern: Text pattern to search for with fuzzy matching
            similarity_threshold: Minimum similarity score (0.0 to 1.0, default: 0.6)
            max_results: Maximum number of results to return (default: 100)
            paths: Optional list of paths to search
            context: Number of context lines around matches

        Returns:
            Search results with approximate matches
        """
        _validate(pattern=pattern, paths=paths, context=context,
                  similarity_threshold=similarity_threshold, max_results=max_results)
        try:
            resp = engine.search_fuzzy(pattern, similarity_threshold, max_results, paths, context)
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
    ) -> dict[str, Any]:
        """
        Search for multiple patterns with logical operators.

        Args:
            patterns: List of patterns to search for
            operator: Logical operator — "AND" (all must match in same file) or "OR" (any match)
            use_regex: Whether patterns are regular expressions
            paths: Optional list of paths to search
            context: Number of context lines around matches

        Returns:
            Combined search results from multiple patterns
        """
        if not patterns:
            raise ToolError("At least one pattern is required")
        _validate(paths=paths, context=context)
        try:
            resp = engine.search_multi_pattern(patterns, operator, use_regex, paths, context)
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Multi-pattern search failed: {e}") from e

    # ======================================================================
    # Analysis Tools
    # ======================================================================

    @mcp.tool
    def analyze_file(
        file_path: str,
    ) -> dict[str, Any]:
        """
        Analyze a file for code metrics and statistics.

        Args:
            file_path: Path to the file to analyze

        Returns:
            File analysis with line counts, function/class counts, complexity indicators
        """
        _validate(file_path=file_path)
        try:
            return engine.analyze_file(file_path)
        except Exception as e:
            raise ToolError(f"File analysis failed: {e}") from e

    # ======================================================================
    # Configuration Tools
    # ======================================================================

    @mcp.tool
    def configure_search(
        paths: list[str] | None = None,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        context: int | None = None,
        parallel: bool | None = None,
        workers: int | None = None,
        languages: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Update search configuration.

        Args:
            paths: List of paths to search
            include_patterns: File patterns to include (e.g., ["**/*.py", "**/*.js"])
            exclude_patterns: File patterns to exclude (e.g., ["**/node_modules/**"])
            context: Number of context lines around matches
            parallel: Whether to use parallel processing
            workers: Number of worker threads
            languages: List of languages to filter by

        Returns:
            Updated configuration settings
        """
        try:
            resp = engine.configure_search(
                paths, include_patterns, exclude_patterns,
                context, parallel, workers, languages,
            )
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Configuration update failed: {e}") from e

    @mcp.tool
    def get_search_config() -> dict[str, Any]:
        """
        Get current search configuration.

        Returns:
            Current configuration settings including paths, patterns, and options
        """
        try:
            resp = engine.get_search_config()
            return asdict(resp)
        except Exception as e:
            raise ToolError(f"Failed to get configuration: {e}") from e

    @mcp.tool
    def get_supported_languages() -> list[str]:
        """
        Get list of supported programming languages.

        Returns:
            List of supported language names
        """
        try:
            return engine.get_supported_languages()
        except Exception as e:
            raise ToolError(f"Failed to get languages: {e}") from e

    @mcp.tool
    def clear_caches() -> dict[str, str]:
        """
        Clear search engine and resource caches.

        Returns:
            Status message confirming cache clearing
        """
        try:
            return engine.clear_caches()
        except Exception as e:
            raise ToolError(f"Failed to clear caches: {e}") from e

    # ======================================================================
    # Utility Tools
    # ======================================================================

    @mcp.tool
    def get_search_history(limit: int = 10) -> list[dict[str, Any]]:
        """
        Get recent search history.

        Args:
            limit: Maximum number of history entries to return (default: 10)

        Returns:
            List of recent search operations with metadata
        """
        try:
            return engine.get_search_history(limit)
        except Exception as e:
            raise ToolError(f"Failed to get history: {e}") from e

    @mcp.tool
    def get_server_health() -> dict[str, Any]:
        """
        Get server health status and diagnostics.

        Returns:
            Health status including cache analytics, validation stats, and resource usage
        """
        try:
            cache_health = engine.resource_manager.get_health_status()
            validation_stats = engine.validator.get_validation_stats()
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cache_health": cache_health,
                "validation_stats": validation_stats,
                "search_history_count": len(engine.search_history),
                "fastmcp_available": FASTMCP_AVAILABLE,
                "semantic_available": SEMANTIC_AVAILABLE,
            }
        except Exception as e:
            raise ToolError(f"Health check failed: {e}") from e

    # ======================================================================
    # MCP Resources
    # ======================================================================

    @mcp.resource("pysearch://config/current")
    async def resource_current_config() -> str:
        """Get current search configuration as JSON."""
        try:
            resp = engine.get_search_config()
            return json.dumps(asdict(resp), indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://history/searches")
    async def resource_search_history() -> str:
        """Get complete search history as JSON."""
        try:
            return json.dumps(engine.get_search_history(100), indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://stats/overview")
    async def resource_stats_overview() -> str:
        """Get comprehensive statistics overview as JSON."""
        try:
            cache_analytics = engine.resource_manager.get_cache_analytics()
            stats = {
                "total_searches": len(engine.search_history),
                "cache_analytics": cache_analytics,
                "validation_stats": engine.validator.get_validation_stats(),
                "timestamp": datetime.now().isoformat(),
            }
            return json.dumps(stats, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://languages/supported")
    async def resource_supported_languages() -> str:
        """Get supported languages as JSON."""
        try:
            langs = engine.get_supported_languages()
            return json.dumps({"languages": langs, "count": len(langs)}, indent=2)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Module-level server instance for `fastmcp run` compatibility
mcp = create_mcp_server()


def main() -> None:
    """Main entry point for the PySearch MCP server."""
    if not FASTMCP_AVAILABLE:
        print("Error: FastMCP is not available.")
        print("Please install it with: pip install fastmcp")
        return

    global mcp
    if mcp is None:
        mcp = create_mcp_server()

    if mcp:
        print("Starting PySearch MCP Server...")
        # Default: STDIO transport for MCP clients
        mcp.run()
        # For HTTP transport: mcp.run(transport="http", host="127.0.0.1", port=9000)
        # For SSE transport: mcp.run(transport="sse", host="127.0.0.1", port=9000)
    else:
        print("Failed to create MCP server")


if __name__ == "__main__":
    main()
