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
from ..shared.progress import ProgressTracker
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

    def _add_to_history(
        self,
        query: Query,
        result: SearchResult,
        *,
        session_id: str | None = None,
        search_type: str = "text",
    ) -> None:
        """Record search in history, session manager, and cache results."""
        execution_time_ms = result.stats.elapsed_ms
        result_count = result.stats.items

        # 1. Internal history
        self.search_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": {
                "pattern": query.pattern,
                "use_regex": query.use_regex,
                "use_ast": query.use_ast,
                "use_semantic": query.use_semantic,
            },
            "result_count": result_count,
            "execution_time_ms": execution_time_ms,
            "matched_files": result.stats.files_matched,
            "session_id": session_id,
            "search_type": search_type,
        })
        if len(self.search_history) > 100:
            self.search_history = self.search_history[-100:]

        # 2. Session manager — record search for context awareness & learning
        if session_id:
            self.session_manager.record_search(
                session_id=session_id,
                query_info={
                    "pattern": query.pattern,
                    "type": search_type,
                    "use_regex": query.use_regex,
                    "use_ast": query.use_ast,
                    "use_semantic": query.use_semantic,
                },
                execution_time=execution_time_ms / 1000.0,
                result_count=result_count,
                success=result_count > 0,
            )

        # 3. Resource manager — cache formatted result for repeat queries
        cache_key = f"search:{search_type}:{query.pattern}:{query.context}"
        formatted = self._format_result(result, query)
        self.resource_manager.set_cache(cache_key, asdict(formatted), ttl=300)

        # 4. Clean expired cache entries periodically
        self.resource_manager.clean_expired()

    # -- Core searches ------------------------------------------------------

    def search_text(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Perform basic text search."""
        # Check cache first
        cache_key = f"search:text:{pattern}:{context}"
        cached = self.resource_manager.get_cache(cache_key)
        if cached is not None:
            return SearchResponse(**cached)

        op_id = f"text_{int(time.time() * 1000)}"
        self.progress_tracker.start_operation(op_id, total_steps=3, description="Text search")
        try:
            self.progress_tracker.update_progress(op_id, 1, "Building query")
            engine = self._get_engine(paths, context)
            query = Query(pattern=pattern, use_regex=False, context=context)

            self.progress_tracker.update_progress(op_id, 2, "Executing search")
            result = engine.run(query)

            self._add_to_history(query, result, session_id=session_id, search_type="text")
            self.progress_tracker.complete_operation(op_id, success=True)
            return self._format_result(result, query)
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

    def search_regex(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Perform regex pattern search."""
        try:
            _re.compile(pattern)
        except _re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e

        cache_key = f"search:regex:{pattern}:{context}"
        cached = self.resource_manager.get_cache(cache_key)
        if cached is not None:
            return SearchResponse(**cached)

        op_id = f"regex_{int(time.time() * 1000)}"
        self.progress_tracker.start_operation(op_id, total_steps=3, description="Regex search")
        try:
            self.progress_tracker.update_progress(op_id, 1, "Compiling pattern")
            engine = self._get_engine(paths, context)
            query = Query(pattern=pattern, use_regex=True, context=context)

            self.progress_tracker.update_progress(op_id, 2, "Executing search")
            result = engine.run(query)

            self._add_to_history(query, result, session_id=session_id, search_type="regex")
            self.progress_tracker.complete_operation(op_id, success=True)
            return self._format_result(result, query)
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

    def search_ast(
        self,
        pattern: str,
        func_name: str | None = None,
        class_name: str | None = None,
        decorator: str | None = None,
        imported: str | None = None,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Perform AST-based structural search."""
        op_id = f"ast_{int(time.time() * 1000)}"
        self.progress_tracker.start_operation(op_id, total_steps=3, description="AST search")
        try:
            self.progress_tracker.update_progress(op_id, 1, "Building AST filters")
            ast_filters = ASTFilters(
                func_name=func_name,
                class_name=class_name,
                decorator=decorator,
                imported=imported,
            )
            engine = self._get_engine(paths, context)
            query = Query(pattern=pattern, use_ast=True, filters=ast_filters, context=context)

            self.progress_tracker.update_progress(op_id, 2, "Executing AST search")
            result = engine.run(query)

            self._add_to_history(query, result, session_id=session_id, search_type="ast")
            self.progress_tracker.complete_operation(op_id, success=True)
            return self._format_result(result, query)
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

    def search_semantic(
        self,
        concept: str,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Perform semantic concept search via pattern expansion."""
        op_id = f"semantic_{int(time.time() * 1000)}"
        self.progress_tracker.start_operation(op_id, total_steps=4, description="Semantic search")
        try:
            self.progress_tracker.update_progress(op_id, 1, "Expanding semantic query")
            patterns = expand_semantic_query(concept)
            if not patterns:
                raise ValueError(f"No patterns found for concept: {concept}")

            self.progress_tracker.update_progress(op_id, 2, "Building combined pattern")
            combined = "|".join(f"({p})" for p in patterns)
            engine = self._get_engine(paths, context)
            query = Query(pattern=combined, use_regex=True, use_semantic=True, context=context)

            self.progress_tracker.update_progress(op_id, 3, "Executing search")
            result = engine.run(query)

            self._add_to_history(query, result, session_id=session_id, search_type="semantic")
            self.progress_tracker.complete_operation(op_id, success=True)
            return self._format_result(result, query)
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

    # -- Advanced searches --------------------------------------------------

    def search_fuzzy(
        self,
        pattern: str,
        similarity_threshold: float = 0.6,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Perform fuzzy search by building a regex from the pattern."""
        op_id = f"fuzzy_{int(time.time() * 1000)}"
        self.progress_tracker.start_operation(op_id, total_steps=4, description="Fuzzy search")
        try:
            self.progress_tracker.update_progress(op_id, 1, "Building fuzzy regex")
            chars = list(pattern)
            fuzzy_regex = ".*".join(_re.escape(c) for c in chars)

            self.progress_tracker.update_progress(op_id, 2, "Executing search")
            engine = self._get_engine(paths, context)
            query = Query(pattern=fuzzy_regex, use_regex=True, context=context)
            result = engine.run(query)
            self._add_to_history(query, result, session_id=session_id, search_type="fuzzy")

            self.progress_tracker.update_progress(op_id, 3, "Filtering results")
            resp = self._format_result(result, query)
            if len(resp.items) > max_results:
                resp.items = resp.items[:max_results]
                resp.total_matches = max_results

            self.progress_tracker.complete_operation(op_id, success=True)
            return resp
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

    def search_multi_pattern(
        self,
        patterns: list[str],
        operator: str = "OR",
        use_regex: bool = False,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """Search for multiple patterns combined with AND/OR logic."""
        if not patterns:
            raise ValueError("At least one pattern is required")

        op_id = f"multi_{int(time.time() * 1000)}"
        total_steps = len(patterns) + 2
        self.progress_tracker.start_operation(op_id, total_steps=total_steps, description="Multi-pattern search")
        try:
            if operator.upper() == "OR":
                self.progress_tracker.update_progress(op_id, 1, "Combining OR patterns")
                if use_regex:
                    combined = "|".join(f"({p})" for p in patterns)
                else:
                    combined = "|".join(f"({_re.escape(p)})" for p in patterns)
                engine = self._get_engine(paths, context)
                query = Query(pattern=combined, use_regex=True, context=context)

                self.progress_tracker.update_progress(op_id, 2, "Executing combined search")
                result = engine.run(query)
                self._add_to_history(query, result, session_id=session_id, search_type="multi_pattern")
                self.progress_tracker.complete_operation(op_id, success=True)
                return self._format_result(result, query)

            elif operator.upper() == "AND":
                engine = self._get_engine(paths, context)
                file_sets: list[set[str]] = []
                all_items: list[dict[str, Any]] = []
                last_query = None

                for idx, p in enumerate(patterns):
                    self.progress_tracker.update_progress(
                        op_id, idx + 1, f"Searching pattern {idx + 1}/{len(patterns)}"
                    )
                    q = Query(pattern=p, use_regex=use_regex, context=context)
                    r = engine.run(q)
                    self._add_to_history(q, r, session_id=session_id, search_type="multi_pattern")
                    files = {str(item.file) for item in r.items}
                    file_sets.append(files)
                    formatted = self._format_result(r, q)
                    all_items.extend(formatted.items)
                    last_query = q

                if not file_sets or last_query is None:
                    raise ValueError("No patterns processed")

                self.progress_tracker.update_progress(op_id, total_steps - 1, "Intersecting results")
                common_files = file_sets[0]
                for fs in file_sets[1:]:
                    common_files &= fs

                filtered = [item for item in all_items if item["file"] in common_files]
                seen: set[tuple[str, int]] = set()
                deduped: list[dict[str, Any]] = []
                for item in filtered:
                    key = (item["file"], item["start_line"])
                    if key not in seen:
                        seen.add(key)
                        deduped.append(item)

                self.progress_tracker.complete_operation(op_id, success=True)
                return SearchResponse(
                    items=deduped,
                    stats={"files_matched": len(common_files), "total_items": len(deduped)},
                    query_info={"patterns": patterns, "operator": "AND"},
                    total_matches=len(deduped),
                    execution_time_ms=0,
                )
            else:
                raise ValueError(f"Unsupported operator: {operator}. Use 'AND' or 'OR'.")
        except Exception:
            self.progress_tracker.complete_operation(op_id, success=False)
            raise

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

    def clear_caches(self) -> dict[str, Any]:
        """Clear all caches, optimize resources, and cleanup stale data."""
        cleared: dict[str, Any] = {}

        # Clear search engine caches
        if self.search_engine:
            self.search_engine.clear_caches()
            cleared["search_engine"] = "cleared"

        # Optimize then clear resource manager cache
        self.resource_manager.optimize_cache()
        expired_count = self.resource_manager.clean_expired()
        cache_info = self.resource_manager.get_cache_analytics()
        self.resource_manager.clear_cache()
        cleared["resource_cache"] = {
            "expired_cleaned": expired_count,
            "analytics_before_clear": cache_info,
        }

        # Cleanup completed progress operations
        self.progress_tracker.cleanup_completed()
        cleared["progress_operations"] = "cleaned"

        # Cleanup old sessions (older than 24 hours)
        sessions_cleaned = self.session_manager.cleanup_old_sessions(max_age_hours=24)
        cleared["sessions_cleaned"] = sessions_cleaned

        cleared["status"] = "All caches and stale data cleared successfully"
        return cleared

    def get_search_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent search history."""
        return self.search_history[-limit:] if self.search_history else []

    def suggest_corrections(
        self,
        word: str,
        max_suggestions: int = 10,
        algorithm: str = "damerau_levenshtein",
        paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Suggest spelling corrections based on codebase identifiers."""
        engine = self._get_engine(paths)
        suggestions = engine.suggest_corrections(
            word=word,
            max_suggestions=max_suggestions,
            algorithm=algorithm,
        )
        return [
            {"identifier": s, "similarity": round(score, 4)}
            for s, score in suggestions
        ]

    def word_level_fuzzy_search(
        self,
        pattern: str,
        max_distance: int = 2,
        min_similarity: float = 0.6,
        algorithms: list[str] | None = None,
        max_results: int = 100,
        paths: list[str] | None = None,
        context: int = 3,
    ) -> SearchResponse:
        """Perform word-level fuzzy search using actual similarity algorithms."""
        engine = self._get_engine(paths, context)
        result = engine.word_level_fuzzy_search(
            pattern=pattern,
            max_distance=max_distance,
            min_similarity=min_similarity,
            algorithms=algorithms,
            max_results=max_results,
            context=context,
        )
        query = Query(pattern=pattern, use_regex=False, context=context)
        self._add_to_history(query, result)
        return self._format_result(result, query)

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
        - suggest_corrections: Spelling corrections based on codebase identifiers
        - search_word_fuzzy: Word-level fuzzy search with similarity algorithms

        Analysis Tools:
        - analyze_file: File content analysis with metrics

        Configuration Tools:
        - configure_search: Update search configuration
        - get_search_config: Get current configuration
        - get_supported_languages: List supported programming languages
        - clear_caches: Clear search engine caches and stale data

        Utility Tools:
        - get_search_history: Get recent search history
        - get_server_health: Get server health and diagnostics

        Session Management Tools:
        - create_session: Create a context-aware search session
        - get_session_info: Get session details, intent, and recommendations

        Progress Tracking Tools:
        - get_operation_progress: Query progress of running operations
        - cancel_operation: Cancel a running operation

        All search tools accept an optional session_id for context tracking.

        Resources:
        - pysearch://config/current: Current search configuration
        - pysearch://history/searches: Search history
        - pysearch://stats/overview: Server statistics with session and progress data
        - pysearch://sessions/analytics: Session management analytics
        - pysearch://languages/supported: Supported languages
        """
    )

    # -- Validation helper --------------------------------------------------

    def _validate(*, is_regex: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Validate and sanitize tool inputs, raising ToolError on failure.

        Args:
            is_regex: If True, use regex-specific pattern validation.
            **kwargs: Tool parameters to validate.
        """
        # Rate limiting — check before spending resources on validation
        rate_result = engine.validator.check_rate_limit("mcp_client")
        if not rate_result.is_valid:
            raise ToolError(
                f"Rate limit exceeded: {rate_result.errors[0].message}"
                if rate_result.errors else "Rate limit exceeded"
            )

        # Use regex-specific validation when applicable
        if is_regex and "pattern" in kwargs:
            regex_result = engine.validator.validate_regex_pattern(kwargs["pattern"])
            engine.validator.record_validation("regex_pattern", regex_result)
            if not regex_result.is_valid:
                errors_msg = "; ".join(e.message for e in regex_result.errors)
                raise ToolError(f"Regex validation failed: {errors_msg}")
            # Replace pattern with sanitized value for downstream
            kwargs["pattern"] = regex_result.sanitized_value

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
            resp = engine.search_text(pattern, paths, context, case_sensitive, session_id=session_id)
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
            resp = engine.search_regex(pattern, paths, context, case_sensitive, session_id=session_id)
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
                pattern, func_name, class_name, decorator, imported, paths, context,
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
        _validate(pattern=pattern, paths=paths, context=context,
                  similarity_threshold=similarity_threshold, max_results=max_results)
        try:
            resp = engine.search_fuzzy(
                pattern, similarity_threshold, max_results, paths, context, session_id=session_id,
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
                patterns, operator, use_regex, paths, context, session_id=session_id,
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
    def clear_caches() -> dict[str, Any]:
        """
        Clear search engine caches, optimize resources, and cleanup stale sessions.

        Returns:
            Detailed status including cache analytics before clearing, sessions cleaned,
            and progress operations cleaned
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
            Comprehensive health status including cache analytics, validation stats,
            session analytics, active operations, and memory usage
        """
        try:
            cache_health = engine.resource_manager.get_health_status()
            validation_stats = engine.validator.get_validation_stats()
            session_analytics = engine.session_manager.get_session_analytics()
            memory_usage = engine.resource_manager.get_memory_usage()
            active_ops = {
                op_id: {
                    "status": op.status.value,
                    "progress": op.progress,
                    "current_step": op.current_step,
                    "elapsed_time": op.elapsed_time,
                }
                for op_id, op in engine.progress_tracker.active_operations.items()
                if op.status.value == "running"
            }
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cache_health": cache_health,
                "memory_usage": memory_usage,
                "validation_stats": validation_stats,
                "session_analytics": session_analytics,
                "active_operations": active_ops,
                "active_operations_count": len(active_ops),
                "search_history_count": len(engine.search_history),
                "fastmcp_available": FASTMCP_AVAILABLE,
                "semantic_available": SEMANTIC_AVAILABLE,
            }
        except Exception as e:
            raise ToolError(f"Health check failed: {e}") from e

    # ======================================================================
    # Session Management Tools
    # ======================================================================

    @mcp.tool
    def create_session(
        user_id: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """
        Create a new search session for context-aware search tracking.

        Sessions track search patterns, infer user intent, and provide
        contextual recommendations across multiple searches.

        Args:
            user_id: Optional user identifier for personalized experience
            priority: Session priority — "low", "normal", "high", "critical" (default: "normal")

        Returns:
            Session info including session_id, creation time, and initial state
        """
        from ..shared.session_manager import SessionPriority
        try:
            prio = SessionPriority(priority)
        except ValueError:
            prio = SessionPriority.NORMAL

        try:
            session = engine.session_manager.create_session(user_id=user_id, priority=prio)
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "priority": session.priority.value,
                "status": "created",
            }
        except Exception as e:
            raise ToolError(f"Session creation failed: {e}") from e

    @mcp.tool
    def get_session_info(
        session_id: str,
    ) -> dict[str, Any]:
        """
        Get detailed information about a search session.

        Returns session state, search history within the session, inferred intent,
        contextual recommendations, and analytics.

        Args:
            session_id: The session ID to query

        Returns:
            Session details including context, intent, recommendations, and analytics
        """
        _validate(session_id=session_id)
        try:
            session = engine.session_manager.get_session(session_id)
            if session is None:
                raise ToolError(f"Session not found: {session_id}")

            recommendations = engine.session_manager.get_contextual_recommendations(session_id)
            context = session.get_current_context()
            suggestions = session.get_contextual_suggestions()

            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "priority": session.priority.value,
                "intent": session.intent.value,
                "total_searches": session.total_searches,
                "successful_searches": session.successful_searches,
                "avg_search_time": session.avg_search_time,
                "search_focus": session.search_focus,
                "recent_patterns": session.recent_patterns[-10:],
                "patterns_discovered_count": len(session.patterns_discovered),
                "current_files_count": len(session.current_files),
                "context": context,
                "suggestions": suggestions,
                "recommendations": recommendations,
            }
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to get session info: {e}") from e

    # ======================================================================
    # Progress Tracking Tools
    # ======================================================================

    @mcp.tool
    def get_operation_progress(
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Get progress of active operations.

        If operation_id is provided, returns progress for that specific operation.
        Otherwise, returns a summary of all active operations.

        Args:
            operation_id: Optional specific operation ID to query

        Returns:
            Progress information including status, completion percentage, and timing
        """
        try:
            if operation_id:
                _validate(operation_id=operation_id)
                status = engine.progress_tracker.get_operation_status(operation_id)
                if status is None:
                    return {"error": f"Operation not found: {operation_id}"}
                return {
                    "operation_id": status.operation_id,
                    "status": status.status.value,
                    "progress": status.progress,
                    "current_step": status.current_step,
                    "total_steps": status.total_steps,
                    "completed_steps": status.completed_steps,
                    "elapsed_time": status.elapsed_time,
                    "estimated_remaining": status.estimated_remaining,
                    "details": status.details,
                }
            else:
                ops = {}
                for op_id, op in engine.progress_tracker.active_operations.items():
                    ops[op_id] = {
                        "status": op.status.value,
                        "progress": op.progress,
                        "current_step": op.current_step,
                        "elapsed_time": op.elapsed_time,
                    }
                return {
                    "active_operations_count": len(ops),
                    "operations": ops,
                }
        except Exception as e:
            raise ToolError(f"Failed to get operation progress: {e}") from e

    @mcp.tool
    def cancel_operation(
        operation_id: str,
    ) -> dict[str, Any]:
        """
        Cancel a running operation.

        Args:
            operation_id: The operation ID to cancel

        Returns:
            Cancellation status
        """
        _validate(operation_id=operation_id)
        try:
            success = engine.progress_tracker.cancel_operation(operation_id)
            if success:
                return {
                    "operation_id": operation_id,
                    "status": "cancelled",
                    "message": f"Operation {operation_id} has been cancelled",
                }
            else:
                return {
                    "operation_id": operation_id,
                    "status": "not_found",
                    "message": f"Operation {operation_id} not found or already completed",
                }
        except Exception as e:
            raise ToolError(f"Failed to cancel operation: {e}") from e

    # ======================================================================
    # IDE Integration Tools
    # ======================================================================

    @mcp.tool
    def ide_jump_to_definition(
        file_path: str,
        line: int,
        symbol: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Find the definition location of a symbol (function, class, variable).

        Args:
            file_path: File where the symbol is referenced
            line: Line number where the symbol appears
            symbol: The identifier to look up
            paths: Optional search paths (default: current config paths)

        Returns:
            Definition location with file, line, symbol_name, and symbol_type
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            result = eng.jump_to_definition(file_path, line, symbol)
            if result is None:
                return {"found": False, "message": f"No definition found for '{symbol}'"}
            return {"found": True, **result}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Jump to definition failed: {e}") from e

    @mcp.tool
    def ide_find_references(
        file_path: str,
        line: int,
        symbol: str,
        include_definition: bool = True,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Find all references to a symbol across the codebase.

        Args:
            file_path: Originating file
            line: Originating line number
            symbol: The identifier to search for
            include_definition: Whether to include the definition itself
            paths: Optional search paths

        Returns:
            List of reference locations with file, line, context, and is_definition flag
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            refs = eng.find_references(file_path, line, symbol, include_definition)
            return {"symbol": symbol, "references": refs, "count": len(refs)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Find references failed: {e}") from e

    @mcp.tool
    def ide_completion(
        file_path: str,
        line: int,
        column: int,
        prefix: str = "",
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Provide auto-completion suggestions for the given cursor position.

        Args:
            file_path: Current file
            line: Cursor line number
            column: Cursor column number
            prefix: Partially typed identifier
            paths: Optional search paths

        Returns:
            List of completion items with label, kind, and detail
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            items = eng.provide_completion(file_path, line, column, prefix)
            return {"prefix": prefix, "completions": items, "count": len(items)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Completion failed: {e}") from e

    @mcp.tool
    def ide_hover(
        file_path: str,
        line: int,
        column: int,
        symbol: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Provide hover information for a symbol (type, docstring, signature).

        Args:
            file_path: Current file
            line: Cursor line number
            column: Cursor column number
            symbol: The hovered identifier
            paths: Optional search paths

        Returns:
            Hover information with symbol_name, symbol_type, contents, and documentation
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            info = eng.provide_hover(file_path, line, column, symbol)
            if info is None:
                return {"found": False, "message": f"No hover info for '{symbol}'"}
            return {"found": True, **info}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Hover failed: {e}") from e

    @mcp.tool
    def ide_document_symbols(
        file_path: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        List all symbols (functions, classes, variables) in a file.

        Args:
            file_path: Path to the file to analyze
            paths: Optional search paths

        Returns:
            List of symbols with name, kind, and line number
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            symbols = eng.get_document_symbols(file_path)
            return {"file": file_path, "symbols": symbols, "count": len(symbols)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Document symbols failed: {e}") from e

    @mcp.tool
    def ide_workspace_symbols(
        query: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search for symbols across the entire workspace.

        Args:
            query: Filter string for symbol names (minimum 2 characters)
            paths: Optional search paths

        Returns:
            List of matching symbols with name, kind, line, and file detail
        """
        _validate(pattern=query, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            symbols = eng.get_workspace_symbols(query)
            return {"query": query, "symbols": symbols, "count": len(symbols)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Workspace symbols failed: {e}") from e

    @mcp.tool
    def ide_diagnostics(
        file_path: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run lightweight diagnostics on a file (TODO/FIXME/HACK markers, circular imports).

        Args:
            file_path: The file to diagnose
            paths: Optional search paths

        Returns:
            List of diagnostics with line, severity, message, and code
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            diags = eng.get_diagnostics(file_path)
            return {"file": file_path, "diagnostics": diags, "count": len(diags)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Diagnostics failed: {e}") from e

    # ======================================================================
    # Distributed Indexing Tools
    # ======================================================================

    @mcp.tool
    def distributed_enable(
        num_workers: int | None = None,
        max_queue_size: int = 10000,
    ) -> dict[str, Any]:
        """
        Enable distributed indexing for large codebases.

        Args:
            num_workers: Number of worker processes (defaults to min(cpu_count, 8))
            max_queue_size: Maximum work queue size

        Returns:
            Status of distributed indexing enablement
        """
        try:
            eng = engine._get_engine()
            success = eng.enable_distributed_indexing(
                num_workers=num_workers, max_queue_size=max_queue_size
            )
            return {
                "enabled": success,
                "num_workers": num_workers or "auto",
                "max_queue_size": max_queue_size,
            }
        except Exception as e:
            raise ToolError(f"Failed to enable distributed indexing: {e}") from e

    @mcp.tool
    def distributed_disable() -> dict[str, Any]:
        """
        Disable distributed indexing and stop all workers.

        Returns:
            Confirmation of distributed indexing disablement
        """
        try:
            eng = engine._get_engine()
            eng.disable_distributed_indexing()
            return {"enabled": False, "message": "Distributed indexing disabled"}
        except Exception as e:
            raise ToolError(f"Failed to disable distributed indexing: {e}") from e

    @mcp.tool
    def distributed_status() -> dict[str, Any]:
        """
        Get the current status of distributed indexing including worker and queue stats.

        Returns:
            Comprehensive status including enabled state, queue stats, and worker info
        """
        try:
            eng = engine._get_engine()
            enabled = eng.is_distributed_indexing_enabled()
            result: dict[str, Any] = {"enabled": enabled}
            if enabled:
                result["queue_stats"] = eng.get_distributed_queue_stats()
            return result
        except Exception as e:
            raise ToolError(f"Failed to get distributed status: {e}") from e

    # ======================================================================
    # Multi-Repository Search Tools
    # ======================================================================

    @mcp.tool
    def multi_repo_enable(
        max_workers: int = 4,
    ) -> dict[str, Any]:
        """
        Enable multi-repository search capabilities.

        Args:
            max_workers: Maximum number of parallel workers for searches

        Returns:
            Status of multi-repo enablement
        """
        try:
            eng = engine._get_engine()
            success = eng.enable_multi_repo(max_workers=max_workers)
            return {"enabled": success, "max_workers": max_workers}
        except Exception as e:
            raise ToolError(f"Failed to enable multi-repo: {e}") from e

    @mcp.tool
    def multi_repo_add(
        name: str,
        path: str,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """
        Add a repository to multi-repository search.

        Args:
            name: Unique name for the repository
            path: Filesystem path to the repository
            priority: Priority level — "high", "normal", "low"

        Returns:
            Result of adding the repository
        """
        _validate(paths=[path])
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                eng.enable_multi_repo()
            success = eng.add_repository(name, path, priority=priority)
            return {"added": success, "name": name, "path": path, "priority": priority}
        except Exception as e:
            raise ToolError(f"Failed to add repository: {e}") from e

    @mcp.tool
    def multi_repo_remove(
        name: str,
    ) -> dict[str, Any]:
        """
        Remove a repository from multi-repository search.

        Args:
            name: Name of the repository to remove

        Returns:
            Result of removing the repository
        """
        try:
            eng = engine._get_engine()
            success = eng.remove_repository(name)
            return {"removed": success, "name": name}
        except Exception as e:
            raise ToolError(f"Failed to remove repository: {e}") from e

    @mcp.tool
    def multi_repo_list() -> dict[str, Any]:
        """
        List all repositories in the multi-repository search system.

        Returns:
            List of repository names and enabled status
        """
        try:
            eng = engine._get_engine()
            enabled = eng.is_multi_repo_enabled()
            repos = eng.list_repositories() if enabled else []
            return {"enabled": enabled, "repositories": repos, "count": len(repos)}
        except Exception as e:
            raise ToolError(f"Failed to list repositories: {e}") from e

    @mcp.tool
    def multi_repo_search(
        pattern: str,
        use_regex: bool = False,
        use_ast: bool = False,
        context: int = 2,
        max_results: int = 1000,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """
        Search across all repositories in the multi-repository system.

        Args:
            pattern: Search pattern
            use_regex: Whether to use regex matching
            use_ast: Whether to use AST-based matching
            context: Number of context lines around matches
            max_results: Maximum total results
            timeout: Timeout per repository search in seconds

        Returns:
            Aggregated search results from all repositories
        """
        _validate(pattern=pattern)
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"error": "Multi-repo not enabled. Call multi_repo_enable first."}
            result = eng.search_all_repositories(
                pattern=pattern,
                use_regex=use_regex,
                use_ast=use_ast,
                context=context,
                max_results=max_results,
                timeout=timeout,
            )
            if result is None:
                return {"total_matches": 0, "repositories": {}}
            return {
                "total_matches": result.total_matches,
                "total_repositories": result.total_repositories,
                "successful_repositories": result.successful_repositories,
                "failed_repositories": result.failed_repositories,
                "success_rate": result.success_rate,
                "repository_results": {
                    name: {
                        "items": len(sr.items),
                        "files_matched": sr.stats.files_matched,
                        "elapsed_ms": sr.stats.elapsed_ms,
                    }
                    for name, sr in result.repository_results.items()
                },
            }
        except Exception as e:
            raise ToolError(f"Multi-repo search failed: {e}") from e

    @mcp.tool
    def multi_repo_health() -> dict[str, Any]:
        """
        Get health status for all repositories in the multi-repository system.

        Returns:
            Health information including per-repository status and overall summary
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.get_multi_repo_health()
        except Exception as e:
            raise ToolError(f"Failed to get multi-repo health: {e}") from e

    @mcp.tool
    def multi_repo_stats() -> dict[str, Any]:
        """
        Get search performance statistics for the multi-repository system.

        Returns:
            Statistics including total searches, average search time, and pattern history
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.get_multi_repo_stats()
        except Exception as e:
            raise ToolError(f"Failed to get multi-repo stats: {e}") from e

    @mcp.tool
    def multi_repo_sync() -> dict[str, Any]:
        """
        Synchronize all repositories (refresh status and health).

        Returns:
            Dictionary mapping repository names to sync success status
        """
        try:
            eng = engine._get_engine()
            if not eng.is_multi_repo_enabled():
                return {"enabled": False, "message": "Multi-repo not enabled"}
            return eng.sync_repositories()
        except Exception as e:
            raise ToolError(f"Failed to sync repositories: {e}") from e

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
            memory_usage = engine.resource_manager.get_memory_usage()
            session_analytics = engine.session_manager.get_session_analytics()
            active_ops_count = len([
                op for op in engine.progress_tracker.active_operations.values()
                if op.status.value == "running"
            ])
            stats = {
                "total_searches": len(engine.search_history),
                "cache_analytics": cache_analytics,
                "memory_usage": memory_usage,
                "validation_stats": engine.validator.get_validation_stats(),
                "session_analytics": session_analytics,
                "active_operations_count": active_ops_count,
                "timestamp": datetime.now().isoformat(),
            }
            return json.dumps(stats, indent=2, default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.resource("pysearch://sessions/analytics")
    async def resource_sessions_analytics() -> str:
        """Get session management analytics as JSON."""
        try:
            analytics = engine.session_manager.get_session_analytics()
            return json.dumps(analytics, indent=2, default=str)
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
