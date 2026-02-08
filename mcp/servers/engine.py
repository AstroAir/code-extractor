#!/usr/bin/env python3
"""
PySearch MCP Engine — Core search engine wrapper and data structures.

Contains the PySearchEngine class that manages the PySearch engine lifecycle,
search execution, configuration, history tracking, and integration with
shared MCP utilities.
"""

from __future__ import annotations

import re as _re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pysearch import PySearch, SearchConfig, get_supported_languages
from pysearch.core.types import (
    ASTFilters,
    Language,
    Query,
    SearchResult,
)

from ..shared.progress import ProgressTracker
from ..shared.resource_manager import ResourceManager
from ..shared.session_manager import EnhancedSessionManager, get_session_manager
from ..shared.validation import InputValidator

# Semantic expansion (optional, graceful fallback)
try:
    from pysearch.semantic import expand_semantic_query

    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

    def expand_semantic_query(concept: str) -> list[str]:  # type: ignore[misc]
        """Fallback: return concept as-is when semantic module unavailable."""
        return [concept]


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
