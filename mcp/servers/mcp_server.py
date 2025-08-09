#!/usr/bin/env python3
"""
PySearch MCP Server

An MCP (Model Context Protocol) server that exposes advanced PySearch functionality
including fuzzy search, multi-pattern search, file analysis, and advanced filtering.

This server provides comprehensive code search capabilities including:
- Fuzzy search with configurable similarity thresholds
- Multi-pattern search with logical operators
- File content analysis and complexity metrics
- Advanced search result ranking
- Comprehensive filtering capabilities
- MCP resource management
- Progress reporting for long operations
- Context-aware search sessions
- Prompt templates for common scenarios
- Composition support for chaining operations

Usage:
    python mcp_server.py

The server runs on stdio transport by default, suitable for MCP clients.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# PySearch imports
from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import (
    Query,
    SearchItem,
)

# Import base server functionality
from .basic_mcp_server import BasicPySearchMCPServer, SearchResponse

# Fuzzy search imports (will be available once dependencies are installed)
try:
    import rapidfuzz
    from rapidfuzz import fuzz, process

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print(
        "Warning: Fuzzy search libraries not available. Install rapidfuzz for fuzzy search support."
    )


class SearchOperator(Enum):
    """Logical operators for multi-pattern search."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


class RankingFactor(Enum):
    """Factors used for search result ranking."""

    PATTERN_MATCH_QUALITY = "pattern_match_quality"
    FILE_IMPORTANCE = "file_importance"
    CONTEXT_RELEVANCE = "context_relevance"
    RECENCY = "recency"
    FILE_SIZE = "file_size"
    LANGUAGE_PRIORITY = "language_priority"


@dataclass
class FuzzySearchConfig:
    """Configuration for fuzzy search operations."""

    similarity_threshold: float = 0.6  # Minimum similarity score (0.0 to 1.0)
    max_results: int = 100  # Maximum number of fuzzy matches to return
    algorithm: str = (
        "ratio"  # Fuzzy matching algorithm: ratio, partial_ratio, token_sort_ratio, token_set_ratio
    )
    case_sensitive: bool = False


@dataclass
class MultiPatternQuery:
    """Query configuration for multi-pattern search."""

    patterns: list[str]
    operator: SearchOperator = SearchOperator.OR
    use_regex: bool = False
    use_fuzzy: bool = False
    fuzzy_config: FuzzySearchConfig | None = None


@dataclass
class FileAnalysisResult:
    """Result of file content analysis."""

    file_path: str
    file_size: int
    line_count: int
    complexity_score: float
    language: str | None
    functions_count: int
    classes_count: int
    imports_count: int
    comments_ratio: float
    code_quality_score: float
    last_modified: datetime
    author: str | None = None


@dataclass
class SearchFilter:
    """Advanced search filtering options."""

    min_file_size: int | None = None
    max_file_size: int | None = None
    modified_after: datetime | None = None
    modified_before: datetime | None = None
    authors: list[str] | None = None
    languages: list[str] | None = None
    min_complexity: float | None = None
    max_complexity: float | None = None
    file_extensions: list[str] | None = None
    exclude_patterns: list[str] | None = None


@dataclass
class RankedSearchResult:
    """Search result with ranking information."""

    item: dict[str, Any]
    relevance_score: float
    ranking_factors: dict[RankingFactor, float]


@dataclass
class SearchSession:
    """Context for maintaining search sessions."""

    session_id: str
    created_at: datetime
    last_accessed: datetime
    queries: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    cached_results: dict[str, Any] = field(default_factory=dict)


class PySearchMCPServer(BasicPySearchMCPServer):
    """
    Advanced MCP Server wrapper for PySearch functionality.

    Extends the basic PySearchMCPServer with advanced features:
    - Fuzzy search capabilities
    - Multi-pattern search with logical operators
    - File content analysis and metrics
    - Advanced result ranking and filtering
    - Search session management
    - Progress reporting for long operations
    """

    def __init__(self, name: str = "Enhanced PySearch MCP Server"):
        super().__init__(name)
        self.search_sessions: dict[str, SearchSession] = {}
        self.file_analysis_cache: dict[str, FileAnalysisResult] = {}
        self.ranking_weights: dict[RankingFactor, float] = {
            RankingFactor.PATTERN_MATCH_QUALITY: 0.4,
            RankingFactor.FILE_IMPORTANCE: 0.2,
            RankingFactor.CONTEXT_RELEVANCE: 0.2,
            RankingFactor.RECENCY: 0.1,
            RankingFactor.FILE_SIZE: 0.05,
            RankingFactor.LANGUAGE_PRIORITY: 0.05,
        }

    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = str(time.time())
        return hashlib.md5(timestamp.encode()).hexdigest()[:12]

    def _get_or_create_session(self, session_id: str | None = None) -> SearchSession:
        """Get existing session or create a new one."""
        if session_id and session_id in self.search_sessions:
            session = self.search_sessions[session_id]
            session.last_accessed = datetime.now()
            return session

        new_session_id = session_id or self._generate_session_id()
        session = SearchSession(
            session_id=new_session_id, created_at=datetime.now(), last_accessed=datetime.now()
        )
        self.search_sessions[new_session_id] = session
        return session

    async def search_fuzzy(
        self,
        pattern: str,
        paths: list[str] | None = None,
        config: FuzzySearchConfig | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """
        Perform fuzzy search with configurable similarity thresholds.

        Args:
            pattern: Text pattern to search for with fuzzy matching
            paths: Optional list of paths to search
            config: Fuzzy search configuration
            context: Number of context lines around matches
            session_id: Optional session ID for context management

        Returns:
            SearchResponse with fuzzy matching results
        """
        if not FUZZY_AVAILABLE:
            raise ValueError("Fuzzy search not available. Install rapidfuzz library.")

        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Use default fuzzy config if not provided
        fuzzy_config = config or FuzzySearchConfig()

        # Get or create session
        session = self._get_or_create_session(session_id)

        # First, get all text content from files
        temp_engine = self._get_search_engine(paths, context)

        # Perform a broad search to get candidate text
        broad_query = Query(pattern=".", use_regex=True, context=context)
        broad_result = temp_engine.run(broad_query)

        # Apply fuzzy matching to the results
        fuzzy_matches = []
        algorithm_func = getattr(fuzz, fuzzy_config.algorithm, fuzz.ratio)

        for item in broad_result.items:
            for line_idx, line in enumerate(item.lines):
                if not fuzzy_config.case_sensitive:
                    line_lower = line.lower()
                    pattern_lower = pattern.lower()
                else:
                    line_lower = line
                    pattern_lower = pattern

                # Calculate fuzzy similarity
                similarity = algorithm_func(pattern_lower, line_lower) / 100.0

                if similarity >= fuzzy_config.similarity_threshold:
                    fuzzy_item = {
                        "file": str(item.file),
                        "start_line": item.start_line + line_idx,
                        "end_line": item.start_line + line_idx,
                        "lines": [line],
                        "match_spans": [(0, (0, len(line)))],
                        "score": similarity,
                        "fuzzy_similarity": similarity,
                    }
                    fuzzy_matches.append(fuzzy_item)

        # Sort by similarity score and limit results
        fuzzy_matches.sort(key=lambda x: x["fuzzy_similarity"], reverse=True)
        fuzzy_matches = fuzzy_matches[: fuzzy_config.max_results]

        # Create response
        response = SearchResponse(
            items=fuzzy_matches,
            stats={
                "files_scanned": broad_result.stats.files_scanned,
                "files_matched": len(set(item["file"] for item in fuzzy_matches)),
                "total_items": len(fuzzy_matches),
                "elapsed_ms": broad_result.stats.elapsed_ms,
                "indexed_files": broad_result.stats.indexed_files,
            },
            query_info={
                "pattern": pattern,
                "use_regex": False,
                "use_ast": False,
                "use_semantic": False,
                "use_fuzzy": True,
                "context": context,
                "fuzzy_config": asdict(fuzzy_config),
                "session_id": session.session_id,
            },
            total_matches=len(fuzzy_matches),
            execution_time_ms=broad_result.stats.elapsed_ms,
        )

        # Add to session history
        session.queries.append(
            {
                "type": "fuzzy_search",
                "pattern": pattern,
                "config": asdict(fuzzy_config),
                "result_count": len(fuzzy_matches),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return response

    async def search_text(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        case_sensitive: bool = False,
        session_id: str | None = None,
    ) -> SearchResponse:
        """
        Enhanced search_text with session support.

        Args:
            pattern: Text pattern to search for
            paths: Optional list of paths to search
            context: Number of context lines around matches
            case_sensitive: Whether search should be case sensitive
            session_id: Optional session ID for context management

        Returns:
            SearchResponse with matching results
        """
        # Get or create session if session_id provided
        if session_id:
            session = self._get_or_create_session(session_id)

        # Call parent method
        result = await super().search_text(pattern, paths, context, case_sensitive)

        # Add session info to query_info if session provided
        if session_id:
            result.query_info["session_id"] = session_id
            # Add to session history
            session.queries.append(
                {
                    "type": "text_search",
                    "pattern": pattern,
                    "result_count": result.total_matches,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return result

    async def search_multi_pattern(
        self,
        query: MultiPatternQuery,
        paths: list[str] | None = None,
        context: int = 3,
        session_id: str | None = None,
    ) -> SearchResponse:
        """
        Perform multi-pattern search with logical operators.

        Args:
            query: Multi-pattern query configuration
            paths: Optional list of paths to search
            context: Number of context lines around matches
            session_id: Optional session ID for context management

        Returns:
            SearchResponse with combined results from multiple patterns
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        if not query.patterns:
            raise ValueError("At least one pattern is required")

        # Get or create session
        session = self._get_or_create_session(session_id)

        temp_engine = self._get_search_engine(paths, context)

        # Execute searches for each pattern
        pattern_results = []
        total_time = 0.0

        for pattern in query.patterns:
            if query.use_fuzzy and FUZZY_AVAILABLE:
                # Use fuzzy search for this pattern
                fuzzy_config = query.fuzzy_config or FuzzySearchConfig()
                result = await self.search_fuzzy(pattern, paths, fuzzy_config, context, session_id)
                pattern_results.append(result.items)
                total_time += result.execution_time_ms
            else:
                # Use regular or regex search
                search_query = Query(pattern=pattern, use_regex=query.use_regex, context=context)
                search_result = temp_engine.run(search_query)

                # Convert to dict format
                items = []
                for search_item in search_result.items:
                    items.append(
                        {
                            "file": str(search_item.file),
                            "start_line": search_item.start_line,
                            "end_line": search_item.end_line,
                            "lines": search_item.lines,
                            "match_spans": search_item.match_spans,
                            "score": getattr(search_item, "score", None),
                            "pattern": pattern,
                        }
                    )

                pattern_results.append(items)
                total_time += search_result.stats.elapsed_ms

        # Combine results based on operator
        combined_items = self._combine_pattern_results(pattern_results, query.operator)

        # Calculate combined statistics
        all_files: set[str] = set()
        for items in pattern_results:
            all_files.update(item["file"] for item in items)

        response = SearchResponse(
            items=combined_items,
            stats={
                "files_scanned": len(all_files),
                "files_matched": len(set(item["file"] for item in combined_items)),
                "total_items": len(combined_items),
                "elapsed_ms": total_time,
                "indexed_files": len(all_files),
            },
            query_info={
                "patterns": query.patterns,
                "operator": query.operator.value,
                "use_regex": query.use_regex,
                "use_fuzzy": query.use_fuzzy,
                "use_ast": False,
                "use_semantic": False,
                "context": context,
                "session_id": session.session_id,
            },
            total_matches=len(combined_items),
            execution_time_ms=total_time,
        )

        # Add to session history
        session.queries.append(
            {
                "type": "multi_pattern_search",
                "patterns": query.patterns,
                "operator": query.operator.value,
                "result_count": len(combined_items),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return response

    def _combine_pattern_results(
        self, pattern_results: list[list[dict[str, Any]]], operator: SearchOperator
    ) -> list[dict[str, Any]]:
        """
        Combine results from multiple patterns based on logical operator.

        Args:
            pattern_results: List of result lists for each pattern
            operator: Logical operator to apply

        Returns:
            Combined list of search results
        """
        if not pattern_results:
            return []

        if len(pattern_results) == 1:
            return pattern_results[0]

        if operator == SearchOperator.OR:
            # Union of all results
            combined = []
            seen_items = set()

            for results in pattern_results:
                for item in results:
                    # Create a unique key for deduplication
                    key = f"{item['file']}:{item['start_line']}:{item['end_line']}"
                    if key not in seen_items:
                        combined.append(item)
                        seen_items.add(key)

            return combined

        elif operator == SearchOperator.AND:
            # Intersection - items that appear in all pattern results
            if not pattern_results:
                return []

            # Start with first pattern results
            base_results = pattern_results[0]

            for results in pattern_results[1:]:
                # Find intersection based on file and line range overlap
                intersected = []
                for base_item in base_results:
                    for item in results:
                        if base_item["file"] == item["file"] and self._ranges_overlap(
                            (base_item["start_line"], base_item["end_line"]),
                            (item["start_line"], item["end_line"]),
                        ):
                            # Merge the items
                            merged_item = self._merge_search_items(base_item, item)
                            intersected.append(merged_item)
                            break

                base_results = intersected

            return base_results

        elif operator == SearchOperator.NOT:
            # First pattern minus subsequent patterns
            if len(pattern_results) < 2:
                return pattern_results[0] if pattern_results else []

            base_results = pattern_results[0]
            exclude_results = []

            for results in pattern_results[1:]:
                exclude_results.extend(results)

            # Remove items that appear in exclude_results
            filtered = []
            for base_item in base_results:
                should_exclude = False
                for exclude_item in exclude_results:
                    if base_item["file"] == exclude_item["file"] and self._ranges_overlap(
                        (base_item["start_line"], base_item["end_line"]),
                        (exclude_item["start_line"], exclude_item["end_line"]),
                    ):
                        should_exclude = True
                        break

                if not should_exclude:
                    filtered.append(base_item)

            return filtered

        return []

    def _ranges_overlap(self, range1: tuple[int, int], range2: tuple[int, int]) -> bool:
        """Check if two line ranges overlap."""
        return range1[0] <= range2[1] and range2[0] <= range1[1]

    def _merge_search_items(self, item1: dict[str, Any], item2: dict[str, Any]) -> dict[str, Any]:
        """Merge two search items that overlap."""
        merged = item1.copy()

        # Extend line range to cover both items
        merged["start_line"] = min(item1["start_line"], item2["start_line"])
        merged["end_line"] = max(item1["end_line"], item2["end_line"])

        # Combine lines (remove duplicates while preserving order)
        all_lines = item1["lines"] + item2["lines"]
        seen = set()
        unique_lines = []
        for line in all_lines:
            if line not in seen:
                unique_lines.append(line)
                seen.add(line)
        merged["lines"] = unique_lines

        # Combine match spans
        merged["match_spans"] = item1.get("match_spans", []) + item2.get("match_spans", [])

        # Combine patterns if present
        patterns = []
        if "pattern" in item1:
            patterns.append(item1["pattern"])
        if "pattern" in item2:
            patterns.append(item2["pattern"])
        if patterns:
            merged["patterns"] = list(set(patterns))

        return merged

    async def analyze_file_content(
        self, file_path: str, include_complexity: bool = True, include_quality_metrics: bool = True
    ) -> FileAnalysisResult:
        """
        Analyze file content for statistics, complexity, and quality metrics.

        Args:
            file_path: Path to the file to analyze
            include_complexity: Whether to calculate complexity metrics
            include_quality_metrics: Whether to calculate code quality indicators

        Returns:
            FileAnalysisResult with comprehensive file analysis
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise ValueError(f"File not found: {file_path}")

        # Check cache first
        cache_key = f"{file_path}:{file_path_obj.stat().st_mtime}"
        if cache_key in self.file_analysis_cache:
            return self.file_analysis_cache[cache_key]

        # Get file statistics
        stat_info = file_path_obj.stat()
        file_size = stat_info.st_size
        last_modified = datetime.fromtimestamp(stat_info.st_mtime)

        # Read file content
        try:
            with open(file_path_obj, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            raise ValueError(f"Error reading file {file_path}: {e}")

        lines = content.split("\n")
        line_count = len(lines)

        # Detect language
        language = self._detect_file_language(file_path_obj)

        # Count structural elements
        functions_count = self._count_functions(content, language)
        classes_count = self._count_classes(content, language)
        imports_count = self._count_imports(content, language)

        # Calculate comment ratio
        comments_ratio = self._calculate_comment_ratio(content, language)

        # Calculate complexity score
        complexity_score = 0.0
        if include_complexity:
            complexity_score = self._calculate_complexity_score(content, language)

        # Calculate code quality score
        code_quality_score = 0.0
        if include_quality_metrics:
            code_quality_score = self._calculate_quality_score(
                content, language, complexity_score, comments_ratio
            )

        # Try to get author information (simplified)
        author = self._get_file_author(file_path_obj)

        result = FileAnalysisResult(
            file_path=str(file_path_obj),
            file_size=file_size,
            line_count=line_count,
            complexity_score=complexity_score,
            language=language,
            functions_count=functions_count,
            classes_count=classes_count,
            imports_count=imports_count,
            comments_ratio=comments_ratio,
            code_quality_score=code_quality_score,
            last_modified=last_modified,
            author=author,
        )

        # Cache the result
        self.file_analysis_cache[cache_key] = result

        return result

    def _detect_file_language(self, file_path: Path) -> str | None:
        """Detect programming language from file extension."""
        extension = file_path.suffix.lower()

        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".go": "go",
            ".rs": "rust",
            ".php": "php",
            ".rb": "ruby",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".sh": "bash",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".xml": "xml",
            ".json": "json",
            ".yaml": "yaml",
            ".yml": "yaml",
        }

        return language_map.get(extension)

    def _count_functions(self, content: str, language: str | None) -> int:
        """Count function definitions in the content."""
        if not language:
            return 0

        patterns = {
            "python": r"^\s*def\s+\w+",
            "javascript": r"^\s*(function\s+\w+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(|\w+\s*:\s*function)",
            "typescript": r"^\s*(function\s+\w+|const\s+\w+\s*=\s*\(|let\s+\w+\s*=\s*\(|\w+\s*:\s*function)",
            "java": r"^\s*(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\(",
            "cpp": r"^\s*\w+\s+\w+\s*\(",
            "c": r"^\s*\w+\s+\w+\s*\(",
            "go": r"^\s*func\s+\w+",
            "rust": r"^\s*fn\s+\w+",
            "php": r"^\s*(public|private|protected)?\s*function\s+\w+",
            "ruby": r"^\s*def\s+\w+",
        }

        pattern = patterns.get(language)
        if not pattern:
            return 0

        return len(re.findall(pattern, content, re.MULTILINE))

    def _count_classes(self, content: str, language: str | None) -> int:
        """Count class definitions in the content."""
        if not language:
            return 0

        patterns = {
            "python": r"^\s*class\s+\w+",
            "javascript": r"^\s*class\s+\w+",
            "typescript": r"^\s*class\s+\w+",
            "java": r"^\s*(public|private|protected)?\s*class\s+\w+",
            "cpp": r"^\s*class\s+\w+",
            "c": r"^\s*struct\s+\w+",
            "go": r"^\s*type\s+\w+\s+struct",
            "rust": r"^\s*struct\s+\w+",
            "php": r"^\s*class\s+\w+",
            "ruby": r"^\s*class\s+\w+",
        }

        pattern = patterns.get(language)
        if not pattern:
            return 0

        return len(re.findall(pattern, content, re.MULTILINE))

    def _count_imports(self, content: str, language: str | None) -> int:
        """Count import statements in the content."""
        if not language:
            return 0

        patterns = {
            "python": r"^\s*(import\s+\w+|from\s+\w+\s+import)",
            "javascript": r"^\s*(import\s+.*from|const\s+.*=\s*require)",
            "typescript": r"^\s*(import\s+.*from|const\s+.*=\s*require)",
            "java": r"^\s*import\s+[\w.]+",
            "cpp": r'^\s*#include\s*[<"]',
            "c": r'^\s*#include\s*[<"]',
            "go": r"^\s*import\s+",
            "rust": r"^\s*use\s+",
            "php": r"^\s*(use\s+|require|include)",
            "ruby": r"^\s*require\s+",
        }

        pattern = patterns.get(language)
        if not pattern:
            return 0

        return len(re.findall(pattern, content, re.MULTILINE))

    def _calculate_comment_ratio(self, content: str, language: str | None) -> float:
        """Calculate the ratio of comment lines to total lines."""
        if not language:
            return 0.0

        lines = content.split("\n")
        total_lines = len([line for line in lines if line.strip()])

        if total_lines == 0:
            return 0.0

        comment_patterns = {
            "python": r"^\s*#",
            "javascript": r"^\s*(//|/\*)",
            "typescript": r"^\s*(//|/\*)",
            "java": r"^\s*(//|/\*)",
            "cpp": r"^\s*(//|/\*)",
            "c": r"^\s*(//|/\*)",
            "go": r"^\s*//",
            "rust": r"^\s*//",
            "php": r"^\s*(//|#|/\*)",
            "ruby": r"^\s*#",
            "sql": r"^\s*--",
            "html": r"^\s*<!--",
            "css": r"^\s*/\*",
        }

        pattern = comment_patterns.get(language)
        if not pattern:
            return 0.0

        comment_lines = 0
        for line in lines:
            if re.match(pattern, line):
                comment_lines += 1

        return comment_lines / total_lines

    def _calculate_complexity_score(self, content: str, language: str | None) -> float:
        """Calculate a simplified complexity score based on control structures."""
        if not language:
            return 0.0

        # Count control structures that increase complexity
        complexity_patterns = {
            "python": [
                r"\bif\b",
                r"\belif\b",
                r"\belse\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\btry\b",
                r"\bexcept\b",
                r"\bfinally\b",
                r"\bwith\b",
            ],
            "javascript": [
                r"\bif\b",
                r"\belse\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bswitch\b",
                r"\btry\b",
                r"\bcatch\b",
                r"\bfinally\b",
            ],
            "java": [
                r"\bif\b",
                r"\belse\b",
                r"\bfor\b",
                r"\bwhile\b",
                r"\bswitch\b",
                r"\btry\b",
                r"\bcatch\b",
                r"\bfinally\b",
            ],
        }

        patterns = complexity_patterns.get(language, complexity_patterns.get("python", []))

        total_complexity = 0
        for pattern in patterns:
            total_complexity += len(re.findall(pattern, content, re.IGNORECASE))

        # Normalize by number of lines
        lines = len(content.split("\n"))
        return total_complexity / max(lines, 1) * 100  # Scale to 0-100

    def _calculate_quality_score(
        self, content: str, language: str | None, complexity_score: float, comments_ratio: float
    ) -> float:
        """Calculate a code quality score based on various metrics."""
        if not language:
            return 0.0

        # Base score
        quality_score = 100.0

        # Penalize high complexity
        if complexity_score > 10:
            quality_score -= min(complexity_score - 10, 30)

        # Reward good comment ratio (10-30% is considered good)
        if 0.1 <= comments_ratio <= 0.3:
            quality_score += 10
        elif comments_ratio < 0.05:
            quality_score -= 15
        elif comments_ratio > 0.5:
            quality_score -= 10

        # Check for common code smells
        lines = content.split("\n")

        # Long lines penalty
        long_lines = sum(1 for line in lines if len(line) > 120)
        if long_lines > len(lines) * 0.1:  # More than 10% long lines
            quality_score -= 10

        # Empty lines ratio (too many or too few)
        empty_lines = sum(1 for line in lines if not line.strip())
        empty_ratio = empty_lines / max(len(lines), 1)
        if empty_ratio > 0.3 or empty_ratio < 0.05:
            quality_score -= 5

        # TODO patterns (indicates incomplete code)
        todo_count = len(re.findall(r"TODO|FIXME|HACK", content, re.IGNORECASE))
        if todo_count > 0:
            quality_score -= min(todo_count * 2, 10)

        return max(0.0, min(100.0, quality_score))

    def _get_file_author(self, file_path: Path) -> str | None:
        """Get file author information (simplified implementation)."""
        # This is a simplified implementation
        # In a real scenario, you might use git blame or file metadata
        try:
            stat_info = file_path.stat()
            # On Unix systems, you could get the owner
            import pwd

            owner = pwd.getpwuid(stat_info.st_uid).pw_name
            return owner
        except:
            return None

    def _get_search_engine(self, paths: list[str] | None, context: int) -> PySearch:
        """Get search engine with appropriate configuration."""
        if not self.current_config:
            raise ValueError("Search configuration not initialized")

        if paths:
            temp_config = SearchConfig(
                paths=paths,
                include=self.current_config.include,
                exclude=self.current_config.exclude,
                context=context,
                parallel=self.current_config.parallel,
                workers=self.current_config.workers,
            )
            return PySearch(temp_config)
        else:
            if context != self.current_config.context:
                temp_config = SearchConfig(
                    paths=self.current_config.paths,
                    include=self.current_config.include,
                    exclude=self.current_config.exclude,
                    context=context,
                    parallel=self.current_config.parallel,
                    workers=self.current_config.workers,
                )
                return PySearch(temp_config)

            if not self.search_engine:
                raise ValueError("Search engine not initialized")
            return self.search_engine

    async def search_with_ranking(
        self,
        pattern: str,
        paths: list[str] | None = None,
        context: int = 3,
        use_regex: bool = False,
        ranking_factors: dict[RankingFactor, float] | None = None,
        max_results: int = 50,
        session_id: str | None = None,
    ) -> list[RankedSearchResult]:
        """
        Perform search with advanced result ranking.

        Args:
            pattern: Search pattern
            paths: Optional list of paths to search
            context: Number of context lines around matches
            use_regex: Whether to use regex search
            ranking_factors: Custom weights for ranking factors
            max_results: Maximum number of results to return
            session_id: Optional session ID for context management

        Returns:
            List of ranked search results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Perform basic search
        temp_engine = self._get_search_engine(paths, context)
        query = Query(pattern=pattern, use_regex=use_regex, context=context)
        search_result = temp_engine.run(query)

        # Convert to ranked results
        ranked_results = []
        weights = ranking_factors or self.ranking_weights

        for search_item in search_result.items:
            # Calculate ranking factors
            factors = await self._calculate_ranking_factors(search_item, pattern, session)

            # Calculate overall relevance score
            relevance_score = sum(
                factors.get(factor, 0.0) * weight for factor, weight in weights.items()
            )

            ranked_result = RankedSearchResult(
                item={
                    "file": str(search_item.file),
                    "start_line": search_item.start_line,
                    "end_line": search_item.end_line,
                    "lines": search_item.lines,
                    "match_spans": search_item.match_spans,
                    "score": getattr(search_item, "score", None),
                },
                relevance_score=relevance_score,
                ranking_factors=factors,
            )

            ranked_results.append(ranked_result)

        # Sort by relevance score
        ranked_results.sort(key=lambda x: x.relevance_score, reverse=True)

        # Limit results
        ranked_results = ranked_results[:max_results]

        # Add to session history
        session.queries.append(
            {
                "type": "ranked_search",
                "pattern": pattern,
                "use_regex": use_regex,
                "result_count": len(ranked_results),
                "avg_relevance": (
                    sum(r.relevance_score for r in ranked_results) / len(ranked_results)
                    if ranked_results
                    else 0
                ),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return ranked_results

    async def _calculate_ranking_factors(
        self, search_item: SearchItem, pattern: str, session: SearchSession
    ) -> dict[RankingFactor, float]:
        """
        Calculate ranking factors for a search result.

        Args:
            search_item: The search result item
            pattern: The search pattern
            session: Current search session

        Returns:
            Dictionary of ranking factor scores (0.0 to 1.0)
        """
        factors = {}

        # Pattern match quality
        factors[RankingFactor.PATTERN_MATCH_QUALITY] = self._calculate_pattern_match_quality(
            search_item, pattern
        )

        # File importance
        factors[RankingFactor.FILE_IMPORTANCE] = await self._calculate_file_importance(
            search_item.file
        )

        # Context relevance
        factors[RankingFactor.CONTEXT_RELEVANCE] = self._calculate_context_relevance(
            search_item, session
        )

        # Recency
        factors[RankingFactor.RECENCY] = await self._calculate_recency_score(search_item.file)

        # File size factor
        factors[RankingFactor.FILE_SIZE] = await self._calculate_file_size_factor(search_item.file)

        # Language priority
        factors[RankingFactor.LANGUAGE_PRIORITY] = self._calculate_language_priority(
            search_item.file
        )

        return factors

    def _calculate_pattern_match_quality(self, search_item: SearchItem, pattern: str) -> float:
        """Calculate pattern match quality score."""
        # Check for exact matches
        exact_matches = 0
        partial_matches = 0

        for line in search_item.lines:
            if pattern.lower() in line.lower():
                if pattern.lower() == line.strip().lower():
                    exact_matches += 1
                else:
                    partial_matches += 1

        # Score based on match type and frequency
        if exact_matches > 0:
            return 1.0
        elif partial_matches > 0:
            return 0.7 + min(partial_matches * 0.1, 0.3)
        else:
            return 0.5  # Default for regex or complex matches

    async def _calculate_file_importance(self, file_path: Path) -> float:
        """Calculate file importance based on various factors."""
        try:
            # Analyze file content
            analysis = await self.analyze_file_content(str(file_path), True, True)

            # Base importance on file characteristics
            importance = 0.5  # Base score

            # Larger files might be more important (up to a point)
            if analysis.file_size > 1000:
                importance += min(analysis.file_size / 10000, 0.2)

            # Files with more functions/classes might be more important
            if analysis.functions_count > 5:
                importance += min(analysis.functions_count / 50, 0.2)

            if analysis.classes_count > 2:
                importance += min(analysis.classes_count / 20, 0.1)

            # Good code quality increases importance
            if analysis.code_quality_score > 80:
                importance += 0.1

            # Certain file patterns are more important
            file_str = str(file_path).lower()
            if any(pattern in file_str for pattern in ["main", "index", "app", "core"]):
                importance += 0.2

            if any(pattern in file_str for pattern in ["test", "spec"]):
                importance -= 0.1  # Tests are less important for general searches

            return min(1.0, importance)

        except Exception:
            return 0.5  # Default importance if analysis fails

    def _calculate_context_relevance(
        self, search_item: SearchItem, session: SearchSession
    ) -> float:
        """Calculate context relevance based on search session."""
        relevance = 0.5  # Base score

        # Check if this file has appeared in recent searches
        recent_files = set()
        for query in session.queries[-5:]:  # Last 5 queries
            if "files" in query:
                recent_files.update(query["files"])

        if str(search_item.file) in recent_files:
            relevance += 0.3

        # Check if this matches session context
        if "preferred_languages" in session.context:
            file_ext = Path(search_item.file).suffix.lower()
            if file_ext in session.context["preferred_languages"]:
                relevance += 0.2

        return min(1.0, relevance)

    async def _calculate_recency_score(self, file_path: Path) -> float:
        """Calculate recency score based on file modification time."""
        try:
            stat_info = file_path.stat()
            last_modified = datetime.fromtimestamp(stat_info.st_mtime)
            now = datetime.now()

            # Calculate days since modification
            days_old = (now - last_modified).days

            # Score decreases with age
            if days_old <= 1:
                return 1.0
            elif days_old <= 7:
                return 0.8
            elif days_old <= 30:
                return 0.6
            elif days_old <= 90:
                return 0.4
            elif days_old <= 365:
                return 0.2
            else:
                return 0.1

        except Exception:
            return 0.5  # Default if can't get modification time

    async def _calculate_file_size_factor(self, file_path: Path) -> float:
        """Calculate file size factor (moderate size is preferred)."""
        try:
            file_size = file_path.stat().st_size

            # Prefer moderate-sized files
            if 1000 <= file_size <= 50000:  # 1KB to 50KB
                return 1.0
            elif 500 <= file_size < 1000 or 50000 < file_size <= 100000:
                return 0.8
            elif file_size < 500 or file_size > 100000:
                return 0.6
            else:
                return 0.4

        except Exception:
            return 0.5

    def _calculate_language_priority(self, file_path: Path) -> float:
        """Calculate language priority score."""
        extension = file_path.suffix.lower()

        # Priority based on common/important languages
        high_priority = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs"}
        medium_priority = {".php", ".rb", ".swift", ".kt", ".scala", ".cs"}
        low_priority = {".txt", ".md", ".json", ".xml", ".yaml", ".yml"}

        if extension in high_priority:
            return 1.0
        elif extension in medium_priority:
            return 0.7
        elif extension in low_priority:
            return 0.3
        else:
            return 0.5

    async def search_with_filters(
        self,
        pattern: str,
        search_filter: SearchFilter,
        paths: list[str] | None = None,
        context: int = 3,
        use_regex: bool = False,
        session_id: str | None = None,
    ) -> SearchResponse:
        """
        Perform search with advanced filtering capabilities.

        Args:
            pattern: Search pattern
            search_filter: Advanced filtering options
            paths: Optional list of paths to search
            context: Number of context lines around matches
            use_regex: Whether to use regex search
            session_id: Optional session ID for context management

        Returns:
            SearchResponse with filtered results
        """
        if not self.search_engine or not self.current_config:
            raise ValueError("Search engine not initialized")

        # Get or create session
        session = self._get_or_create_session(session_id)

        # Perform basic search first
        temp_engine = self._get_search_engine(paths, context)
        query = Query(pattern=pattern, use_regex=use_regex, context=context)
        search_result = temp_engine.run(query)

        # Apply filters to results
        filtered_items = []

        for search_item in search_result.items:
            if await self._passes_filters(search_item, search_filter):
                filtered_items.append(
                    {
                        "file": str(search_item.file),
                        "start_line": search_item.start_line,
                        "end_line": search_item.end_line,
                        "lines": search_item.lines,
                        "match_spans": search_item.match_spans,
                        "score": getattr(search_item, "score", None),
                    }
                )

        # Create filtered response
        response = SearchResponse(
            items=filtered_items,
            stats={
                "files_scanned": search_result.stats.files_scanned,
                "files_matched": len(set(item["file"] for item in filtered_items)),
                "total_items": len(filtered_items),
                "elapsed_ms": search_result.stats.elapsed_ms,
                "indexed_files": search_result.stats.indexed_files,
                "filtered_out": len(search_result.items) - len(filtered_items),
            },
            query_info={
                "pattern": pattern,
                "use_regex": use_regex,
                "use_ast": False,
                "use_semantic": False,
                "context": context,
                "filters_applied": asdict(search_filter),
                "session_id": session.session_id,
            },
            total_matches=len(filtered_items),
            execution_time_ms=search_result.stats.elapsed_ms,
        )

        # Add to session history
        session.queries.append(
            {
                "type": "filtered_search",
                "pattern": pattern,
                "filters": asdict(search_filter),
                "result_count": len(filtered_items),
                "filtered_out": len(search_result.items) - len(filtered_items),
                "timestamp": datetime.now().isoformat(),
            }
        )

        return response

    async def _passes_filters(self, search_item: SearchItem, search_filter: SearchFilter) -> bool:
        """
        Check if a search item passes all the specified filters.

        Args:
            search_item: The search result item to check
            search_filter: The filters to apply

        Returns:
            True if the item passes all filters, False otherwise
        """
        file_path = Path(search_item.file)

        try:
            # File size filters
            if search_filter.min_file_size is not None or search_filter.max_file_size is not None:
                file_size = file_path.stat().st_size

                if (
                    search_filter.min_file_size is not None
                    and file_size < search_filter.min_file_size
                ):
                    return False

                if (
                    search_filter.max_file_size is not None
                    and file_size > search_filter.max_file_size
                ):
                    return False

            # Modification date filters
            if (
                search_filter.modified_after is not None
                or search_filter.modified_before is not None
            ):
                last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)

                if (
                    search_filter.modified_after is not None
                    and last_modified < search_filter.modified_after
                ):
                    return False

                if (
                    search_filter.modified_before is not None
                    and last_modified > search_filter.modified_before
                ):
                    return False

            # File extension filters
            if search_filter.file_extensions is not None:
                file_ext = file_path.suffix.lower()
                if file_ext not in [ext.lower() for ext in search_filter.file_extensions]:
                    return False

            # Language filters
            if search_filter.languages is not None:
                detected_language = self._detect_file_language(file_path)
                if detected_language not in [lang.lower() for lang in search_filter.languages]:
                    return False

            # Exclude patterns
            if search_filter.exclude_patterns is not None:
                file_str = str(file_path)
                for pattern in search_filter.exclude_patterns:
                    if re.search(pattern, file_str, re.IGNORECASE):
                        return False

            # Complexity filters (requires file analysis)
            if search_filter.min_complexity is not None or search_filter.max_complexity is not None:
                try:
                    analysis = await self.analyze_file_content(str(file_path), True, False)

                    if (
                        search_filter.min_complexity is not None
                        and analysis.complexity_score < search_filter.min_complexity
                    ):
                        return False

                    if (
                        search_filter.max_complexity is not None
                        and analysis.complexity_score > search_filter.max_complexity
                    ):
                        return False

                except Exception:
                    # If analysis fails, skip complexity filtering for this file
                    pass

            # Author filters (simplified implementation)
            if search_filter.authors is not None:
                try:
                    file_author = self._get_file_author(file_path)
                    if file_author and file_author not in search_filter.authors:
                        return False
                except Exception:
                    # If can't get author info, skip this filter
                    pass

            return True

        except Exception:
            # If any filter check fails, exclude the item to be safe
            return False

    async def get_file_statistics(
        self, paths: list[str] | None = None, include_analysis: bool = False
    ) -> dict[str, Any]:
        """
        Get comprehensive statistics about files in the search paths.

        Args:
            paths: Optional list of paths to analyze
            include_analysis: Whether to include detailed file analysis

        Returns:
            Dictionary with file statistics and analysis
        """
        if not self.current_config:
            raise ValueError("Search configuration not initialized")

        search_paths = paths or self.current_config.paths

        stats: dict[str, Any] = {
            "total_files": 0,
            "total_size": 0,
            "languages": {},
            "file_extensions": {},
            "size_distribution": {
                "small": 0,  # < 1KB
                "medium": 0,  # 1KB - 100KB
                "large": 0,  # 100KB - 1MB
                "very_large": 0,  # > 1MB
            },
            "complexity_distribution": {},
            "quality_distribution": {},
            "modification_dates": {
                "last_day": 0,
                "last_week": 0,
                "last_month": 0,
                "last_year": 0,
                "older": 0,
            },
        }

        now = datetime.now()

        for search_path in search_paths:
            path_obj = Path(search_path)

            if not path_obj.exists():
                continue

            if path_obj.is_file():
                files_to_process = [path_obj]
            else:
                # Get all files matching include patterns
                files_to_process = []
                include_patterns = self.current_config.include or ["**/*"]
                for include_pattern in include_patterns:
                    files_to_process.extend(path_obj.glob(include_pattern))

            for file_path in files_to_process:
                if not file_path.is_file():
                    continue

                try:
                    file_stat = file_path.stat()
                    file_size = file_stat.st_size
                    last_modified = datetime.fromtimestamp(file_stat.st_mtime)

                    stats["total_files"] += 1
                    stats["total_size"] += file_size

                    # File extension stats
                    ext = file_path.suffix.lower()
                    stats["file_extensions"][ext] = stats["file_extensions"].get(ext, 0) + 1

                    # Language stats
                    language = self._detect_file_language(file_path)
                    if language:
                        stats["languages"][language] = stats["languages"].get(language, 0) + 1

                    # Size distribution
                    if file_size < 1024:
                        stats["size_distribution"]["small"] += 1
                    elif file_size < 102400:
                        stats["size_distribution"]["medium"] += 1
                    elif file_size < 1048576:
                        stats["size_distribution"]["large"] += 1
                    else:
                        stats["size_distribution"]["very_large"] += 1

                    # Modification date distribution
                    days_old = (now - last_modified).days
                    if days_old <= 1:
                        stats["modification_dates"]["last_day"] += 1
                    elif days_old <= 7:
                        stats["modification_dates"]["last_week"] += 1
                    elif days_old <= 30:
                        stats["modification_dates"]["last_month"] += 1
                    elif days_old <= 365:
                        stats["modification_dates"]["last_year"] += 1
                    else:
                        stats["modification_dates"]["older"] += 1

                    # Detailed analysis if requested
                    if include_analysis:
                        try:
                            analysis = await self.analyze_file_content(str(file_path), True, True)

                            # Complexity distribution
                            complexity_bucket = self._get_complexity_bucket(
                                analysis.complexity_score
                            )
                            stats["complexity_distribution"][complexity_bucket] = (
                                stats["complexity_distribution"].get(complexity_bucket, 0) + 1
                            )

                            # Quality distribution
                            quality_bucket = self._get_quality_bucket(analysis.code_quality_score)
                            stats["quality_distribution"][quality_bucket] = (
                                stats["quality_distribution"].get(quality_bucket, 0) + 1
                            )

                        except Exception:
                            # Skip analysis for files that can't be analyzed
                            pass

                except Exception:
                    # Skip files that can't be processed
                    continue

        return stats

    def _get_complexity_bucket(self, complexity_score: float) -> str:
        """Get complexity bucket for a complexity score."""
        if complexity_score < 5:
            return "low"
        elif complexity_score < 15:
            return "medium"
        elif complexity_score < 30:
            return "high"
        else:
            return "very_high"

    def _get_quality_bucket(self, quality_score: float) -> str:
        """Get quality bucket for a quality score."""
        if quality_score >= 90:
            return "excellent"
        elif quality_score >= 75:
            return "good"
        elif quality_score >= 60:
            return "fair"
        elif quality_score >= 40:
            return "poor"
        else:
            return "very_poor"


def create_mcp_server() -> PySearchMCPServer:
    """Create and configure the MCP server instance."""
    return PySearchMCPServer()


if __name__ == "__main__":
    print("PySearch MCP Server")
    print("=" * 50)
    print()
    print("This server provides advanced search capabilities including:")
    print("- Fuzzy search with configurable similarity thresholds")
    print("- Multi-pattern search with logical operators")
    print("- File content analysis and complexity metrics")
    print("- Advanced result ranking and filtering")
    print("- Search session management")
    print("- Progress reporting for long operations")
    print()
    print("To run the server:")
    print("1. Install dependencies: pip install rapidfuzz fuzzywuzzy python-levenshtein")
    print("2. Install FastMCP: pip install fastmcp")
    print("3. Uncomment FastMCP integration code")
    print("4. Run: python mcp_server.py")
    print()
    print("Enhanced server implementation is ready for FastMCP integration.")
