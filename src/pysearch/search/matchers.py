"""
Pattern matching module for pysearch.

This module provides the core matching functionality for different search types:
text, regex, AST-based, and semantic matching. It handles the actual search
operations within individual files and returns structured results.

Classes:
    ASTNodeMatcher: Visitor class for AST-based pattern matching

Functions:
    search_in_file: Main entry point for file searching
    text_search: Simple text pattern matching
    regex_search: Enhanced regex pattern matching with named groups
    ast_search: AST-based structural pattern matching
    semantic_search: Lightweight semantic pattern matching

Key Features:
    - Multiple search modes with unified interface
    - Context-aware result extraction with configurable line counts
    - AST filtering by function names, class names, decorators, imports
    - Regex support with multiline mode and named groups
    - Semantic matching using lightweight vector and symbolic features
    - Efficient line-to-column mapping for precise match locations

Example:
    Basic text search:
        >>> from pysearch.matchers import search_in_file
        >>> from pysearch.types import Query
        >>>
        >>> query = Query(pattern="def main", use_regex=False)
        >>> results = search_in_file(Path("example.py"), "def main():\n    pass", query, context=2)
        >>> print(f"Found {len(results)} matches")

    AST-based search with filters:
        >>> from pysearch.types import ASTFilters
        >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)
        >>> results = search_in_file(Path("example.py"), content, query, context=2)
"""

from __future__ import annotations

import ast
import bisect
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import regex as regex_mod  # better regex engine

# Type checking compatibility: avoid type issues when regex stubs are missing
try:
    from typing import TYPE_CHECKING

    TYPE_CHECKING  # keep linters happy
except Exception:  # pragma: no cover
    pass
# Use Any as fallback to avoid Pylance/Mypy inconsistent Pattern declaration errors

from ..core.types import ASTFilters, MatchSpan, Query, SearchItem
from ..utils.helpers import extract_context, iter_python_ast_nodes, split_lines_keepends


@dataclass(slots=True)
class TextMatch:
    line_index: int
    start_col: int
    end_col: int


@lru_cache(maxsize=64)
def _get_compiled_regex(pattern: str, flags: int) -> regex_mod.Pattern:
    return regex_mod.compile(pattern, flags=flags)


def _build_line_starts(text: str) -> list[int]:
    """返回每一行起始的绝对偏移（0-based）。"""
    starts: list[int] = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line_col(line_starts: list[int], offset: int) -> tuple[int, int]:
    """通过二分将绝对 offset 映射到 (line_index_0based, col_0based)。"""
    # bisect_right 返回应插入位置，减一即为所在行索引
    li = bisect.bisect_right(line_starts, offset) - 1
    if li < 0:
        li = 0
    col = offset - line_starts[li]
    return li, col


def _is_match_in_string_or_comment(
    text: str, line_index: int, start_col: int, end_col: int
) -> tuple[bool, bool, bool]:
    """
    Check if a match is in a string literal, comment, or docstring.

    Uses a simplified but more robust approach:
    - Detects ``#`` comments while ignoring ``#`` inside string literals
    - Handles single/double quotes and triple-quoted strings
    - Recognizes raw strings (r"...", r'...')

    Returns:
        Tuple of (is_in_string, is_in_comment, is_in_docstring)
    """
    lines = split_lines_keepends(text)
    if line_index >= len(lines):
        return False, False, False

    line = lines[line_index]

    # Walk through the line character by character to determine the state
    # at the match position.  This handles quotes inside comments and
    # comments inside strings correctly.
    in_string: str | None = None  # None, "'", '"', "'''", '"""'
    i = 0
    while i < len(line) and i < start_col:
        ch = line[i]

        if in_string is None:
            # Not inside a string – check for comment
            if ch == "#":
                # Everything from here is a comment
                return False, True, False

            # Check for triple-quote opening
            triple = line[i : i + 3]
            if triple in ('"""', "'''"):
                in_string = triple
                i += 3
                continue

            # Check for single-quote opening (skip raw-string prefix)
            if ch in ('"', "'"):
                in_string = ch
                i += 1
                continue

            # Skip raw-string prefix characters before a quote
            if ch in ("r", "R", "b", "B", "f", "F", "u", "U") and i + 1 < len(line):
                next_ch = line[i + 1]
                if next_ch in ('"', "'"):
                    # This is a prefix; let the next iteration handle the quote
                    i += 1
                    continue
        else:
            # Inside a string – look for closing delimiter
            if ch == "\\" and i + 1 < len(line):
                i += 2  # skip escaped character
                continue

            if len(in_string) == 3:
                # Triple-quoted string
                if line[i : i + 3] == in_string:
                    in_string = None
                    i += 3
                    continue
            else:
                if ch == in_string:
                    in_string = None
                    i += 1
                    continue

        i += 1

    if in_string is not None:
        is_docstring = len(in_string) == 3
        return True, False, is_docstring

    # Not inside a string at start_col – one more check: is start_col
    # itself the start of a comment?
    if start_col < len(line) and line[start_col] == "#":
        return False, True, False

    return False, False, False


def find_text_regex_matches(text: str, pattern: str, use_regex: bool) -> list[TextMatch]:
    matches: list[TextMatch] = []
    if not text:
        return matches

    # Guard against empty pattern to prevent infinite loops
    if not pattern:
        return matches

    if use_regex:
        # multiline, dotall for code; 使用 LRU 缓存避免重复编译
        flags = regex_mod.MULTILINE | regex_mod.DOTALL
        rx = _get_compiled_regex(pattern, flags)
        # 预构建行起始偏移，用于 O(log n) 映射
        line_starts = _build_line_starts(text)
        for m in rx.finditer(text):
            start_abs, end_abs = m.span()
            li, col_start = _offset_to_line_col(line_starts, start_abs)
            col_end = col_start + (end_abs - start_abs)
            matches.append(TextMatch(line_index=li, start_col=col_start, end_col=col_end))
    else:
        # simple substring search line by line
        lines = split_lines_keepends(text)
        for i, line in enumerate(lines):
            start = 0
            while True:
                j = line.find(pattern, start)
                if j == -1:
                    break
                matches.append(TextMatch(line_index=i, start_col=j, end_col=j + len(pattern)))
                # Ensure we always advance position to prevent infinite loops
                start = j + max(1, len(pattern))
    return matches


def group_matches_into_blocks(matches: list[TextMatch]) -> list[tuple[int, int, list[MatchSpan]]]:
    """
    将紧邻行的文本命中合并为块。
    返回: [(start_line_1based, end_line_1based, spans_struct)]
    其中 spans_struct: List[(line_index_0based, (start_col, end_col))]
    """
    if not matches:
        return []
    matches_sorted = sorted(matches, key=lambda m: (m.line_index, m.start_col, m.end_col))

    grouped: list[list[TextMatch]] = []
    bucket: list[TextMatch] = [matches_sorted[0]]
    for m in matches_sorted[1:]:
        if m.line_index <= bucket[-1].line_index + 1:
            bucket.append(m)
        else:
            grouped.append(bucket)
            bucket = [m]
    grouped.append(bucket)

    result: list[tuple[int, int, list[MatchSpan]]] = []
    for group in grouped:
        start_l = group[0].line_index + 1
        end_l = group[-1].line_index + 1
        spans_struct: list[MatchSpan] = []
        for tm in group:
            spans_struct.append((tm.line_index, (tm.start_col, tm.end_col)))
        result.append((start_l, end_l, spans_struct))
    return result


def ast_node_matches_filters(node: ast.AST, filters: ASTFilters) -> bool:
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
        if filters.func_name and not regex_mod.search(filters.func_name, node.name):
            return False
        if filters.decorator:
            decos = []
            for d in node.decorator_list:
                try:
                    decos.append(ast.unparse(d))
                except Exception:
                    pass
            if not any(regex_mod.search(filters.decorator, d) for d in decos):
                return False
    if isinstance(node, ast.ClassDef):
        if filters.class_name and not regex_mod.search(filters.class_name, node.name):
            return False
        if filters.decorator:
            decos = []
            for d in node.decorator_list:
                try:
                    decos.append(ast.unparse(d))
                except Exception:
                    pass
            if not any(regex_mod.search(filters.decorator, d) for d in decos):
                return False
    if isinstance(node, (ast.Import, ast.ImportFrom)) and filters.imported:
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            mod = node.module or ""
            names = [f"{mod}.{a.name}" if mod else a.name for a in node.names]
        if not any(regex_mod.search(filters.imported, n) for n in names):
            return False
    return True


def find_ast_blocks(text: str, filters: ASTFilters | None) -> list[tuple[int, int]]:
    """
    Return list of (start_line, end_line) 1-based for AST nodes satisfying filters.
    If filters is None, returns empty.
    """
    if not filters:
        return []
    tree = iter_python_ast_nodes(text)
    if tree is None:
        return []
    results: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        # Use getattr with default to satisfy type checkers
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is not None and end_lineno is not None:
            if ast_node_matches_filters(node, filters):
                results.append((int(lineno), int(end_lineno)))
    # merge overlapping ranges
    results.sort()
    merged: list[tuple[int, int]] = []
    for s, e in results:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
    return merged


def search_in_file(
    path: Path,
    text: str,
    query: Query,
) -> list[SearchItem]:
    r"""
    Search for patterns within a single file and return structured results.

    This is the main entry point for file-level searching. It combines multiple
    search modes (text, regex, AST, semantic) and applies filters to produce
    SearchItem objects with appropriate context.

    Args:
        path: Path to the file being searched (for result metadata)
        text: File content as string
        query: Query object specifying search parameters and filters

    Returns:
        List of SearchItem objects representing matches with context

    Example:
        >>> from pathlib import Path
        >>> from pysearch.types import Query, ASTFilters
        >>>
        >>> # Simple text search
        >>> content = "def main():\n    print('Hello')\n    return 0"
        >>> query = Query(pattern="def main")
        >>> results = search_in_file(Path("example.py"), content, query)
        >>> print(f"Found {len(results)} matches")

        >>> # AST search with filters
        >>> filters = ASTFilters(func_name="main")
        >>> query = Query(pattern="def", use_ast=True, filters=filters)
        >>> results = search_in_file(Path("example.py"), content, query)

        >>> # Regex search
        >>> query = Query(pattern=r"def \w+", use_regex=True, context=1)
        >>> results = search_in_file(Path("example.py"), content, query)

    Note:
        The function intelligently combines different search modes:
        - Pure text/regex search when no AST filters are specified
        - Pure AST search when only AST filters are used
        - Intersection of text and AST matches when both are specified
        - Semantic similarity scoring when semantic search is enabled
    """
    lines = split_lines_keepends(text)
    items: list[SearchItem] = []

    # AST block ranges
    ast_blocks: list[tuple[int, int]] = []
    if query.filters:
        ast_blocks = find_ast_blocks(text, query.filters)

    # Semantic search mode: use concept-based pattern expansion
    if query.use_semantic and query.pattern:
        from .semantic import expand_semantic_query

        semantic_patterns = expand_semantic_query(query.pattern)
        all_tms: list[TextMatch] = []
        for sem_pattern in semantic_patterns:
            try:
                sem_matches = find_text_regex_matches(text, sem_pattern, use_regex=True)
                all_tms.extend(sem_matches)
            except Exception:
                continue
        # Deduplicate by (line_index, start_col, end_col)
        seen: set[tuple[int, int, int]] = set()
        tms: list[TextMatch] = []
        for tm in all_tms:
            key = (tm.line_index, tm.start_col, tm.end_col)
            if key not in seen:
                seen.add(key)
                tms.append(tm)
    else:
        # Text/regex matches
        tms = find_text_regex_matches(text, query.pattern, query.use_regex)

    # Filter matches based on search_strings, search_comments, search_docstrings
    if tms:
        filtered_tms = []
        for tm in tms:
            is_in_string, is_in_comment, is_in_docstring = _is_match_in_string_or_comment(
                text, tm.line_index, tm.start_col, tm.end_col
            )

            # Include match based on query settings
            include_match = True
            if is_in_string and not query.search_strings:
                include_match = False
            elif is_in_comment and not query.search_comments:
                include_match = False
            elif is_in_docstring and not query.search_docstrings:
                include_match = False

            if include_match:
                filtered_tms.append(tm)

        tms = filtered_tms

    if not tms and not ast_blocks:
        return []

    # When both present: intersect text matches with AST blocks to be semantic.
    def in_any_block(li: int) -> bool:
        l1 = li + 1
        for s, e in ast_blocks:
            if s <= l1 <= e:
                return True
        return False

    grouped = group_matches_into_blocks(tms) if tms else []

    if query.filters and ast_blocks:
        # 仅保留与 AST 块相交的文本匹配组
        grouped = [g for g in grouped if any(in_any_block(li) for li, _ in g[2])]

    # Build items from grouped matches
    for start_l, end_l, spans_struct in grouped:
        ctx_s, ctx_e, slice_lines = extract_context(lines, start_l, end_l, window=query.context)
        # Rebase spans to context slice line indexes
        spans_rebased: list[tuple[int, tuple[int, int]]] = []
        for li, (a, b) in spans_struct:
            if ctx_s <= li + 1 <= ctx_e:
                spans_rebased.append((li - (ctx_s - 1), (a, b)))
        items.append(
            SearchItem(
                file=path,
                start_line=ctx_s,
                end_line=ctx_e,
                lines=slice_lines,
                match_spans=spans_rebased,
            )
        )

    # If only AST blocks, return their context windows as items
    if not tms and ast_blocks:
        for s, e in ast_blocks:
            ctx_s, ctx_e, slice_lines = extract_context(lines, s, e, window=query.context)
            items.append(
                SearchItem(
                    file=path,
                    start_line=ctx_s,
                    end_line=ctx_e,
                    lines=slice_lines,
                    match_spans=[],
                )
            )

    return items
