from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from functools import lru_cache
import bisect

import regex as regex_mod  # better regex engine
# 类型检查兼容：避免在缺少 regex stubs 时产生类型问题
try:
    from typing import TYPE_CHECKING
    TYPE_CHECKING  # keep linters happy
except Exception:  # pragma: no cover
    pass
# 使用 Any 兜底，避免 Pylance/Mypy 对 Pattern 的声明不一致报错
from typing import Any as _RegexPattern

from .types import ASTFilters, Query, SearchItem, MatchSpan
from .utils import extract_context, iter_python_ast_nodes, split_lines_keepends


@dataclass(slots=True)
class TextMatch:
    line_index: int
    start_col: int
    end_col: int


@lru_cache(maxsize=64)
def _get_compiled_regex(pattern: str, flags: int) -> regex_mod.Pattern:
    return regex_mod.compile(pattern, flags=flags)


def _build_line_starts(text: str) -> List[int]:
    """返回每一行起始的绝对偏移（0-based）。"""
    starts: List[int] = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            starts.append(i + 1)
    return starts


def _offset_to_line_col(line_starts: List[int], offset: int) -> Tuple[int, int]:
    """通过二分将绝对 offset 映射到 (line_index_0based, col_0based)。"""
    # bisect_right 返回应插入位置，减一即为所在行索引
    li = bisect.bisect_right(line_starts, offset) - 1
    if li < 0:
        li = 0
    col = offset - line_starts[li]
    return li, col


def find_text_regex_matches(text: str, pattern: str, use_regex: bool) -> List[TextMatch]:
    matches: List[TextMatch] = []
    if not text:
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
                start = j + len(pattern)
    return matches


def group_matches_into_blocks(matches: List[TextMatch]) -> List[Tuple[int, int, List[MatchSpan]]]:
    """
    将紧邻行的文本命中合并为块。
    返回: [(start_line_1based, end_line_1based, spans_struct)]
    其中 spans_struct: List[(line_index_0based, (start_col, end_col))]
    """
    if not matches:
        return []
    matches_sorted = sorted(matches, key=lambda m: (m.line_index, m.start_col, m.end_col))

    grouped: List[List[TextMatch]] = []
    bucket: List[TextMatch] = [matches_sorted[0]]
    for m in matches_sorted[1:]:
        if m.line_index <= bucket[-1].line_index + 1:
            bucket.append(m)
        else:
            grouped.append(bucket)
            bucket = [m]
    grouped.append(bucket)

    result: List[Tuple[int, int, List[MatchSpan]]] = []
    for group in grouped:
        start_l = group[0].line_index + 1
        end_l = group[-1].line_index + 1
        spans_struct: List[MatchSpan] = []
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


def find_ast_blocks(text: str, filters: Optional[ASTFilters]) -> List[Tuple[int, int]]:
    """
    Return list of (start_line, end_line) 1-based for AST nodes satisfying filters.
    If filters is None, returns empty.
    """
    if not filters:
        return []
    tree = iter_python_ast_nodes(text)
    if tree is None:
        return []
    results: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        # Use getattr with default to satisfy type checkers
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        if lineno is not None and end_lineno is not None:
            if ast_node_matches_filters(node, filters):
                results.append((int(lineno), int(end_lineno)))
    # merge overlapping ranges
    results.sort()
    merged: List[Tuple[int, int]] = []
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
) -> List[SearchItem]:
    """
    Combines text/regex and AST filters to produce SearchItems with context window.
    """
    lines = split_lines_keepends(text)
    items: List[SearchItem] = []

    # AST block ranges
    ast_blocks: List[Tuple[int, int]] = []
    if query.filters:
        ast_blocks = find_ast_blocks(text, query.filters)

    # Text/regex matches
    tms = find_text_regex_matches(text, query.pattern, query.use_regex)

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
        spans_rebased: List[Tuple[int, Tuple[int, int]]] = []
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