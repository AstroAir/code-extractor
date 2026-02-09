#!/usr/bin/env python3
"""
Example 02: AST Structural Search — AST 结构化搜索示例

Demonstrates:
- AST-based function/class/decorator/import filtering
- Combining regex with AST filters
- Query object construction for AST search
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig
from pysearch.core.types import ASTFilters, Query

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_items(items: list, max_items: int = 5) -> None:
    for item in items[:max_items]:
        print(f"  {item.file}:{item.start_line}-{item.end_line}")
        for line in item.lines[:4]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 1. Search functions by name pattern / 按函数名模式搜索
# ---------------------------------------------------------------------------


def demo_filter_by_func_name(engine: PySearch) -> None:
    section("1. Filter by Function Name / 按函数名过滤")

    filters = ASTFilters(func_name=r".*search.*")
    query = Query(
        pattern="def",
        use_ast=True,
        filters=filters,
        context=2,
    )
    results = engine.run(query)

    print(f"Functions matching '.*search.*': {len(results.items)} found\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 2. Search classes by name pattern / 按类名模式搜索
# ---------------------------------------------------------------------------


def demo_filter_by_class_name(engine: PySearch) -> None:
    section("2. Filter by Class Name / 按类名过滤")

    filters = ASTFilters(class_name=r".*Manager.*")
    query = Query(
        pattern="class",
        use_ast=True,
        filters=filters,
        context=2,
    )
    results = engine.run(query)

    print(f"Classes matching '.*Manager.*': {len(results.items)} found\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 3. Search by decorator / 按装饰器搜索
# ---------------------------------------------------------------------------


def demo_filter_by_decorator(engine: PySearch) -> None:
    section("3. Filter by Decorator / 按装饰器过滤")

    filters = ASTFilters(decorator=r"dataclass|property")
    query = Query(
        pattern="def|class",
        use_ast=True,
        use_regex=True,
        filters=filters,
        context=3,
    )
    results = engine.run(query)

    print(f"Items with @dataclass or @property: {len(results.items)} found\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 4. Search by import / 按导入搜索
# ---------------------------------------------------------------------------


def demo_filter_by_import(engine: PySearch) -> None:
    section("4. Filter by Import / 按导入过滤")

    filters = ASTFilters(imported=r"pathlib|Path")
    query = Query(
        pattern="from|import",
        use_ast=True,
        use_regex=True,
        filters=filters,
        context=1,
    )
    results = engine.run(query)

    print(f"Files importing pathlib/Path: {len(results.items)} found\n")
    print_items(results.items, max_items=3)


# ---------------------------------------------------------------------------
# 5. Combined AST + regex search / 组合 AST + 正则搜索
# ---------------------------------------------------------------------------


def demo_combined_ast_regex(engine: PySearch) -> None:
    section("5. Combined AST + Regex / 组合 AST + 正则")

    # Find async functions with 'search' in name
    # 查找名称含 'search' 的异步函数
    filters = ASTFilters(func_name=r".*search.*")
    query = Query(
        pattern=r"async\s+def",
        use_ast=True,
        use_regex=True,
        filters=filters,
        context=3,
    )
    results = engine.run(query)

    print(f"Async functions with 'search' in name: {len(results.items)} found\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 6. Convenience API for AST search / 便捷 API
# ---------------------------------------------------------------------------


def demo_convenience_ast_search(engine: PySearch) -> None:
    section("6. Convenience API / 便捷 API")

    # Using engine.search() with use_ast kwarg
    filters = ASTFilters(func_name=r"__init__")
    results = engine.search("def", use_ast=True, filters=filters, context=1)

    print(f"__init__ methods: {len(results.items)} found\n")
    print_items(results.items, max_items=3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    search_path = sys.argv[1] if len(sys.argv) > 1 else "./src"

    if not Path(search_path).exists():
        print(f"Error: search path '{search_path}' does not exist.")
        sys.exit(1)

    config = SearchConfig(paths=[search_path], include=["**/*.py"], context=2)
    engine = PySearch(config)

    demo_filter_by_func_name(engine)
    demo_filter_by_class_name(engine)
    demo_filter_by_decorator(engine)
    demo_filter_by_import(engine)
    demo_combined_ast_regex(engine)
    demo_convenience_ast_search(engine)

    print("\n✓ All AST search examples completed.")


if __name__ == "__main__":
    main()
