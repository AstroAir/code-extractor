#!/usr/bin/env python3
"""
Example 04: Boolean Search — 布尔逻辑搜索示例

Demonstrates:
- AND / OR / NOT boolean queries / 布尔逻辑查询
- Nested boolean expressions / 嵌套布尔表达式
- check_boolean_match() for file-level checks / 文件级布尔检查
- Count-only boolean search / 布尔计数搜索
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig
from pysearch.core.types import Query

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
        for line in item.lines[:3]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 1. Simple AND query / 简单 AND 查询
# ---------------------------------------------------------------------------


def demo_and_query(engine: PySearch) -> None:
    section("1. AND Query / AND 查询")

    # Find lines containing both 'async' and 'def'
    # 查找同时包含 'async' 和 'def' 的行
    query = Query(
        pattern="async AND def",
        use_boolean=True,
        context=2,
    )
    results = engine.run(query)

    print("Query: 'async AND def'")
    print(f"Found {len(results.items)} matches in {results.stats.files_matched} files\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 2. OR query / OR 查询
# ---------------------------------------------------------------------------


def demo_or_query(engine: PySearch) -> None:
    section("2. OR Query / OR 查询")

    query = Query(
        pattern="SearchConfig OR SearchResult",
        use_boolean=True,
        context=1,
    )
    results = engine.run(query)

    print("Query: 'SearchConfig OR SearchResult'")
    print(f"Found {len(results.items)} matches\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 3. NOT query / NOT 查询
# ---------------------------------------------------------------------------


def demo_not_query(engine: PySearch) -> None:
    section("3. NOT Query / NOT 查询")

    # Find 'import' lines that do NOT contain 'typing'
    # 查找不包含 'typing' 的 'import' 行
    query = Query(
        pattern="import NOT typing",
        use_boolean=True,
        context=0,
        max_per_file=2,
    )
    results = engine.run(query)

    print("Query: 'import NOT typing'")
    print(f"Found {len(results.items)} matches (max 2 per file)\n")
    print_items(results.items, max_items=5)


# ---------------------------------------------------------------------------
# 4. Nested boolean expressions / 嵌套布尔表达式
# ---------------------------------------------------------------------------


def demo_nested_boolean(engine: PySearch) -> None:
    section("4. Nested Boolean / 嵌套布尔表达式")

    # (async AND handler) NOT test
    query = Query(
        pattern="(async AND search) NOT test",
        use_boolean=True,
        context=2,
    )
    results = engine.run(query)

    print("Query: '(async AND search) NOT test'")
    print(f"Found {len(results.items)} matches\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 5. Complex boolean with OR + AND / 复杂布尔组合
# ---------------------------------------------------------------------------


def demo_complex_boolean(engine: PySearch) -> None:
    section("5. Complex Boolean / 复杂布尔组合")

    query = Query(
        pattern="(cache OR index) AND (enable OR disable)",
        use_boolean=True,
        context=2,
    )
    results = engine.run(query)

    print("Query: '(cache OR index) AND (enable OR disable)'")
    print(f"Found {len(results.items)} matches\n")
    print_items(results.items)


# ---------------------------------------------------------------------------
# 6. check_boolean_match — file-level check / 文件级布尔检查
# ---------------------------------------------------------------------------


def demo_boolean_file_check(engine: PySearch) -> None:
    section("6. File-Level Boolean Check / 文件级布尔检查")

    # Check specific files against a boolean expression
    # 检查特定文件是否匹配布尔表达式
    test_files = [
        Path("src/pysearch/core/api.py"),
        Path("src/pysearch/core/config.py"),
        Path("src/pysearch/search/fuzzy.py"),
    ]

    query_str = "(search AND config) NOT test"
    print(f"Boolean expression: '{query_str}'\n")

    for f in test_files:
        if f.exists():
            matches = engine.check_boolean_match(query_str, f)
            status = "MATCH" if matches else "NO MATCH"
            print(f"  {f}: {status}")
        else:
            print(f"  {f}: (file not found)")


# ---------------------------------------------------------------------------
# 7. Count-only boolean search / 布尔计数搜索
# ---------------------------------------------------------------------------


def demo_boolean_count(engine: PySearch) -> None:
    section("7. Count-Only Boolean Search / 布尔计数搜索")

    count_result = engine.search_count_only(
        "(async AND def) NOT test",
        use_boolean=True,
    )
    print("Query: '(async AND def) NOT test'")
    print(f"Total matches: {count_result.total_matches}")
    print(f"Files matched: {count_result.files_matched}")
    print(f"Elapsed: {count_result.stats.elapsed_ms:.1f}ms")


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

    demo_and_query(engine)
    demo_or_query(engine)
    demo_not_query(engine)
    demo_nested_boolean(engine)
    demo_complex_boolean(engine)
    demo_boolean_file_check(engine)
    demo_boolean_count(engine)

    print("\n✓ All boolean search examples completed.")


if __name__ == "__main__":
    main()
