#!/usr/bin/env python3
"""
Example 01: Basic Search — 基础搜索示例

Demonstrates:
- Text search / 文本搜索
- Regex search / 正则搜索
- Context lines / 上下文行数
- Output formats (TEXT, JSON, HIGHLIGHT) / 输出格式
- Search statistics / 搜索统计
- Count-only mode / 仅计数模式
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig
from pysearch.core.types import OutputFormat, Query
from pysearch.utils.formatter import format_result, render_highlight_console

# ---------------------------------------------------------------------------
# Helper: print a section header / 打印分节标题
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Create search engine / 创建搜索引擎
# ---------------------------------------------------------------------------


def demo_basic_text_search(engine: PySearch) -> None:
    """Simple text search for 'def search'. / 简单文本搜索"""
    section("1. Basic Text Search / 基础文本搜索")

    results = engine.search("def search")

    print(f"Found {len(results.items)} matches")
    print(f"Files scanned: {results.stats.files_scanned}")
    print(f"Files matched: {results.stats.files_matched}")
    print(f"Elapsed: {results.stats.elapsed_ms:.1f}ms\n")

    # Show first 3 matches / 展示前 3 个匹配
    for item in results.items[:3]:
        print(f"  {item.file}:{item.start_line}-{item.end_line}")
        for line in item.lines[:3]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 2. Regex search / 正则搜索
# ---------------------------------------------------------------------------


def demo_regex_search(engine: PySearch) -> None:
    """Regex search for function definitions matching a pattern."""
    section("2. Regex Search / 正则搜索")

    results = engine.search(r"def \w+_search", regex=True, context=3)

    print("Pattern: r'def \\w+_search'")
    print(f"Found {len(results.items)} matches in {results.stats.files_matched} files\n")

    for item in results.items[:3]:
        print(f"  {item.file}:{item.start_line}-{item.end_line}")
        for line in item.lines[:5]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 3. Output formats / 输出格式
# ---------------------------------------------------------------------------


def demo_output_formats(engine: PySearch) -> None:
    """Demonstrate different output formats (TEXT, JSON, HIGHLIGHT)."""
    section("3. Output Formats / 输出格式")

    results = engine.search("class PySearch", context=1)

    # TEXT format / 文本格式
    print("--- TEXT format ---")
    text_output = format_result(results, OutputFormat.TEXT)
    if isinstance(text_output, bytes):
        text_output = text_output.decode("utf-8")
    print(text_output[:500] if len(text_output) > 500 else text_output)

    # JSON format / JSON 格式
    print("\n--- JSON format (first 500 chars) ---")
    json_output = format_result(results, OutputFormat.JSON)
    if isinstance(json_output, bytes):
        json_output = json_output.decode("utf-8")
    print(json_output[:500])

    # HIGHLIGHT format / 高亮格式
    print("\n--- HIGHLIGHT format ---")
    try:
        from rich.console import Console

        console = Console()
        render_highlight_console(results, console)
    except ImportError:
        print("  (rich library not installed, skipping highlight output)")


# ---------------------------------------------------------------------------
# 4. Search with Query object / 使用 Query 对象搜索
# ---------------------------------------------------------------------------


def demo_query_object(engine: PySearch) -> None:
    """Build a Query object for fine-grained control."""
    section("4. Query Object / Query 对象控制")

    query = Query(
        pattern=r"class \w+Config",
        use_regex=True,
        context=2,
        output=OutputFormat.TEXT,
        max_per_file=3,
    )

    results = engine.run(query)
    print("Pattern: r'class \\w+Config'")
    print(f"Found {len(results.items)} matches (max 3 per file)\n")

    for item in results.items[:5]:
        print(f"  {item.file}:{item.start_line}")
        for line in item.lines[:3]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 5. Count-only search / 仅计数搜索
# ---------------------------------------------------------------------------


def demo_count_only(engine: PySearch) -> None:
    """Fast count-only search without returning content."""
    section("5. Count-Only Search / 仅计数搜索")

    count_result = engine.search_count_only("def ", regex=False)
    print(f"Total 'def ' occurrences: {count_result.total_matches}")
    print(f"Files containing 'def ': {count_result.files_matched}")
    print(f"Elapsed: {count_result.stats.elapsed_ms:.1f}ms")

    # Count with regex
    count_result2 = engine.search_count_only(r"class \w+:", regex=True)
    print(f"\nTotal 'class <Name>:' occurrences: {count_result2.total_matches}")
    print(f"Files: {count_result2.files_matched}")


# ---------------------------------------------------------------------------
# 6. Search with context / 带上下文搜索
# ---------------------------------------------------------------------------


def demo_context_lines(engine: PySearch) -> None:
    """Show how context lines affect output."""
    section("6. Context Lines / 上下文行数")

    for ctx in [0, 2, 5]:
        results = engine.search("def run(", context=ctx)
        if results.items:
            item = results.items[0]
            print(
                f"  context={ctx}: lines {item.start_line}-{item.end_line} "
                f"({len(item.lines)} lines shown)"
            )


# ---------------------------------------------------------------------------
# Main / 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    # Default search path: project's src/ directory
    # 默认搜索路径: 项目的 src/ 目录
    search_path = "./src"
    if len(sys.argv) > 1:
        search_path = sys.argv[1]

    if not Path(search_path).exists():
        print(f"Error: search path '{search_path}' does not exist.")
        print("Please run from the project root directory.")
        sys.exit(1)

    config = SearchConfig(
        paths=[search_path],
        include=["**/*.py"],
        context=2,
    )
    engine = PySearch(config)

    demo_basic_text_search(engine)
    demo_regex_search(engine)
    demo_output_formats(engine)
    demo_query_object(engine)
    demo_count_only(engine)
    demo_context_lines(engine)

    print("\n✓ All basic search examples completed.")


if __name__ == "__main__":
    main()
