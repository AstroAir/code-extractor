#!/usr/bin/env python3
"""
Example 05: Semantic Search — 语义搜索示例

Demonstrates:
- Concept-based semantic search / 概念级语义搜索
- TF-IDF embedding-based semantic search / 基于 TF-IDF 嵌入的语义搜索
- Comparing semantic vs text search / 语义搜索与文本搜索对比
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig

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
# 1. Concept-based semantic search / 概念级语义搜索
# ---------------------------------------------------------------------------


def demo_concept_search(engine: PySearch) -> None:
    section("1. Concept-Based Semantic Search / 概念级语义搜索")

    # Search by concept — pysearch expands concepts into related patterns
    # 按概念搜索 — pysearch 将概念扩展为相关模式
    concepts = [
        "error handling",
        "file operations",
        "caching",
        "configuration",
    ]

    for concept in concepts:
        results = engine.semantic_search(concept)
        print(f"  Concept: '{concept}'")
        print(f"    Matches: {len(results.items)} in {results.stats.files_matched} files")

        if results.items:
            item = results.items[0]
            first_line = item.lines[0].rstrip() if item.lines else ""
            print(f"    Top hit: {item.file}:{item.start_line} — {first_line[:60]}")
        print()


# ---------------------------------------------------------------------------
# 2. TF-IDF embedding search / TF-IDF 嵌入搜索
# ---------------------------------------------------------------------------


def demo_tfidf_search(engine: PySearch) -> None:
    section("2. TF-IDF Embedding Search / TF-IDF 嵌入搜索")

    # search_semantic uses vector-based similarity with TF-IDF
    # search_semantic 使用基于 TF-IDF 的向量相似度
    queries = [
        "database connection pooling",
        "parallel processing workers",
        "search result ranking",
    ]

    for query in queries:
        results = engine.search_semantic(
            query=query,
            threshold=0.05,  # low threshold to show more results
            max_results=5,
        )
        print(f"  Query: '{query}'")
        print(
            f"    Matches: {len(results.items)} "
            f"(scanned {results.stats.files_scanned} files, "
            f"{results.stats.elapsed_ms:.0f}ms)"
        )

        for item in results.items[:2]:
            first_line = item.lines[0].rstrip() if item.lines else ""
            print(f"    → {item.file}:{item.start_line} — {first_line[:60]}")
        print()


# ---------------------------------------------------------------------------
# 3. Semantic vs text search comparison / 语义搜索与文本搜索对比
# ---------------------------------------------------------------------------


def demo_semantic_vs_text(engine: PySearch) -> None:
    section("3. Semantic vs Text Search / 语义搜索 vs 文本搜索")

    query = "error handling"

    # Text search — literal match only
    # 文本搜索 — 仅匹配字面文本
    text_results = engine.search(query)
    print(f"  Text search for '{query}':")
    print(f"    Found {len(text_results.items)} matches\n")

    # Semantic search — expands to related concepts (try/except/raise/error/etc.)
    # 语义搜索 — 扩展到相关概念
    semantic_results = engine.semantic_search(query)
    print(f"  Semantic search for '{query}':")
    print(f"    Found {len(semantic_results.items)} matches\n")

    # Show unique files from semantic that text search missed
    text_files = {str(item.file) for item in text_results.items}
    semantic_only = [item for item in semantic_results.items if str(item.file) not in text_files]
    if semantic_only:
        print("  Files found by semantic search but NOT text search:")
        for item in semantic_only[:5]:
            first_line = item.lines[0].rstrip() if item.lines else ""
            print(f"    → {item.file}:{item.start_line} — {first_line[:60]}")
    else:
        print("  (No additional files found by semantic search)")


# ---------------------------------------------------------------------------
# 4. Semantic search with custom threshold / 自定义阈值
# ---------------------------------------------------------------------------


def demo_threshold_tuning(engine: PySearch) -> None:
    section("4. Threshold Tuning / 阈值调优")

    query = "indexing pipeline"
    thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]

    print(f"  Query: '{query}'\n")
    for threshold in thresholds:
        results = engine.search_semantic(query=query, threshold=threshold, max_results=100)
        print(f"    threshold={threshold:.2f}: {len(results.items)} matches")

    print("\n  Lower threshold = more results (higher recall, lower precision)")
    print("  Higher threshold = fewer results (lower recall, higher precision)")


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

    demo_concept_search(engine)
    demo_tfidf_search(engine)
    demo_semantic_vs_text(engine)
    demo_threshold_tuning(engine)

    print("\n✓ All semantic search examples completed.")


if __name__ == "__main__":
    main()
