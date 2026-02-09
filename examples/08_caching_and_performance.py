#!/usr/bin/env python3
"""
Example 08: Caching & Performance — 缓存与性能示例

Demonstrates:
- Memory and disk caching / 内存与磁盘缓存
- Cache statistics and hit rate / 缓存统计与命中率
- Ranking strategies / 排序策略
- Result clustering / 结果聚类
- Result grouping by file / 按文件分组
- Ranking analysis / 排序分析
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pysearch import PySearch, SearchConfig

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Memory caching / 内存缓存
# ---------------------------------------------------------------------------


def demo_memory_caching(engine: PySearch) -> None:
    section("1. Memory Caching / 内存缓存")

    # Enable memory caching
    success = engine.enable_caching(backend="memory", max_size=1000, default_ttl=3600)
    print(f"  Caching enabled: {success}")
    print(f"  Is caching enabled: {engine.is_caching_enabled()}")

    # First search — cache miss
    results1 = engine.search("def search", context=2)
    print(f"\n  1st search: {len(results1.items)} matches, " f"{results1.stats.elapsed_ms:.1f}ms")

    # Second search — cache hit (should be faster)
    results2 = engine.search("def search", context=2)
    print(f"  2nd search: {len(results2.items)} matches, " f"{results2.stats.elapsed_ms:.1f}ms")

    # Cache stats
    stats = engine.get_cache_stats()
    print("\n  Cache stats:")
    for key, value in stats.items():
        if isinstance(value, int | float | str | bool):
            print(f"    {key}: {value}")

    hit_rate = engine.get_cache_hit_rate()
    print(f"\n  Cache hit rate: {hit_rate:.1f}%")

    # Clear cache
    engine.clear_cache()
    print("  Cache cleared.")


# ---------------------------------------------------------------------------
# 2. Disk caching / 磁盘缓存
# ---------------------------------------------------------------------------


def demo_disk_caching(engine: PySearch) -> None:
    section("2. Disk Caching / 磁盘缓存")

    with tempfile.TemporaryDirectory() as tmpdir:
        success = engine.enable_caching(
            backend="disk",
            cache_dir=tmpdir,
            max_size=5000,
            default_ttl=7200,
            compression=True,
        )
        print(f"  Disk caching enabled: {success}")
        print(f"  Cache directory: {tmpdir}")

        # Perform searches
        results = engine.search("class PySearch")
        print(f"  Search: {len(results.items)} matches")

        stats = engine.get_cache_stats()
        print(f"  Cache entries: {stats.get('total_entries', 'N/A')}")

    # Disable caching
    engine.disable_caching()
    print("  Disk caching disabled.")


# ---------------------------------------------------------------------------
# 3. Cache TTL and invalidation / 缓存 TTL 与失效
# ---------------------------------------------------------------------------


def demo_cache_ttl(engine: PySearch) -> None:
    section("3. Cache TTL & Invalidation / 缓存 TTL 与失效")

    engine.enable_caching(backend="memory", default_ttl=60)

    # Set custom TTL
    engine.set_cache_ttl(120)
    print("  TTL set to 120 seconds")

    # Search and cache
    engine.search("import pathlib")
    print("  Search cached.")

    # Invalidate cache for a specific file
    engine.invalidate_cache_for_file("src/pysearch/core/api.py")
    print("  Invalidated cache for api.py")

    # Check stats after invalidation
    stats = engine.get_cache_stats()
    print(f"  Cache entries after invalidation: {stats.get('total_entries', 'N/A')}")

    engine.disable_caching()


# ---------------------------------------------------------------------------
# 4. Ranking strategies / 排序策略
# ---------------------------------------------------------------------------


def demo_ranking_strategies(engine: PySearch) -> None:
    section("4. Ranking Strategies / 排序策略")

    strategies = ["relevance", "frequency", "recency", "popularity", "hybrid"]

    for strategy in strategies:
        results = engine.search_with_ranking(
            pattern="def ",
            ranking_strategy=strategy,
            regex=False,
        )
        # Show top 3 file:line for each strategy
        top_files = [f"{item.file.name}:{item.start_line}" for item in results.items[:3]]
        print(f"  {strategy:12s}: {len(results.items)} matches — top: {', '.join(top_files)}")


# ---------------------------------------------------------------------------
# 5. Result clustering / 结果聚类
# ---------------------------------------------------------------------------


def demo_result_clustering(engine: PySearch) -> None:
    section("5. Result Clustering / 结果聚类")

    results = engine.search("def ", context=1)

    # Cluster by similarity
    clusters = engine.get_result_clusters(results, similarity_threshold=0.8)
    print(f"  Total results: {len(results.items)}")
    print(f"  Clusters (threshold=0.8): {len(clusters)}")

    for i, cluster in enumerate(clusters[:5], 1):
        files = {item.file.name for item in cluster}
        print(f"    Cluster {i}: {len(cluster)} items from {files}")

    # Search with clustering enabled
    results_clustered = engine.search_with_ranking(
        pattern="class ",
        ranking_strategy="hybrid",
        cluster_results=True,
    )
    print(f"\n  Clustered search for 'class ': {len(results_clustered.items)} items")


# ---------------------------------------------------------------------------
# 6. Results grouped by file / 按文件分组
# ---------------------------------------------------------------------------


def demo_grouped_results(engine: PySearch) -> None:
    section("6. Results Grouped by File / 按文件分组")

    results = engine.search("self.", context=0)
    grouped = engine.get_results_grouped_by_file(results)

    print(f"  Total matches: {len(results.items)}")
    print(f"  Files with matches: {len(grouped)}\n")

    for file_path, items in list(grouped.items())[:5]:
        print(f"  {file_path.name}: {len(items)} matches")
        for item in items[:2]:
            line = item.lines[0].rstrip() if item.lines else ""
            print(f"    L{item.start_line}: {line[:70]}")


# ---------------------------------------------------------------------------
# 7. Ranking analysis / 排序分析
# ---------------------------------------------------------------------------


def demo_ranking_analysis(engine: PySearch) -> None:
    section("7. Ranking Analysis / 排序分析")

    results = engine.search("def search")
    suggestions = engine.get_ranking_suggestions("def search", results)

    print(f"  Query type: {suggestions.get('query_type', 'unknown')}")
    print(f"  Recommended strategy: {suggestions.get('recommended_strategy', 'N/A')}")
    print(f"  File spread: {suggestions.get('file_spread', 0)}")
    print(f"  Result diversity: {suggestions.get('result_diversity', 0):.2f}")

    if suggestions.get("suggestions"):
        print("\n  Suggestions:")
        for s in suggestions["suggestions"]:
            print(f"    - {s}")


# ---------------------------------------------------------------------------
# 8. Indexer stats and cache cleanup / 索引器统计与缓存清理
# ---------------------------------------------------------------------------


def demo_indexer_stats(engine: PySearch) -> None:
    section("8. Indexer Stats / 索引器统计")

    stats = engine.get_indexer_stats()
    print("  Indexer cache statistics:")
    for key, value in stats.items():
        if isinstance(value, int | float | str | bool):
            print(f"    {key}: {value}")

    # Cleanup old cache entries
    removed = engine.cleanup_old_cache_entries(days_old=30)
    print(f"\n  Cleaned up cache entries older than 30 days: {removed}")


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

    demo_memory_caching(engine)
    demo_disk_caching(engine)
    demo_cache_ttl(engine)
    demo_ranking_strategies(engine)
    demo_result_clustering(engine)
    demo_grouped_results(engine)
    demo_ranking_analysis(engine)
    demo_indexer_stats(engine)

    print("\n✓ All caching & performance examples completed.")


if __name__ == "__main__":
    main()
