"""
Advanced Usage Examples for pysearch

This file demonstrates advanced features and use cases of pysearch including:
- Semantic search capabilities
- Metadata filtering
- Custom scoring and ranking
- Error handling and logging
- Performance optimization techniques

Usage:
    python examples/advanced_examples.py
"""

from __future__ import annotations

import time
from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.logging_config import enable_debug_logging
from pysearch.types import (
    ASTFilters,
    Language,
    MetadataFilters,
    Query,
)


def semantic_search_example() -> None:
    """Demonstrate semantic search for conceptually related code."""
    print("=== Semantic Search Example ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(
        paths=[str(repo_root / "src")],
        context=3,
        enable_docstrings=True,
        enable_comments=True,
    )

    engine = PySearch(cfg)

    # Search for database-related code using semantic matching
    query = Query(pattern="database connection", use_semantic=True, context=3)

    results = engine.run(query)
    print(f"Found {len(results.items)} semantically related matches")

    # Show results with semantic scores
    for item in results.items[:3]:
        print(f"\nSemantic match in {item.file.name}:")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            print(f"  {line_num:3d}: {line}")


def metadata_filtering_example() -> None:
    """Demonstrate advanced metadata-based filtering."""
    print("\n=== Metadata Filtering Example ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=2)
    engine = PySearch(cfg)

    # Create metadata filters
    metadata_filters = MetadataFilters(
        min_size=1000,  # Files larger than 1KB
        max_size=50000,  # Files smaller than 50KB
        min_lines=10,  # Files with at least 10 lines
        languages={Language.PYTHON},  # Only Python files
        # modified_after=time.time() - 86400,  # Modified in last 24 hours
    )

    query = Query(pattern="class", use_regex=False, metadata_filters=metadata_filters, context=2)

    results = engine.run(query)
    print(f"Found {len(results.items)} matches in filtered files")

    # Show file metadata for matches
    for item in results.items[:2]:
        file_size = item.file.stat().st_size if item.file.exists() else 0
        print(f"\nMatch in {item.file.name} ({file_size} bytes):")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            print(f"  {line_num:3d}: {line}")


def performance_optimization_example() -> None:
    """Demonstrate performance optimization techniques."""
    print("\n=== Performance Optimization Example ===")

    repo_root = Path(__file__).resolve().parents[1]

    # Configuration optimized for speed
    fast_cfg = SearchConfig(
        paths=[str(repo_root / "src")],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**"],
        context=1,
        parallel=True,
        workers=4,
        strict_hash_check=False,  # Fast mode
        dir_prune_exclude=True,  # Skip excluded directories
        enable_docstrings=False,  # Skip docstrings for speed
        enable_comments=False,  # Skip comments for speed
        enable_strings=False,  # Skip string literals for speed
    )

    # Configuration optimized for accuracy
    accurate_cfg = SearchConfig(
        paths=[str(repo_root / "src")],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**"],
        context=3,
        parallel=True,
        workers=2,
        strict_hash_check=True,  # Exact change detection
        dir_prune_exclude=True,
        enable_docstrings=True,
        enable_comments=True,
        enable_strings=True,
    )

    # Compare performance
    pattern = "def search"

    # Fast search
    start_time = time.perf_counter()
    fast_engine = PySearch(fast_cfg)
    fast_results = fast_engine.search(pattern, regex=False)
    fast_time = time.perf_counter() - start_time

    # Accurate search
    start_time = time.perf_counter()
    accurate_engine = PySearch(accurate_cfg)
    accurate_results = accurate_engine.search(pattern, regex=False)
    accurate_time = time.perf_counter() - start_time

    print(f"Fast search: {len(fast_results.items)} results in {fast_time*1000:.2f}ms")
    print(f"Accurate search: {len(accurate_results.items)} results in {accurate_time*1000:.2f}ms")
    print(f"Speed improvement: {accurate_time/fast_time:.1f}x faster")


def error_handling_example() -> None:
    """Demonstrate error handling and logging."""
    print("\n=== Error Handling Example ===")

    # Enable debug logging
    enable_debug_logging()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(
        paths=[str(repo_root / "src"), "/nonexistent/path"],  # Include invalid path
        include=["**/*.py"],
        context=2,
    )

    engine = PySearch(cfg)

    try:
        # This will handle the invalid path gracefully
        results = engine.search("def", regex=False)
        print(f"Search completed with {len(results.items)} results")
        print(f"Files scanned: {results.stats.files_scanned}")

        # Check for errors
        if hasattr(engine, "error_collector") and engine.error_collector.has_errors():
            print("Errors encountered during search:")
            for error in engine.error_collector.get_errors():
                print(f"  - {error}")

    except Exception as e:
        print(f"Search failed with error: {e}")


def complex_query_example() -> None:
    """Demonstrate complex queries combining multiple search modes."""
    print("\n=== Complex Query Example ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=3)
    engine = PySearch(cfg)

    # Complex AST filters
    filters = ASTFilters(
        func_name=r"(search|find|match).*",  # Functions related to searching
        class_name=r".*Search.*",  # Classes with "Search" in name
        decorator=r"(lru_cache|cache|property)",  # Common decorators
    )

    # Metadata filters
    metadata_filters = MetadataFilters(
        min_lines=20,  # Substantial files only
        languages={Language.PYTHON},
    )

    query = Query(
        pattern="search",
        use_regex=True,
        use_ast=True,
        filters=filters,
        metadata_filters=metadata_filters,
        context=3,
        search_docstrings=True,
        search_comments=True,
        search_strings=False,
    )

    results = engine.run(query)
    print(f"Complex query found {len(results.items)} matches")

    # Group results by file
    files_with_matches = {}
    for item in results.items:
        file_name = item.file.name
        if file_name not in files_with_matches:
            files_with_matches[file_name] = []
        files_with_matches[file_name].append(item)

    print(f"Matches found in {len(files_with_matches)} files:")
    for file_name, items in files_with_matches.items():
        print(f"  {file_name}: {len(items)} matches")


def main() -> None:
    """Run all advanced example demonstrations."""
    print("pysearch Advanced Examples")
    print("=" * 60)

    semantic_search_example()
    metadata_filtering_example()
    performance_optimization_example()
    error_handling_example()
    complex_query_example()

    print("\n" + "=" * 60)
    print("Advanced examples completed!")
    print("These examples show the full power of pysearch for complex search scenarios.")


if __name__ == "__main__":
    main()
