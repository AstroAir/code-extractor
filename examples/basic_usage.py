"""
Basic Usage Example for pysearch

This example demonstrates the fundamental usage of pysearch through its API,
showing how to configure the search engine and execute different types of searches.

Usage:
    python examples/basic_usage.py
"""

from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import ASTFilters, OutputFormat, Query


def basic_text_search() -> None:
    """Demonstrate basic text search functionality."""
    print("=== Basic Text Search ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(
        paths=[str(repo_root / "src")],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**"],
        context=2,  # Show 2 lines of context around matches
        output_format=OutputFormat.TEXT,
        enable_docstrings=True,
        enable_comments=True,
        enable_strings=False,
    )

    engine = PySearch(cfg)

    # Simple text search for function definitions
    results = engine.search("def main", regex=False)
    print(f"Found {len(results.items)} matches for 'def main'")
    print(f"Scanned {results.stats.files_scanned} files in {results.stats.elapsed_ms:.2f}ms")

    # Show first result as example
    if results.items:
        item = results.items[0]
        print(f"\nFirst match in {item.file}:")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            print(f"  {line_num:3d}: {line}")


def regex_search_example() -> None:
    """Demonstrate regex search capabilities."""
    print("\n=== Regex Search ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=1)
    engine = PySearch(cfg)

    # Search for function definitions with regex
    query = Query(
        pattern=r"def \w+_handler", use_regex=True, context=1  # Functions ending with "_handler"
    )

    results = engine.run(query)
    print(f"Found {len(results.items)} handler functions")

    # Show all matches
    for item in results.items[:3]:  # Show first 3 matches
        print(f"\nMatch in {item.file.name}:")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            marker = " -> " if any(span[0] == i for span in item.match_spans) else "    "
            print(f"{marker}{line_num:3d}: {line}")


def ast_search_example() -> None:
    """Demonstrate AST-based search with filters."""
    print("\n=== AST Search with Filters ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=2)
    engine = PySearch(cfg)

    # Search for functions with specific decorators
    filters = ASTFilters(
        func_name=r".*search.*",  # Functions with "search" in the name
        decorator=r"lru_cache",  # That have lru_cache decorator
    )

    query = Query(pattern="def", use_ast=True, filters=filters, context=2)

    results = engine.run(query)
    print(f"Found {len(results.items)} cached search functions")

    for item in results.items:
        print(f"\nCached function in {item.file.name}:")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            print(f"  {line_num:3d}: {line}")


def advanced_configuration_example() -> None:
    """Demonstrate advanced configuration options."""
    print("\n=== Advanced Configuration ===")

    repo_root = Path(__file__).resolve().parents[1]

    # Performance-optimized configuration
    cfg = SearchConfig(
        paths=[str(repo_root / "src"), str(repo_root / "tests")],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**", "**/build/**"],
        context=3,
        parallel=True,  # Enable parallel processing
        workers=4,  # Use 4 worker threads
        strict_hash_check=False,  # Fast mode - use mtime/size for change detection
        dir_prune_exclude=True,  # Skip excluded directories during traversal
        enable_docstrings=True,
        enable_comments=True,
        enable_strings=True,
    )

    engine = PySearch(cfg)

    # Search for class definitions
    results = engine.search(r"class \w+:", regex=True, context=3)
    print(f"Found {len(results.items)} class definitions")
    print(
        f"Performance: {results.stats.elapsed_ms:.2f}ms, {results.stats.files_scanned} files scanned"
    )


def main() -> None:
    """Run all example demonstrations."""
    print("pysearch API Examples")
    print("=" * 50)

    basic_text_search()
    regex_search_example()
    ast_search_example()
    advanced_configuration_example()

    print("\n" + "=" * 50)
    print("Examples completed! Try modifying the patterns and configurations above.")


if __name__ == "__main__":
    main()
