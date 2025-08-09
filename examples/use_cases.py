"""
Real-world Use Cases for pysearch

This file demonstrates practical scenarios where pysearch can be valuable
for code analysis, refactoring, debugging, and maintenance tasks.

Usage:
    python examples/use_cases.py
"""

from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query, SearchItem


def find_deprecated_apis() -> None:
    """Find usage of deprecated APIs or functions."""
    print("=== Finding Deprecated API Usage ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=2)
    engine = PySearch(cfg)

    # List of deprecated patterns to search for
    deprecated_patterns = [
        r"\.iteritems\(\)",  # Python 2 dict method
        r"\.iterkeys\(\)",  # Python 2 dict method
        r"\.itervalues\(\)",  # Python 2 dict method
        r"imp\.load_source",  # Deprecated import method
        r"platform\.dist\(\)",  # Deprecated platform method
    ]

    total_issues = 0
    for pattern in deprecated_patterns:
        query = Query(pattern=pattern, use_regex=True, context=2)
        results = engine.run(query)

        if results.items:
            print(f"\nFound {len(results.items)} uses of deprecated pattern: {pattern}")
            for item in results.items:
                print(f"  {item.file.name}:{item.start_line}")
                total_issues += 1

    print(f"\nTotal deprecated API usage found: {total_issues}")


def analyze_error_handling() -> None:
    """Analyze error handling patterns in the codebase."""
    print("\n=== Error Handling Analysis ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=3)
    engine = PySearch(cfg)

    # Find different error handling patterns
    patterns = {
        "try_blocks": r"try:",
        "except_clauses": r"except\s+\w+",
        "bare_except": r"except:",
        "finally_blocks": r"finally:",
        "raise_statements": r"raise\s+\w+",
    }

    error_stats = {}
    for name, pattern in patterns.items():
        query = Query(pattern=pattern, use_regex=True, context=1)
        results = engine.run(query)
        error_stats[name] = len(results.items)

        if name == "bare_except" and results.items:
            print(f"\nWarning: Found {len(results.items)} bare except clauses:")
            for item in results.items[:3]:  # Show first 3
                print(f"  {item.file.name}:{item.start_line}")

    print("\nError handling statistics:")
    for name, count in error_stats.items():
        print(f"  {name.replace('_', ' ').title()}: {count}")


def find_security_issues() -> None:
    """Find potential security issues in the code."""
    print("\n=== Security Issue Detection ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=2)
    engine = PySearch(cfg)

    # Security-related patterns to check
    security_patterns = {
        "sql_injection": r"(execute|query).*%.*%",
        "command_injection": r"(os\.system|subprocess\.call).*\+",
        "hardcoded_passwords": r"(password|passwd|pwd)\s*=\s*['\"][^'\"]+['\"]",
        "eval_usage": r"\beval\s*\(",
        "exec_usage": r"\bexec\s*\(",
        "pickle_usage": r"pickle\.loads?",
    }

    security_issues = 0
    for issue_type, pattern in security_patterns.items():
        query = Query(pattern=pattern, use_regex=True, context=2)
        results = engine.run(query)

        if results.items:
            print(f"\nPotential {issue_type.replace('_', ' ')} issues:")
            for item in results.items:
                print(f"  {item.file.name}:{item.start_line}")
                security_issues += 1

    if security_issues == 0:
        print("\nNo obvious security issues found.")
    else:
        print(f"\nTotal potential security issues: {security_issues}")


def analyze_code_complexity() -> None:
    """Analyze code complexity indicators."""
    print("\n=== Code Complexity Analysis ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=1)
    engine = PySearch(cfg)

    # Complexity indicators
    complexity_patterns = {
        "nested_loops": r"for.*:\s*\n.*for.*:",
        "deep_nesting": r"if.*:\s*\n.*if.*:\s*\n.*if.*:",
        "long_functions": r"def\s+\w+.*:",  # We'll count lines separately
        "multiple_returns": r"return\s+",
        "global_variables": r"global\s+\w+",
    }

    complexity_stats = {}
    for name, pattern in complexity_patterns.items():
        query = Query(pattern=pattern, use_regex=True, context=1)
        results = engine.run(query)
        complexity_stats[name] = len(results.items)

    print("Code complexity indicators:")
    for name, count in complexity_stats.items():
        print(f"  {name.replace('_', ' ').title()}: {count}")

    # Find functions with many parameters (complexity indicator)
    query = Query(
        pattern=r"def\s+\w+\([^)]{50,}\)",  # Functions with long parameter lists
        use_regex=True,
        context=1,
    )
    results = engine.run(query)
    if results.items:
        print(f"\nFunctions with many parameters: {len(results.items)}")
        for item in results.items[:3]:
            print(f"  {item.file.name}:{item.start_line}")


def find_code_duplication() -> None:
    """Find potential code duplication patterns."""
    print("\n=== Code Duplication Detection ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=0)
    engine = PySearch(cfg)

    # Look for similar function signatures
    query = Query(pattern=r"def\s+\w+\(.*\):", use_regex=True, context=0)
    results = engine.run(query)

    # Group by function signature patterns
    signature_patterns: dict[str, list[SearchItem]] = {}
    for item in results.items:
        if item.lines:
            # Extract function signature
            line = item.lines[0].strip()
            # Normalize by removing specific names but keeping structure
            normalized = line.replace(r"\w+", "NAME")  # Simple normalization

            if normalized not in signature_patterns:
                signature_patterns[normalized] = []
            signature_patterns[normalized].append(item)

    # Find patterns that appear multiple times
    duplicates = {pattern: items for pattern, items in signature_patterns.items() if len(items) > 1}

    if duplicates:
        print(f"Found {len(duplicates)} potentially duplicated function patterns:")
        for pattern, items in list(duplicates.items())[:3]:  # Show first 3
            print(f"\nPattern appears {len(items)} times:")
            print(f"  {pattern}")
            for item in items[:3]:  # Show first 3 occurrences
                print(f"    {item.file.name}:{item.start_line}")
    else:
        print("No obvious function duplication patterns found.")


def analyze_dependencies() -> None:
    """Analyze import dependencies and usage patterns."""
    print("\n=== Dependency Analysis ===")

    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(paths=[str(repo_root / "src")], context=0)
    engine = PySearch(cfg)

    # Find all import statements
    import_patterns = [
        r"^import\s+(\w+)",
        r"^from\s+(\w+)\s+import",
    ]

    all_imports: set[str] = set()
    for pattern in import_patterns:
        query = Query(pattern=pattern, use_regex=True, context=0)
        results = engine.run(query)

        for item in results.items:
            if item.lines:
                line = item.lines[0].strip()
                # Extract module name (simplified)
                if line.startswith("import "):
                    module = line.split()[1].split(".")[0]
                    all_imports.add(module)
                elif line.startswith("from "):
                    module = line.split()[1].split(".")[0]
                    all_imports.add(module)

    print(f"Found {len(all_imports)} unique top-level imports:")

    # Categorize imports
    stdlib_modules = {
        "os",
        "sys",
        "time",
        "json",
        "pathlib",
        "typing",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "collections",
        "threading",
        "multiprocessing",
        "concurrent",
        "asyncio",
        "re",
        "math",
    }

    stdlib_imports = all_imports & stdlib_modules
    third_party_imports = all_imports - stdlib_modules - {"pysearch"}

    print(f"  Standard library: {len(stdlib_imports)}")
    print(f"  Third-party: {len(third_party_imports)}")

    if third_party_imports:
        print(f"  Third-party modules: {', '.join(sorted(third_party_imports))}")


def main() -> None:
    """Run all use case demonstrations."""
    print("pysearch Real-world Use Cases")
    print("=" * 50)

    find_deprecated_apis()
    analyze_error_handling()
    find_security_issues()
    analyze_code_complexity()
    find_code_duplication()
    analyze_dependencies()

    print("\n" + "=" * 50)
    print("Use case analysis completed!")
    print("\nThese examples show how pysearch can be used for:")
    print("- Code quality analysis")
    print("- Security auditing")
    print("- Refactoring assistance")
    print("- Dependency management")
    print("- Technical debt identification")


if __name__ == "__main__":
    main()
