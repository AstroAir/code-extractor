#!/usr/bin/env python3
"""
Example 12: Error Handling & Logging — 错误处理与日志示例

Demonstrates:
- Logging configuration (level, format, file) / 日志配置
- Debug logging / 调试日志
- ErrorCollector for error tracking / 错误收集器
- Error reports and summaries / 错误报告与摘要
- Error category filtering and suppression / 错误分类与抑制
- Critical error detection / 严重错误检测
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
# 1. Configure logging / 配置日志
# ---------------------------------------------------------------------------


def demo_logging_config(engine: PySearch) -> None:
    section("1. Logging Configuration / 日志配置")

    # Set log level and format
    # 设置日志级别和格式
    engine.configure_logging(
        level="INFO",
        format_type="simple",
    )
    print("  Logging configured: level=INFO, format=simple")

    # Perform a search to see log output
    results = engine.search("def main")
    print(f"  Search completed: {len(results.items)} matches")

    # Switch to detailed format
    engine.configure_logging(
        level="WARNING",
        format_type="detailed",
    )
    print("  Logging changed: level=WARNING, format=detailed")

    # JSON format logging
    engine.configure_logging(
        level="INFO",
        format_type="json",
    )
    print("  Logging changed: level=INFO, format=json")


# ---------------------------------------------------------------------------
# 2. File logging / 文件日志
# ---------------------------------------------------------------------------


def demo_file_logging(engine: PySearch) -> None:
    section("2. File Logging / 文件日志")

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w") as f:
        log_path = f.name

    engine.configure_logging(
        level="DEBUG",
        format_type="structured",
        log_file=log_path,
        enable_file=True,
    )
    print(f"  File logging enabled: {log_path}")

    # Perform search to generate log entries
    engine.search("class Config")
    print("  Search executed — log entries written to file.")

    # Read and display log file
    log_content = Path(log_path).read_text(errors="ignore")
    lines = log_content.strip().splitlines()
    print(f"  Log file has {len(lines)} lines")
    for line in lines[:5]:
        print(f"    {line[:100]}")
    if len(lines) > 5:
        print(f"    ... and {len(lines) - 5} more lines")

    # Reset logging
    engine.configure_logging(level="INFO", format_type="simple")
    Path(log_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Debug logging / 调试日志
# ---------------------------------------------------------------------------


def demo_debug_logging(engine: PySearch) -> None:
    section("3. Debug Logging / 调试日志")

    # Enable debug logging for troubleshooting
    # 启用调试日志以排查问题
    engine.enable_debug_logging()
    print("  Debug logging enabled.")

    # Perform search — more verbose output
    results = engine.search("import sys", context=0)
    print(f"  Search completed: {len(results.items)} matches")

    # Disable logging entirely
    engine.disable_logging()
    print("  Logging disabled.")

    # Re-enable normal logging
    engine.configure_logging(level="INFO", format_type="simple")
    print("  Normal logging restored.")


# ---------------------------------------------------------------------------
# 4. Error summary and report / 错误摘要与报告
# ---------------------------------------------------------------------------


def demo_error_summary(engine: PySearch) -> None:
    section("4. Error Summary & Report / 错误摘要与报告")

    # Clear any previous errors
    engine.clear_errors()
    print("  Errors cleared.")

    # Try to search in a non-existent path to trigger errors
    bad_config = SearchConfig(paths=["./non_existent_path_xyz"], include=["**/*.py"])
    bad_engine = PySearch(bad_config)
    _ = bad_engine.search("test")

    # Get error summary
    summary = bad_engine.get_error_summary()
    print("  Error summary:")
    for key, value in summary.items():
        if isinstance(value, (int, float, str, bool)):
            print(f"    {key}: {value}")

    # Get detailed report
    report = bad_engine.get_error_report()
    print("\n  Error report (first 500 chars):")
    print(f"    {report[:500]}")

    # Check for critical errors
    has_critical = bad_engine.has_critical_errors()
    print(f"\n  Has critical errors: {has_critical}")


# ---------------------------------------------------------------------------
# 5. Error categories / 错误分类
# ---------------------------------------------------------------------------


def demo_error_categories(engine: PySearch) -> None:
    section("5. Error Categories / 错误分类")

    # Get errors by category
    # 按类别获取错误
    categories = ["io", "parsing", "configuration", "search", "indexing"]

    for category in categories:
        errors = engine.get_errors_by_category(category)
        if errors:
            print(f"  {category}: {len(errors)} errors")
            for err in errors[:2]:
                print(f"    - {err}")
        else:
            print(f"  {category}: no errors")


# ---------------------------------------------------------------------------
# 6. Error suppression / 错误抑制
# ---------------------------------------------------------------------------


def demo_error_suppression(engine: PySearch) -> None:
    section("6. Error Suppression / 错误抑制")

    # Suppress a specific error category
    # 抑制特定错误类别
    engine.suppress_error_category("io")
    print("  Suppressed 'io' error category.")

    # Subsequent I/O errors will not be collected
    results = engine.search("test pattern")
    print(f"  Search completed: {len(results.items)} matches")

    summary = engine.get_error_summary()
    print(f"  Errors after suppression: {summary}")


# ---------------------------------------------------------------------------
# 7. Using pysearch exception classes / 使用异常类
# ---------------------------------------------------------------------------


def demo_exception_classes() -> None:
    section("7. Exception Classes / 异常类")

    from pysearch import (
        EncodingError,
        FileAccessError,
        ParsingError,
        SearchError,
    )

    # Demonstrate exception hierarchy
    # 演示异常层级
    dummy_path = Path("example.py")
    exceptions = [
        SearchError("General search error"),
        FileAccessError("Cannot read file", file_path=dummy_path),
        EncodingError("Invalid encoding", file_path=dummy_path, encoding="utf-8"),
        ParsingError("Failed to parse AST", file_path=dummy_path, line_number=42),
    ]

    for exc in exceptions:
        print(f"  {type(exc).__name__}: {exc}")
        print(f"    Is SearchError: {isinstance(exc, SearchError)}")

    # Example: catching specific exceptions
    print("\n  Exception handling pattern:")
    print("    try:")
    print("        engine.search('pattern')")
    print("    except FileAccessError as e:")
    print("        print(f'File error: {e}')")
    print("    except ParsingError as e:")
    print("        print(f'Parse error: {e}')")
    print("    except SearchError as e:")
    print("        print(f'Search error: {e}')")


# ---------------------------------------------------------------------------
# 8. Logging utilities / 日志工具
# ---------------------------------------------------------------------------


def demo_logging_utilities() -> None:
    section("8. Logging Utilities / 日志工具")

    from pysearch import configure_logging, get_logger

    # Get a custom logger
    logger = get_logger()
    print(f"  Logger: {logger}")

    # Configure global logging
    configure_logging()
    print("  Global logging configured with defaults.")

    # These are module-level utilities for quick setup
    # 这些是模块级工具，用于快速设置
    print("\n  Available logging utilities:")
    print("    configure_logging() — Full configuration")
    print("    enable_debug_logging() — Quick debug mode")
    print("    disable_logging() — Silence all logging")
    print("    get_logger() — Get logger instance")


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

    demo_logging_config(engine)
    demo_file_logging(engine)
    demo_debug_logging(engine)
    demo_error_summary(engine)
    demo_error_categories(engine)
    demo_error_suppression(engine)
    demo_exception_classes()
    demo_logging_utilities()

    print("\n✓ All error handling & logging examples completed.")


if __name__ == "__main__":
    main()
