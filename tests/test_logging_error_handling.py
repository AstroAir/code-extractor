import tempfile
from pathlib import Path

import pytest

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch.utils.error_handling import (
    EncodingError,
    ErrorCategory,
    ErrorCollector,
    ErrorSeverity,
    FileAccessError,
    SearchError,
    create_error_report,
    handle_file_error,
)

# Import custom PermissionError with alias to avoid conflict with built-in
from pysearch.utils.error_handling import PermissionError as CustomPermissionError
from pysearch.utils.logging_config import (
    LogFormat,
    LogLevel,
    SearchLogger,
    configure_logging,
    disable_logging,
    enable_debug_logging,
    get_logger,
)
from pysearch import OutputFormat


def test_search_logger_basic():
    """Test basic logging functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "test.log"

        logger = SearchLogger(
            name="test_logger",
            level=LogLevel.DEBUG,
            format_type=LogFormat.SIMPLE,
            log_file=log_file,
            enable_file=True,
            enable_console=False,
        )

        logger.info("Test info message")
        logger.warning("Test warning message")
        logger.error("Test error message")

        # Check that log file was created and contains messages
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Test info message" in log_content
        assert "Test warning message" in log_content
        assert "Test error message" in log_content


def test_search_logger_formats():
    """Test different log formats."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test JSON format
        json_log = Path(tmpdir) / "json.log"
        json_logger = SearchLogger(
            format_type=LogFormat.JSON, log_file=json_log, enable_file=True, enable_console=False
        )
        json_logger.info("JSON test message", operation="test", count=42)

        json_content = json_log.read_text()
        assert '"message": "JSON test message"' in json_content
        assert '"operation": "test"' in json_content
        assert '"count": 42' in json_content

        # Test structured format
        struct_log = Path(tmpdir) / "struct.log"
        struct_logger = SearchLogger(
            format_type=LogFormat.STRUCTURED,
            log_file=struct_log,
            enable_file=True,
            enable_console=False,
        )
        struct_logger.info("Structured test", key="value")

        struct_content = struct_log.read_text()
        assert "Structured test" in struct_content
        assert "key=value" in struct_content


def test_error_collector():
    """Test error collection and categorization."""
    collector = ErrorCollector()

    # Add different types of errors
    file_error = FileNotFoundError("File not found")
    permission_error = PermissionError("Permission denied")  # Built-in PermissionError
    encoding_error = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid start byte")

    collector.add_error(file_error, file_path=Path("test.py"))
    collector.add_error(permission_error, file_path=Path("restricted.py"))
    collector.add_error(encoding_error, file_path=Path("binary.py"))

    # Check error counts
    summary = collector.get_summary()
    assert summary["total_errors"] == 3
    assert summary["by_category"][ErrorCategory.FILE_ACCESS.value] == 1
    assert summary["by_category"][ErrorCategory.PERMISSION.value] == 1
    assert summary["by_category"][ErrorCategory.ENCODING.value] == 1

    # Test category filtering
    file_errors = collector.get_errors_by_category(ErrorCategory.FILE_ACCESS)
    assert len(file_errors) == 1
    assert "File not found" in file_errors[0].message


def test_custom_search_errors():
    """Test custom search error types."""
    # Test FileAccessError
    file_error = FileAccessError("Cannot read file", Path("test.py"))
    assert file_error.category == ErrorCategory.FILE_ACCESS
    assert file_error.severity == ErrorSeverity.MEDIUM
    assert file_error.file_path == Path("test.py")

    # Test PermissionError with suggestions
    perm_error = CustomPermissionError("Access denied", Path("restricted.py"))
    assert perm_error.category == ErrorCategory.PERMISSION
    assert perm_error.severity == ErrorSeverity.HIGH
    assert len(perm_error.suggestions) > 0
    assert "Check file permissions" in perm_error.suggestions

    # Test EncodingError
    enc_error = EncodingError("Invalid encoding", Path("binary.py"), encoding="utf-8")
    assert enc_error.category == ErrorCategory.ENCODING
    assert enc_error.severity == ErrorSeverity.LOW
    assert "utf-8" in str(enc_error.context)


def test_handle_file_error():
    """Test file error handling utility."""
    collector = ErrorCollector()

    # Test different exception types
    exceptions = [
        (FileNotFoundError("File not found"), ErrorCategory.FILE_ACCESS),
        (
            PermissionError("Permission denied"),
            ErrorCategory.PERMISSION,
        ),  # Built-in PermissionError
        (UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), ErrorCategory.ENCODING),
        (SyntaxError("Invalid syntax"), ErrorCategory.PARSING),
    ]

    for exception, expected_category in exceptions:
        handle_file_error(
            file_path=Path("test.py"),
            operation="read",
            exception=exception,
            error_collector=collector,
        )

    # Check that all errors were categorized correctly
    summary = collector.get_summary()
    assert summary["total_errors"] == len(exceptions)

    for _, expected_category in exceptions:
        assert summary["by_category"][expected_category.value] >= 1


def test_error_report_generation():
    """Test error report generation."""
    collector = ErrorCollector()

    # Add some errors
    collector.add_error(
        SearchError("Critical system error", severity=ErrorSeverity.CRITICAL),
        file_path=Path("critical.py"),
    )
    collector.add_error(FileAccessError("Cannot read file", Path("missing.py")))

    # Generate report
    report = create_error_report(collector)

    assert "Search Error Report" in report
    assert "Total errors: 2" in report
    assert "Critical errors: 1" in report
    assert "Critical system error" in report
    assert "General Suggestions" in report


def test_api_error_integration():
    """Test error handling integration with the API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a file with permission issues (simulate)
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # Create a directory that will cause issues
        problem_dir = tmp_path / "problem"
        problem_dir.mkdir()
        (problem_dir / "file.py").write_text("invalid content")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Configure logging to capture errors
        engine.configure_logging(level="DEBUG", format_type="simple")

        # Run search
        result = engine.search("hello", output=OutputFormat.JSON)

        # Check that we can get error information
        error_summary = engine.get_error_summary()
        assert isinstance(error_summary, dict)
        assert "total_errors" in error_summary

        # Test error report
        error_report = engine.get_error_report()
        assert isinstance(error_report, str)


def test_logging_configuration():
    """Test logging configuration through API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_file = tmp_path / "test.py"
        test_file.write_text("print('test')")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test different logging configurations
        log_file = tmp_path / "search.log"

        engine.configure_logging(
            level="DEBUG", format_type="detailed", log_file=str(log_file), enable_file=True
        )

        # Run a search to generate logs
        result = engine.search("test", output=OutputFormat.JSON)

        # Check that log file was created
        assert log_file.exists()
        log_content = log_file.read_text()
        assert "Starting search" in log_content or "Search completed" in log_content


def test_error_suppression():
    """Test error category suppression."""
    collector = ErrorCollector()

    # Add errors before suppression
    collector.add_error(FileNotFoundError("File 1"), file_path=Path("file1.py"))
    collector.add_error(
        PermissionError("Permission 1"), file_path=Path("file2.py")
    )  # Built-in PermissionError

    assert collector.get_summary()["total_errors"] == 2

    # Suppress file access errors
    collector.suppress_category(ErrorCategory.FILE_ACCESS)

    # Add more errors
    collector.add_error(
        FileNotFoundError("File 2"), file_path=Path("file3.py")
    )  # Should be suppressed
    collector.add_error(
        PermissionError("Permission 2"), file_path=Path("file4.py")
    )  # Should not be suppressed (built-in PermissionError)

    # Check that only permission errors were added
    summary = collector.get_summary()
    assert summary["total_errors"] == 3  # 2 original + 1 new permission error
    assert summary["by_category"][ErrorCategory.FILE_ACCESS.value] == 1  # Only the original one
    assert summary["by_category"][ErrorCategory.PERMISSION.value] == 2  # Both permission errors


def test_global_logging_functions():
    """Test global logging configuration functions."""
    # Test configure_logging
    logger = configure_logging(
        level=LogLevel.WARNING, format_type=LogFormat.SIMPLE, enable_console=True, enable_file=False
    )
    assert isinstance(logger, SearchLogger)
    assert logger.level == LogLevel.WARNING

    # Test get_logger
    global_logger = get_logger()
    assert isinstance(global_logger, SearchLogger)

    # Test enable_debug_logging
    enable_debug_logging()
    # Should not raise an exception

    # Test disable_logging
    disable_logging()
    # Should not raise an exception


if __name__ == "__main__":
    pytest.main([__file__])
