"""
Expanded comprehensive tests for error handling module.

This module tests error handling functionality that is currently not covered,
including edge cases, error classification, recovery mechanisms, and advanced features.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pysearch.error_handling import (
    ErrorCategory,
    ErrorCollector,
    ErrorInfo,
    ErrorSeverity,
    SearchError,
    FileAccessError,
    PermissionError,
    EncodingError,
    ParsingError,
    ConfigurationError,
    handle_file_error,
    create_error_report,
)


class TestErrorSeverityAndCategory:
    """Test ErrorSeverity and ErrorCategory enums."""

    def test_error_severity_values(self) -> None:
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.HIGH == "high"
        assert ErrorSeverity.CRITICAL == "critical"

    def test_error_category_values(self) -> None:
        """Test ErrorCategory enum values."""
        assert ErrorCategory.FILE_ACCESS == "file_access"
        assert ErrorCategory.PERMISSION == "permission"
        assert ErrorCategory.ENCODING == "encoding"
        assert ErrorCategory.PARSING == "parsing"
        assert ErrorCategory.MEMORY == "memory"
        assert ErrorCategory.TIMEOUT == "timeout"
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.NETWORK == "network"
        assert ErrorCategory.VALIDATION == "validation"
        assert ErrorCategory.UNKNOWN == "unknown"

    def test_enum_string_inheritance(self) -> None:
        """Test that enums inherit from str."""
        assert isinstance(ErrorSeverity.LOW, str)
        assert isinstance(ErrorCategory.FILE_ACCESS, str)


class TestErrorInfo:
    """Test ErrorInfo dataclass."""

    def test_error_info_creation(self) -> None:
        """Test ErrorInfo creation with required fields."""
        error_info = ErrorInfo(
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.HIGH,
            message="Test error message"
        )
        
        assert error_info.category == ErrorCategory.FILE_ACCESS
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.message == "Test error message"
        assert error_info.file_path is None
        assert error_info.line_number is None
        assert error_info.exception_type is None
        assert error_info.traceback_str is None
        assert isinstance(error_info.timestamp, float)
        assert error_info.context == {}
        assert error_info.suggestions == []

    def test_error_info_with_all_fields(self) -> None:
        """Test ErrorInfo creation with all fields."""
        test_path = Path("test.py")
        test_context = {"key": "value"}
        test_suggestions = ["suggestion1", "suggestion2"]
        
        error_info = ErrorInfo(
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.CRITICAL,
            message="Parsing failed",
            file_path=test_path,
            line_number=42,
            exception_type="SyntaxError",
            traceback_str="Traceback...",
            context=test_context,
            suggestions=test_suggestions
        )
        
        assert error_info.category == ErrorCategory.PARSING
        assert error_info.severity == ErrorSeverity.CRITICAL
        assert error_info.message == "Parsing failed"
        assert error_info.file_path == test_path
        assert error_info.line_number == 42
        assert error_info.exception_type == "SyntaxError"
        assert error_info.traceback_str == "Traceback..."
        assert error_info.context == test_context
        assert error_info.suggestions == test_suggestions

    def test_error_info_timestamp_default(self) -> None:
        """Test that ErrorInfo gets a default timestamp."""
        before = time.time()
        error_info = ErrorInfo(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.LOW,
            message="Test"
        )
        after = time.time()
        
        assert before <= error_info.timestamp <= after


class TestSearchError:
    """Test SearchError base exception class."""

    def test_search_error_basic(self) -> None:
        """Test basic SearchError creation."""
        error = SearchError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.category == ErrorCategory.UNKNOWN
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.file_path is None
        assert error.suggestions == []
        assert error.context == {}
        assert isinstance(error.timestamp, float)

    def test_search_error_with_all_parameters(self) -> None:
        """Test SearchError with all parameters."""
        test_path = Path("test.py")
        test_suggestions = ["suggestion1", "suggestion2"]
        test_context = {"key": "value"}
        
        error = SearchError(
            message="Custom error",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.HIGH,
            file_path=test_path,
            suggestions=test_suggestions,
            context=test_context
        )
        
        assert error.message == "Custom error"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.file_path == test_path
        assert error.suggestions == test_suggestions
        assert error.context == test_context

    def test_search_error_inheritance(self) -> None:
        """Test that SearchError inherits from Exception."""
        error = SearchError("Test")
        assert isinstance(error, Exception)


class TestSpecificErrorTypes:
    """Test specific error type classes."""

    def test_file_access_error(self) -> None:
        """Test FileAccessError creation."""
        test_path = Path("missing.py")
        error = FileAccessError("File not found", test_path)
        
        assert error.message == "File not found"
        assert error.category == ErrorCategory.FILE_ACCESS
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.file_path == test_path
        assert len(error.suggestions) > 0
        assert "Check if file exists" in error.suggestions

    def test_permission_error(self) -> None:
        """Test PermissionError creation."""
        test_path = Path("protected.py")
        error = PermissionError("Access denied", test_path)
        
        assert error.message == "Access denied"
        assert error.category == ErrorCategory.PERMISSION
        assert error.severity == ErrorSeverity.HIGH
        assert error.file_path == test_path
        assert len(error.suggestions) > 0
        assert "Check file permissions" in error.suggestions

    def test_encoding_error(self) -> None:
        """Test EncodingError creation."""
        test_path = Path("binary.py")
        error = EncodingError("Invalid encoding", test_path, encoding="utf-8")
        
        assert error.message == "Invalid encoding"
        assert error.category == ErrorCategory.ENCODING
        assert error.severity == ErrorSeverity.LOW
        assert error.file_path == test_path
        assert error.context["encoding"] == "utf-8"
        assert len(error.suggestions) > 0
        assert "Try different encoding (current: utf-8)" in error.suggestions

    def test_encoding_error_with_context(self) -> None:
        """Test EncodingError with additional context."""
        test_path = Path("test.py")
        additional_context = {"extra": "info"}
        
        error = EncodingError(
            "Encoding failed",
            test_path,
            encoding="latin-1",
            context=additional_context
        )
        
        assert error.context["encoding"] == "latin-1"
        assert error.context["extra"] == "info"

    def test_parsing_error(self) -> None:
        """Test ParsingError creation."""
        test_path = Path("syntax_error.py")
        error = ParsingError("Syntax error", test_path, line_number=10)
        
        assert error.message == "Syntax error"
        assert error.category == ErrorCategory.PARSING
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.file_path == test_path
        assert error.context["line_number"] == 10
        assert len(error.suggestions) > 0
        assert "Check syntax at line 10" in error.suggestions

    def test_parsing_error_without_line_number(self) -> None:
        """Test ParsingError without line number."""
        test_path = Path("error.py")
        error = ParsingError("Parse failed", test_path)
        
        assert error.context.get("line_number") is None
        assert "Check file syntax" in error.suggestions

    def test_configuration_error(self) -> None:
        """Test ConfigurationError creation."""
        test_context = {"config_file": "settings.toml"}
        error = ConfigurationError("Invalid config", context=test_context)
        
        assert error.message == "Invalid config"
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == test_context
        assert len(error.suggestions) > 0
        assert "Check configuration file syntax" in error.suggestions


class TestErrorCollectorBasic:
    """Test basic ErrorCollector functionality."""

    def test_error_collector_initialization(self) -> None:
        """Test ErrorCollector initialization."""
        collector = ErrorCollector()
        
        assert collector.max_errors == 100
        assert collector.errors == []
        assert collector.error_counts == {}
        assert collector.suppressed_categories == set()

    def test_error_collector_custom_max_errors(self) -> None:
        """Test ErrorCollector with custom max_errors."""
        collector = ErrorCollector(max_errors=50)
        
        assert collector.max_errors == 50

    def test_add_search_error(self) -> None:
        """Test adding SearchError to collector."""
        collector = ErrorCollector()
        error = SearchError("Test error", category=ErrorCategory.VALIDATION)
        
        collector.add_error(error)
        
        assert len(collector.errors) == 1
        assert collector.errors[0].message == "Test error"
        assert collector.errors[0].category == ErrorCategory.VALIDATION
        assert collector.error_counts[ErrorCategory.VALIDATION] == 1

    def test_add_generic_exception(self) -> None:
        """Test adding generic Exception to collector."""
        collector = ErrorCollector()
        exception = ValueError("Invalid value")
        
        collector.add_error(exception, category=ErrorCategory.VALIDATION)
        
        assert len(collector.errors) == 1
        assert collector.errors[0].message == "Invalid value"
        assert collector.errors[0].category == ErrorCategory.VALIDATION
        assert collector.errors[0].exception_type == "ValueError"

    def test_add_error_with_override_parameters(self) -> None:
        """Test adding error with parameter overrides."""
        collector = ErrorCollector()
        error = SearchError("Test", category=ErrorCategory.UNKNOWN)
        test_path = Path("test.py")
        
        collector.add_error(
            error,
            category=ErrorCategory.FILE_ACCESS,  # Override
            severity=ErrorSeverity.CRITICAL,     # Override
            file_path=test_path,
            context={"extra": "info"},
            suggestions=["Try this"]
        )
        
        # Original error category should be preserved for SearchError
        assert collector.errors[0].category == ErrorCategory.UNKNOWN
        # But file_path should be set
        assert collector.errors[0].file_path == test_path

    def test_max_errors_limit(self) -> None:
        """Test that collector respects max_errors limit."""
        collector = ErrorCollector(max_errors=2)
        
        for i in range(5):
            collector.add_error(Exception(f"Error {i}"))
        
        # Should only store 2 errors
        assert len(collector.errors) == 2
        # But should count all errors
        assert collector.error_counts[ErrorCategory.UNKNOWN] == 5

    def test_clear_errors(self) -> None:
        """Test clearing all errors."""
        collector = ErrorCollector()
        collector.add_error(Exception("Test"))
        
        assert len(collector.errors) == 1
        assert len(collector.error_counts) == 1
        
        collector.clear()
        
        assert len(collector.errors) == 0
        assert len(collector.error_counts) == 0


class TestErrorCollectorAdvanced:
    """Test advanced ErrorCollector functionality."""

    def test_suppress_categories(self) -> None:
        """Test suppressing error categories."""
        collector = ErrorCollector()
        collector.suppress_category(ErrorCategory.ENCODING)

        # Add suppressed error
        collector.add_error(Exception("Encoding error"), category=ErrorCategory.ENCODING)
        # Add non-suppressed error
        collector.add_error(Exception("File error"), category=ErrorCategory.FILE_ACCESS)

        assert len(collector.errors) == 1
        assert collector.errors[0].category == ErrorCategory.FILE_ACCESS
        # Counts should still track suppressed errors
        assert collector.error_counts.get(ErrorCategory.ENCODING, 0) == 0
        assert collector.error_counts[ErrorCategory.FILE_ACCESS] == 1

    def test_unsuppress_categories(self) -> None:
        """Test unsuppressing error categories."""
        collector = ErrorCollector()
        collector.suppress_category(ErrorCategory.ENCODING)
        collector.unsuppress_category(ErrorCategory.ENCODING)

        collector.add_error(Exception("Encoding error"), category=ErrorCategory.ENCODING)

        assert len(collector.errors) == 1
        assert collector.errors[0].category == ErrorCategory.ENCODING

    def test_get_errors_by_category(self) -> None:
        """Test filtering errors by category."""
        collector = ErrorCollector()

        collector.add_error(Exception("File error 1"), category=ErrorCategory.FILE_ACCESS)
        collector.add_error(Exception("File error 2"), category=ErrorCategory.FILE_ACCESS)
        collector.add_error(Exception("Parse error"), category=ErrorCategory.PARSING)

        file_errors = collector.get_errors_by_category(ErrorCategory.FILE_ACCESS)
        parse_errors = collector.get_errors_by_category(ErrorCategory.PARSING)

        assert len(file_errors) == 2
        assert len(parse_errors) == 1
        assert all(e.category == ErrorCategory.FILE_ACCESS for e in file_errors)
        assert parse_errors[0].category == ErrorCategory.PARSING

    def test_get_errors_by_severity(self) -> None:
        """Test filtering errors by severity."""
        collector = ErrorCollector()

        collector.add_error(Exception("Low error"), severity=ErrorSeverity.LOW)
        collector.add_error(Exception("High error"), severity=ErrorSeverity.HIGH)
        collector.add_error(Exception("Critical error"), severity=ErrorSeverity.CRITICAL)

        low_errors = collector.get_errors_by_severity(ErrorSeverity.LOW)
        high_errors = collector.get_errors_by_severity(ErrorSeverity.HIGH)
        critical_errors = collector.get_errors_by_severity(ErrorSeverity.CRITICAL)

        assert len(low_errors) == 1
        assert len(high_errors) == 1
        assert len(critical_errors) == 1
        assert low_errors[0].severity == ErrorSeverity.LOW
        assert critical_errors[0].severity == ErrorSeverity.CRITICAL

    def test_get_critical_errors(self) -> None:
        """Test getting critical errors."""
        collector = ErrorCollector()

        collector.add_error(Exception("Normal error"), severity=ErrorSeverity.MEDIUM)
        collector.add_error(Exception("Critical error"), severity=ErrorSeverity.CRITICAL)

        critical_errors = collector.get_critical_errors()

        assert len(critical_errors) == 1
        assert critical_errors[0].severity == ErrorSeverity.CRITICAL
        assert critical_errors[0].message == "Critical error"

    def test_has_critical_errors(self) -> None:
        """Test checking for critical errors."""
        collector = ErrorCollector()

        assert not collector.has_critical_errors()

        collector.add_error(Exception("Normal error"), severity=ErrorSeverity.MEDIUM)
        assert not collector.has_critical_errors()

        collector.add_error(Exception("Critical error"), severity=ErrorSeverity.CRITICAL)
        assert collector.has_critical_errors()

    def test_get_summary(self) -> None:
        """Test getting error summary statistics."""
        collector = ErrorCollector()

        collector.add_error(Exception("File error"), category=ErrorCategory.FILE_ACCESS, severity=ErrorSeverity.HIGH)
        collector.add_error(Exception("Parse error"), category=ErrorCategory.PARSING, severity=ErrorSeverity.MEDIUM)
        collector.add_error(Exception("Critical error"), category=ErrorCategory.UNKNOWN, severity=ErrorSeverity.CRITICAL)

        summary = collector.get_summary()

        assert summary["total_errors"] == 3
        assert summary["by_category"][ErrorCategory.FILE_ACCESS] == 1
        assert summary["by_category"][ErrorCategory.PARSING] == 1
        assert summary["by_category"][ErrorCategory.UNKNOWN] == 1
        assert summary["by_severity"]["high"] == 1
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["critical"] == 1
        assert summary["has_critical"] is True

    def test_classify_exception_builtin_errors(self) -> None:
        """Test automatic classification of built-in exceptions."""
        collector = ErrorCollector()

        # Test various exception types
        import builtins
        test_cases = [
            (FileNotFoundError("File not found"), ErrorCategory.FILE_ACCESS),
            (builtins.PermissionError("Permission denied"), ErrorCategory.PERMISSION),
            (UnicodeDecodeError("utf-8", b"", 0, 1, "invalid"), ErrorCategory.ENCODING),
            (SyntaxError("Invalid syntax"), ErrorCategory.PARSING),
            (MemoryError("Out of memory"), ErrorCategory.MEMORY),
            (TimeoutError("Operation timed out"), ErrorCategory.TIMEOUT),
            (ValueError("Invalid value"), ErrorCategory.VALIDATION),
            (RuntimeError("Runtime error"), ErrorCategory.UNKNOWN),
        ]

        for exception, expected_category in test_cases:
            collector.clear()
            collector.add_error(exception)

            assert len(collector.errors) == 1
            assert collector.errors[0].category == expected_category

    def test_traceback_capture(self) -> None:
        """Test that traceback is captured when available."""
        collector = ErrorCollector()

        try:
            raise ValueError("Test error")
        except ValueError as e:
            collector.add_error(e)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.traceback_str is not None
        assert "ValueError: Test error" in error_info.traceback_str

    def test_no_traceback_when_not_in_exception_context(self) -> None:
        """Test that no traceback is captured when not in exception context."""
        collector = ErrorCollector()
        exception = ValueError("Test error")

        collector.add_error(exception)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.traceback_str is None


class TestHandleFileError:
    """Test handle_file_error function."""

    def test_handle_file_not_found_error(self) -> None:
        """Test handling FileNotFoundError."""
        collector = ErrorCollector()
        test_path = Path("missing.py")
        exception = FileNotFoundError("File not found")

        handle_file_error(test_path, "read", exception, collector)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.category == ErrorCategory.FILE_ACCESS
        assert "Cannot read file" in error_info.message
        assert error_info.file_path == test_path

    def test_handle_permission_error(self) -> None:
        """Test handling PermissionError."""
        collector = ErrorCollector()
        test_path = Path("protected.py")

        # Use built-in PermissionError to avoid conflict
        import builtins
        exception = builtins.PermissionError("Permission denied")

        handle_file_error(test_path, "write", exception, collector)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.category == ErrorCategory.PERMISSION
        assert "Permission denied during write" in error_info.message

    def test_handle_unicode_decode_error(self) -> None:
        """Test handling UnicodeDecodeError."""
        collector = ErrorCollector()
        test_path = Path("binary.py")
        exception = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        handle_file_error(test_path, "parse", exception, collector)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.category == ErrorCategory.ENCODING
        assert "Encoding error during parse" in error_info.message

    def test_handle_syntax_error(self) -> None:
        """Test handling SyntaxError."""
        collector = ErrorCollector()
        test_path = Path("syntax_error.py")
        exception = SyntaxError("invalid syntax")
        exception.lineno = 42

        handle_file_error(test_path, "compile", exception, collector)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert error_info.category == ErrorCategory.PARSING
        assert "Parsing error during compile" in error_info.message

    def test_handle_generic_exception(self) -> None:
        """Test handling generic exception."""
        collector = ErrorCollector()
        test_path = Path("test.py")
        exception = RuntimeError("Unexpected error")

        handle_file_error(test_path, "process", exception, collector)

        assert len(collector.errors) == 1
        error_info = collector.errors[0]
        assert "Unexpected error during process" in error_info.message

    def test_handle_file_error_with_logger(self) -> None:
        """Test handle_file_error with logger."""
        collector = ErrorCollector()
        mock_logger = Mock()
        test_path = Path("test.py")
        exception = FileNotFoundError("File not found")

        handle_file_error(test_path, "read", exception, collector, mock_logger)

        # Should log the error
        mock_logger.log_file_error.assert_called_once()
        args = mock_logger.log_file_error.call_args[0]
        assert str(test_path) in args[0]
        assert "operation" in mock_logger.log_file_error.call_args[1]

    def test_handle_file_error_without_collector(self) -> None:
        """Test handle_file_error without error collector."""
        test_path = Path("test.py")
        exception = FileNotFoundError("File not found")

        # Should not raise exception
        handle_file_error(test_path, "read", exception)


class TestCreateErrorReport:
    """Test create_error_report function."""

    def test_create_error_report_empty(self) -> None:
        """Test creating error report with no errors."""
        collector = ErrorCollector()

        report = create_error_report(collector)

        assert "Error Report" in report
        assert "Total errors: 0" in report
        assert "No errors to report" in report

    def test_create_error_report_with_errors(self) -> None:
        """Test creating error report with errors."""
        collector = ErrorCollector()

        collector.add_error(Exception("File error"), category=ErrorCategory.FILE_ACCESS)
        collector.add_error(Exception("Parse error"), category=ErrorCategory.PARSING)
        collector.add_error(Exception("Critical error"), severity=ErrorSeverity.CRITICAL)

        report = create_error_report(collector)

        assert "Error Report" in report
        assert "Total errors: 3" in report
        assert "file_access: 1" in report
        assert "parsing: 1" in report
        assert "Critical Errors:" in report
        assert "Critical error" in report
        assert "General Suggestions:" in report

    def test_create_error_report_with_critical_errors(self) -> None:
        """Test error report includes critical error details."""
        collector = ErrorCollector()
        test_path = Path("critical.py")

        critical_error = SearchError(
            "Critical failure",
            severity=ErrorSeverity.CRITICAL,
            file_path=test_path,
            suggestions=["Fix this immediately", "Check logs"]
        )
        collector.add_error(critical_error)

        report = create_error_report(collector)

        assert "Critical Errors:" in report
        assert "Critical failure" in report
        assert f"File: {test_path}" in report
        assert "Fix this immediately, Check logs" in report
