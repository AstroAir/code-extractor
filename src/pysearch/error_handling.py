"""
Comprehensive error handling and reporting system for pysearch.

This module provides a robust error handling framework that categorizes, tracks,
and reports errors encountered during search operations. It enables graceful
degradation and provides detailed diagnostics for troubleshooting.

Key Features:
    - Hierarchical error classification system
    - Detailed error information collection
    - Error severity assessment
    - Batch error collection and reporting
    - Context-aware error handling
    - Performance impact tracking
    - Recovery strategy suggestions

Error Categories:
    - FILE_ACCESS: File permission and access issues
    - PERMISSION: Security and permission errors
    - ENCODING: Text encoding and character set issues
    - PARSING: Code parsing and syntax errors
    - MEMORY: Memory allocation and resource issues
    - TIMEOUT: Operation timeout errors
    - CONFIGURATION: Configuration and setup errors
    - NETWORK: Network connectivity issues
    - VALIDATION: Input validation errors

Classes:
    ErrorSeverity: Error severity levels (LOW, MEDIUM, HIGH, CRITICAL)
    ErrorCategory: Error classification categories
    ErrorInfo: Detailed error information container
    ErrorCollector: Batch error collection and analysis
    PySearchError: Base exception class for pysearch errors

Functions:
    handle_file_error: Specialized file error handling
    create_error_report: Generate comprehensive error reports
    classify_error: Automatic error classification

Example:
    Basic error handling:
        >>> from pysearch.error_handling import ErrorCollector, handle_file_error
        >>> from pathlib import Path
        >>>
        >>> collector = ErrorCollector()
        >>> try:
        ...     # Perform file operation
        ...     content = Path("file.py").read_text()
        ... except Exception as e:
        ...     error_info = handle_file_error(e, Path("file.py"))
        ...     collector.add_error_info(error_info)
        >>>
        >>> # Generate report
        >>> report = collector.generate_report()
        >>> print(f"Collected {len(report.errors)} errors")

    Advanced error analysis:
        >>> # Analyze error patterns
        >>> by_category = report.group_by_category()
        >>> for category, errors in by_category.items():
        ...     print(f"{category}: {len(errors)} errors")
        >>>
        >>> # Get recovery suggestions
        >>> suggestions = report.get_recovery_suggestions()
        >>> for suggestion in suggestions:
        ...     print(f"- {suggestion}")
"""

from __future__ import annotations

# Import built-in exceptions before defining custom ones
import builtins
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

BuiltinPermissionError = builtins.PermissionError


class ErrorSeverity(str, Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories for classification."""

    FILE_ACCESS = "file_access"
    PERMISSION = "permission"
    ENCODING = "encoding"
    PARSING = "parsing"
    MEMORY = "memory"
    TIMEOUT = "timeout"
    CONFIGURATION = "configuration"
    NETWORK = "network"
    VALIDATION = "validation"
    UNKNOWN = "unknown"


@dataclass
class ErrorInfo:
    """Detailed error information."""

    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    file_path: Path | None = None
    line_number: int | None = None
    exception_type: str | None = None
    traceback_str: str | None = None
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


class SearchError(Exception):
    """Base exception for search-related errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        file_path: Path | None = None,
        suggestions: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message: str = message
        self.category: ErrorCategory = category
        self.severity: ErrorSeverity = severity
        self.file_path: Path | None = file_path
        self.suggestions: list[str] = suggestions or []
        self.context: dict[str, Any] = context or {}
        self.timestamp: float = time.time()


class FileAccessError(SearchError):
    """Error accessing files."""

    def __init__(
        self, message: str, file_path: Path, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.MEDIUM,
            file_path=file_path,
            context=context,
        )


class PermissionError(SearchError):
    """Permission-related errors."""

    def __init__(
        self, message: str, file_path: Path, context: dict[str, Any] | None = None
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.PERMISSION,
            severity=ErrorSeverity.HIGH,
            file_path=file_path,
            suggestions=[
                "Check file permissions",
                "Run with appropriate user privileges",
                "Verify file ownership",
            ],
            context=context,
        )


class EncodingError(SearchError):
    """File encoding-related errors."""

    def __init__(
        self,
        message: str,
        file_path: Path,
        encoding: str = "unknown",
        context: dict[str, Any] | None = None,
    ) -> None:
        merged_context: dict[str, Any] = {}
        if context:
            merged_context.update(context)
        merged_context["encoding"] = encoding

        super().__init__(
            message,
            category=ErrorCategory.ENCODING,
            severity=ErrorSeverity.LOW,
            file_path=file_path,
            suggestions=[
                f"Try different encoding (current: {encoding})",
                "Check if file is binary",
                "Use encoding detection tools",
            ],
            context=merged_context,
        )


class ParsingError(SearchError):
    """Code parsing-related errors."""

    def __init__(
        self,
        message: str,
        file_path: Path,
        line_number: int | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.LOW,
            file_path=file_path,
            suggestions=[
                "Check file syntax",
                "Verify file is valid source code",
                "Skip AST parsing for this file",
            ],
            context=context,
        )
        self.line_number: int | None = line_number


class ConfigurationError(SearchError):
    """Configuration-related errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            severity=ErrorSeverity.HIGH,
            suggestions=[
                "Check configuration file syntax",
                "Verify all required settings",
                "Use default configuration",
            ],
            context=context,
        )


class ErrorCollector:
    """Collects and manages errors during search operations."""

    def __init__(self, max_errors: int = 100) -> None:
        self.max_errors = max_errors
        self.errors: list[ErrorInfo] = []
        self.error_counts: dict[ErrorCategory, int] = {}
        self.suppressed_categories: set[ErrorCategory] = set()

    def add_error(
        self,
        exception: Exception | SearchError,
        category: ErrorCategory | None = None,
        severity: ErrorSeverity | None = None,
        file_path: Path | None = None,
        context: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        """Add an error to the collection."""
        # Determine error details
        if isinstance(exception, SearchError):
            error_category = exception.category
            error_severity = exception.severity
            error_file_path = exception.file_path or file_path
            error_suggestions = exception.suggestions or suggestions or []
            error_context = {**exception.context, **(context or {})}
        else:
            error_category = category or self._classify_exception(exception)
            error_severity = severity or ErrorSeverity.MEDIUM
            error_file_path = file_path
            error_suggestions = suggestions or []
            error_context = context or {}

        # Skip if category is suppressed
        if error_category in self.suppressed_categories:
            return

        # Create error info
        error_info = ErrorInfo(
            category=error_category,
            severity=error_severity,
            message=str(exception),
            file_path=error_file_path,
            exception_type=type(exception).__name__,
            traceback_str=traceback.format_exc() if sys.exc_info()[0] else None,
            context=error_context,
            suggestions=error_suggestions,
        )

        # Add to collection (with limit)
        if len(self.errors) < self.max_errors:
            self.errors.append(error_info)

        # Update counts
        self.error_counts[error_category] = self.error_counts.get(error_category, 0) + 1

    def _classify_exception(self, exception: Exception) -> ErrorCategory:
        """Classify exception into error category."""
        exception_type = type(exception).__name__

        if exception_type in ["FileNotFoundError", "IsADirectoryError", "NotADirectoryError"]:
            return ErrorCategory.FILE_ACCESS
        elif exception_type in ["PermissionError", "OSError"]:
            return ErrorCategory.PERMISSION
        elif exception_type in ["UnicodeDecodeError", "UnicodeError", "LookupError"]:
            return ErrorCategory.ENCODING
        elif exception_type in ["SyntaxError", "IndentationError", "TabError"]:
            return ErrorCategory.PARSING
        elif exception_type in ["MemoryError"]:
            return ErrorCategory.MEMORY
        elif exception_type in ["TimeoutError"]:
            return ErrorCategory.TIMEOUT
        else:
            return ErrorCategory.UNKNOWN

    def suppress_category(self, category: ErrorCategory) -> None:
        """Suppress errors of a specific category."""
        self.suppressed_categories.add(category)

    def unsuppress_category(self, category: ErrorCategory) -> None:
        """Stop suppressing errors of a specific category."""
        self.suppressed_categories.discard(category)

    def get_errors_by_category(self, category: ErrorCategory) -> list[ErrorInfo]:
        """Get all errors of a specific category."""
        return [error for error in self.errors if error.category == category]

    def get_errors_by_severity(self, severity: ErrorSeverity) -> list[ErrorInfo]:
        """Get all errors of a specific severity."""
        return [error for error in self.errors if error.severity == severity]

    def get_critical_errors(self) -> list[ErrorInfo]:
        """Get all critical errors."""
        return self.get_errors_by_severity(ErrorSeverity.CRITICAL)

    def has_critical_errors(self) -> bool:
        """Check if there are any critical errors."""
        return len(self.get_critical_errors()) > 0

    def get_summary(self) -> dict[str, Any]:
        """Get error summary statistics."""
        return {
            "total_errors": len(self.errors),
            "by_category": dict(self.error_counts),
            "by_severity": {
                severity.value: len(self.get_errors_by_severity(severity))
                for severity in ErrorSeverity
            },
            "suppressed_categories": list(self.suppressed_categories),
            "has_critical": self.has_critical_errors(),
        }

    def clear(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
        self.error_counts.clear()


def handle_file_error(
    file_path: Path,
    operation: str,
    exception: Exception,
    error_collector: ErrorCollector | None = None,
    logger: Any | None = None,
) -> None:
    """
    Handle file-related errors with appropriate classification and logging.

    Args:
        file_path: Path to the file that caused the error
        operation: Operation being performed (e.g., "read", "parse", "index")
        exception: The exception that occurred
        error_collector: Optional error collector to add the error to
        logger: Optional logger to log the error
    """
    # Classify the error
    error: SearchError
    if isinstance(exception, (FileNotFoundError, IsADirectoryError)):
        error = FileAccessError(f"Cannot {operation} file: {exception}", file_path)
    elif isinstance(exception, BuiltinPermissionError):
        error = PermissionError(f"Permission denied during {operation}: {exception}", file_path)
    elif isinstance(exception, (UnicodeDecodeError, UnicodeError)):
        error = EncodingError(f"Encoding error during {operation}: {exception}", file_path)
    elif isinstance(exception, (SyntaxError, IndentationError)):
        line_num = getattr(exception, "lineno", None)
        error = ParsingError(f"Parsing error during {operation}: {exception}", file_path, line_num)
    else:
        error = SearchError(
            f"Unexpected error during {operation}: {exception}", file_path=file_path
        )

    # Add to error collector
    if error_collector:
        error_collector.add_error(error)

    # Log the error
    if logger:
        logger.log_file_error(str(file_path), str(error), operation=operation)


def create_error_report(error_collector: ErrorCollector) -> str:
    """Create a human-readable error report."""
    if not error_collector.errors:
        return "No errors occurred during the search operation."

    summary = error_collector.get_summary()

    report = ["Search Error Report", "=" * 50, ""]

    # Summary
    report.append(f"Total errors: {summary['total_errors']}")
    report.append(f"Critical errors: {summary['by_severity']['critical']}")
    report.append("")

    # By category
    report.append("Errors by category:")
    for category, count in summary["by_category"].items():
        report.append(f"  {category}: {count}")
    report.append("")

    # Critical errors details
    critical_errors = error_collector.get_critical_errors()
    if critical_errors:
        report.append("Critical Errors:")
        for error in critical_errors:
            report.append(f"  - {error.message}")
            if error.file_path:
                report.append(f"    File: {error.file_path}")
            if error.suggestions:
                report.append(f"    Suggestions: {', '.join(error.suggestions)}")
        report.append("")

    # Suggestions
    report.append("General Suggestions:")
    report.append("  - Check file permissions and accessibility")
    report.append("  - Verify file encodings are correct")
    report.append("  - Consider excluding problematic directories")
    report.append("  - Use --debug flag for more detailed error information")

    return "\n".join(report)
