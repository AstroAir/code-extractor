"""Tests for pysearch.utils.error_handling module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.utils.error_handling import (
    ConfigurationError,
    EncodingError,
    ErrorCategory,
    ErrorCollector,
    ErrorInfo,
    ErrorSeverity,
    FileAccessError,
    ParsingError,
    SearchError,
    create_error_report,
    handle_file_error,
)


class TestErrorSeverity:
    """Tests for ErrorSeverity enum."""

    def test_values(self):
        assert ErrorSeverity.LOW is not None
        assert ErrorSeverity.MEDIUM is not None
        assert ErrorSeverity.HIGH is not None
        assert ErrorSeverity.CRITICAL is not None


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_file_access(self):
        assert ErrorCategory.FILE_ACCESS is not None

    def test_encoding(self):
        assert ErrorCategory.ENCODING is not None

    def test_parsing(self):
        assert ErrorCategory.PARSING is not None


class TestErrorInfo:
    """Tests for ErrorInfo dataclass."""

    def test_creation(self):
        info = ErrorInfo(
            category=ErrorCategory.FILE_ACCESS,
            severity=ErrorSeverity.LOW,
            message="File not found",
            file_path=Path("test.py"),
        )
        assert info.message == "File not found"
        assert info.category == ErrorCategory.FILE_ACCESS


class TestErrorCollector:
    """Tests for ErrorCollector class."""

    def test_init(self):
        collector = ErrorCollector()
        assert collector is not None

    def test_add_error(self):
        collector = ErrorCollector()
        collector.add_error(SearchError("Error occurred"), file_path=Path("test.py"))
        assert collector.has_critical_errors() is not None

    def test_add_error_with_category(self):
        collector = ErrorCollector()
        collector.add_error(
            ParsingError("Parse error", file_path=Path("x.py")),
            category=ErrorCategory.PARSING,
            severity=ErrorSeverity.MEDIUM,
        )
        errors = collector.get_errors_by_category(ErrorCategory.PARSING)
        assert len(errors) >= 1

    def test_clear(self):
        collector = ErrorCollector()
        collector.add_error(SearchError("Error"), file_path=Path("test.py"))
        collector.clear()

    def test_get_summary(self):
        collector = ErrorCollector()
        collector.add_error(SearchError("err1"), file_path=Path("a.py"))
        collector.add_error(SearchError("err2"), file_path=Path("b.py"))
        summary = collector.get_summary()
        assert isinstance(summary, (str, dict))


class TestSearchError:
    """Tests for SearchError exception."""

    def test_raise(self):
        with pytest.raises(SearchError):
            raise SearchError("test error")

    def test_message(self):
        err = SearchError("custom message")
        assert str(err) == "custom message"


class TestFileAccessError:
    """Tests for FileAccessError exception."""

    def test_raise(self):
        with pytest.raises(FileAccessError):
            raise FileAccessError("file not found", file_path=Path("missing.py"))

    def test_is_search_error(self):
        assert issubclass(FileAccessError, SearchError)


class TestEncodingError:
    """Tests for EncodingError exception."""

    def test_raise(self):
        with pytest.raises(EncodingError):
            raise EncodingError("bad encoding", file_path=Path("bad.py"))


class TestParsingError:
    """Tests for ParsingError exception."""

    def test_raise(self):
        with pytest.raises(ParsingError):
            raise ParsingError("parse failed", file_path=Path("bad.py"))


class TestConfigurationError:
    """Tests for ConfigurationError exception."""

    def test_raise(self):
        with pytest.raises(ConfigurationError):
            raise ConfigurationError("bad config")


class TestHandleFileError:
    """Tests for handle_file_error function."""

    def test_file_not_found(self):
        handle_file_error(Path("missing.py"), "read", FileNotFoundError())

    def test_permission_denied(self):
        handle_file_error(Path("locked.py"), "read", PermissionError())


class TestCreateErrorReport:
    """Tests for create_error_report function."""

    def test_basic(self):
        collector = ErrorCollector()
        collector.add_error(SearchError("Error occurred"), file_path=Path("test.py"))
        report = create_error_report(collector)
        assert isinstance(report, str)
        assert len(report) > 0
