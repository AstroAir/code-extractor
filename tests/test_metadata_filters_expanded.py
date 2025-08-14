"""
Expanded comprehensive tests for metadata filters module.

This module tests metadata filtering functionality that is currently not covered,
including edge cases, error conditions, validation logic, and advanced filter combinations.
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from pysearch.utils.metadata_filters import (
    apply_metadata_filters,
    create_date_filter,
    create_metadata_filters,
    create_size_filter,
    get_file_author,
)
from pysearch import FileMetadata, Language, MetadataFilters


class TestGetFileAuthor:
    """Test get_file_author function."""

    def test_get_file_author_git_success(self, tmp_path: Path) -> None:
        """Test getting file author via git when successful."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        
        # Mock successful git command
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "John Doe\n"
        
        with patch('subprocess.run', return_value=mock_result) as mock_run:
            author = get_file_author(test_file)
            
            assert author == "John Doe"
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            assert args[0] == "git"
            assert args[1] == "log"
            assert str(test_file) in args

    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_git_failure(self, tmp_path: Path) -> None:
        """Test getting file author when git fails."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # Mock failed git command
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch('subprocess.run', return_value=mock_result), \
             patch('pwd.getpwuid') as mock_pwd:
            
            # Mock pwd fallback
            mock_user = Mock()
            mock_user.pw_name = "testuser"
            mock_pwd.return_value = mock_user
            
            author = get_file_author(test_file)
            
            assert author == "testuser"

    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_git_timeout(self, tmp_path: Path) -> None:
        """Test getting file author when git times out."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("git", 5)), \
             patch('pwd.getpwuid') as mock_pwd:
            
            # Mock pwd fallback
            mock_user = Mock()
            mock_user.pw_name = "fallbackuser"
            mock_pwd.return_value = mock_user
            
            author = get_file_author(test_file)
            
            assert author == "fallbackuser"

    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_no_git_no_pwd(self, tmp_path: Path) -> None:
        """Test getting file author when both git and pwd fail."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        with patch('subprocess.run', side_effect=FileNotFoundError("git not found")), \
             patch('pwd.getpwuid', side_effect=ImportError("pwd not available")):
            
            author = get_file_author(test_file)
            
            assert author is None

    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_pwd_key_error(self, tmp_path: Path) -> None:
        """Test getting file author when pwd raises KeyError."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        with patch('subprocess.run', side_effect=FileNotFoundError("git not found")), \
             patch('pwd.getpwuid', side_effect=KeyError("user not found")):
            
            author = get_file_author(test_file)
            
            assert author is None

    @pytest.mark.skipif(sys.platform == "win32", reason="Path mocking not supported on Windows")
    def test_get_file_author_os_error(self, tmp_path: Path) -> None:
        """Test getting file author when file stat fails."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # Mock git to fail and file operations to fail
        with patch('subprocess.run', side_effect=FileNotFoundError("git not found")):
            # Mock exists to return True but stat to fail
            with patch.object(test_file.parent, 'exists', return_value=True), \
                 patch.object(test_file, 'stat', side_effect=OSError("stat failed")):
                author = get_file_author(test_file)
                assert author is None

    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_empty_git_output(self, tmp_path: Path) -> None:
        """Test getting file author when git returns empty output."""
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # Mock git command with empty output
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "   \n"  # Whitespace only

        with patch('subprocess.run', return_value=mock_result), \
             patch('pwd.getpwuid') as mock_pwd:
            
            # Mock pwd fallback
            mock_user = Mock()
            mock_user.pw_name = "fallbackuser"
            mock_pwd.return_value = mock_user
            
            author = get_file_author(test_file)
            
            assert author == "fallbackuser"


class TestCreateSizeFilter:
    """Test create_size_filter function."""

    def test_create_size_filter_bytes(self) -> None:
        """Test creating size filter with byte values."""
        min_bytes, max_bytes = create_size_filter("100", "1000")
        
        assert min_bytes == 100
        assert max_bytes == 1000

    def test_create_size_filter_kilobytes(self) -> None:
        """Test creating size filter with kilobyte values."""
        min_bytes, max_bytes = create_size_filter("1KB", "5K")
        
        assert min_bytes == 1024
        assert max_bytes == 5 * 1024

    def test_create_size_filter_megabytes(self) -> None:
        """Test creating size filter with megabyte values."""
        min_bytes, max_bytes = create_size_filter("1MB", "10M")
        
        assert min_bytes == 1024 * 1024
        assert max_bytes == 10 * 1024 * 1024

    def test_create_size_filter_gigabytes(self) -> None:
        """Test creating size filter with gigabyte values."""
        min_bytes, max_bytes = create_size_filter("1GB", "2G")
        
        assert min_bytes == 1024 * 1024 * 1024
        assert max_bytes == 2 * 1024 * 1024 * 1024

    def test_create_size_filter_terabytes(self) -> None:
        """Test creating size filter with terabyte values."""
        min_bytes, max_bytes = create_size_filter("1TB", "2T")
        
        assert min_bytes == 1024 * 1024 * 1024 * 1024
        assert max_bytes == 2 * 1024 * 1024 * 1024 * 1024

    def test_create_size_filter_decimal_values(self) -> None:
        """Test creating size filter with decimal values."""
        min_bytes, max_bytes = create_size_filter("1.5KB", "2.5MB")
        
        assert min_bytes == int(1.5 * 1024)
        assert max_bytes == int(2.5 * 1024 * 1024)

    def test_create_size_filter_none_values(self) -> None:
        """Test creating size filter with None values."""
        min_bytes, max_bytes = create_size_filter(None, None)
        
        assert min_bytes is None
        assert max_bytes is None

    def test_create_size_filter_mixed_values(self) -> None:
        """Test creating size filter with mixed values."""
        min_bytes, max_bytes = create_size_filter("1KB", None)
        
        assert min_bytes == 1024
        assert max_bytes is None

    def test_create_size_filter_invalid_format(self) -> None:
        """Test creating size filter with invalid format."""
        with pytest.raises(ValueError, match="Invalid size format"):
            create_size_filter("invalid", None)

    def test_create_size_filter_unknown_unit(self) -> None:
        """Test creating size filter with unknown unit."""
        with pytest.raises(ValueError, match="Invalid size format"):
            create_size_filter("1XB", None)

    def test_create_size_filter_whitespace(self) -> None:
        """Test creating size filter with whitespace."""
        min_bytes, max_bytes = create_size_filter("  1 KB  ", "  5 MB  ")
        
        assert min_bytes == 1024
        assert max_bytes == 5 * 1024 * 1024

    def test_create_size_filter_case_insensitive(self) -> None:
        """Test creating size filter is case insensitive."""
        min_bytes, max_bytes = create_size_filter("1kb", "5mb")
        
        assert min_bytes == 1024
        assert max_bytes == 5 * 1024 * 1024


class TestCreateDateFilter:
    """Test create_date_filter function."""

    def test_create_date_filter_unix_timestamp(self) -> None:
        """Test creating date filter with Unix timestamps."""
        timestamp = 1701234567.0
        mod_after, mod_before, create_after, create_before = create_date_filter(
            modified_after=str(timestamp),
            modified_before=str(timestamp + 3600),
            created_after=str(timestamp - 3600),
            created_before=str(timestamp + 7200)
        )
        
        assert mod_after == timestamp
        assert mod_before == timestamp + 3600
        assert create_after == timestamp - 3600
        assert create_before == timestamp + 7200

    def test_create_date_filter_iso_format(self) -> None:
        """Test creating date filter with ISO format dates."""
        mod_after, mod_before, create_after, create_before = create_date_filter(
            modified_after="2023-12-01",
            modified_before="2023-12-01T10:30:00",
            created_after="2023-11-01 09:15:30",
            created_before="2023-12-31"
        )
        
        assert mod_after is not None
        assert mod_before is not None
        assert create_after is not None
        assert create_before is not None
        
        # Verify the dates are parsed correctly
        assert mod_after == datetime(2023, 12, 1).timestamp()
        assert mod_before == datetime(2023, 12, 1, 10, 30, 0).timestamp()

    def test_create_date_filter_relative_days(self) -> None:
        """Test creating date filter with relative days."""
        before_time = time.time()
        
        mod_after, mod_before, create_after, create_before = create_date_filter(
            modified_after="1d",
            modified_before="7d",
            created_after="30d",
            created_before="365d"
        )
        
        after_time = time.time()
        
        # Should be approximately 1 day ago
        expected_1d = before_time - (24 * 60 * 60)
        assert abs(mod_after - expected_1d) < 10  # Within 10 seconds
        
        # Should be approximately 7 days ago
        expected_7d = before_time - (7 * 24 * 60 * 60)
        assert abs(mod_before - expected_7d) < 10

    def test_create_date_filter_relative_weeks(self) -> None:
        """Test creating date filter with relative weeks."""
        before_time = time.time()
        
        mod_after, _, _, _ = create_date_filter(modified_after="2w")
        
        expected = before_time - (2 * 7 * 24 * 60 * 60)
        assert abs(mod_after - expected) < 10

    def test_create_date_filter_relative_months(self) -> None:
        """Test creating date filter with relative months."""
        before_time = time.time()
        
        mod_after, _, _, _ = create_date_filter(modified_after="3m")
        
        expected = before_time - (3 * 30 * 24 * 60 * 60)  # Approximate
        assert abs(mod_after - expected) < 3600  # Within 1 hour

    def test_create_date_filter_relative_years(self) -> None:
        """Test creating date filter with relative years."""
        before_time = time.time()
        
        mod_after, _, _, _ = create_date_filter(modified_after="1y")
        
        expected = before_time - (365 * 24 * 60 * 60)  # Approximate
        assert abs(mod_after - expected) < 3600  # Within 1 hour

    def test_create_date_filter_none_values(self) -> None:
        """Test creating date filter with None values."""
        mod_after, mod_before, create_after, create_before = create_date_filter()
        
        assert mod_after is None
        assert mod_before is None
        assert create_after is None
        assert create_before is None

    def test_create_date_filter_invalid_format(self) -> None:
        """Test creating date filter with invalid format."""
        with pytest.raises(ValueError, match="Invalid date format"):
            create_date_filter(modified_after="invalid-date")

    def test_create_date_filter_invalid_relative_unit(self) -> None:
        """Test creating date filter with invalid relative unit."""
        with pytest.raises(ValueError, match="Invalid date format"):
            create_date_filter(modified_after="1x")  # Invalid unit

    def test_create_date_filter_empty_string(self) -> None:
        """Test creating date filter with empty string."""
        mod_after, _, _, _ = create_date_filter(modified_after="")

        assert mod_after is None


class TestApplyMetadataFiltersEdgeCases:
    """Test apply_metadata_filters function edge cases."""

    def create_test_metadata(self, **kwargs: Any) -> FileMetadata:
        """Create test metadata with default values."""
        # Set defaults
        path = kwargs.get("path", Path("test.py"))
        size = kwargs.get("size", 1000)
        mtime = kwargs.get("mtime", time.time())
        language = kwargs.get("language", Language.PYTHON)
        encoding = kwargs.get("encoding", "utf-8")
        line_count = kwargs.get("line_count", 50)
        author = kwargs.get("author", "testuser")
        created_date = kwargs.get("created_date", time.time())
        modified_date = kwargs.get("modified_date", time.time())

        return FileMetadata(
            path=path,
            size=size,
            mtime=mtime,
            language=language,
            encoding=encoding,
            line_count=line_count,
            author=author,
            created_date=created_date,
            modified_date=modified_date
        )

    def test_apply_metadata_filters_none_metadata_fields(self) -> None:
        """Test applying filters when metadata fields are None."""
        metadata = self.create_test_metadata(
            created_date=None,
            line_count=None,
            author=None
        )

        # Filters that require these fields should fail
        filters = MetadataFilters(
            created_after=time.time() - 3600,
            min_lines=10,
            author_pattern="test.*"
        )

        result = apply_metadata_filters(metadata, filters)
        assert result is False

    def test_apply_metadata_filters_created_date_none_with_filters(self) -> None:
        """Test created date filters when created_date is None."""
        metadata = self.create_test_metadata(created_date=None)

        # created_after filter should fail
        filters = MetadataFilters(created_after=time.time() - 3600)
        assert apply_metadata_filters(metadata, filters) is False

        # created_before filter should fail
        filters = MetadataFilters(created_before=time.time() + 3600)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_line_count_none_with_filters(self) -> None:
        """Test line count filters when line_count is None."""
        metadata = self.create_test_metadata(line_count=None)

        # min_lines filter should fail
        filters = MetadataFilters(min_lines=10)
        assert apply_metadata_filters(metadata, filters) is False

        # max_lines filter should fail
        filters = MetadataFilters(max_lines=100)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_author_none_with_pattern(self) -> None:
        """Test author pattern filter when author is None."""
        metadata = self.create_test_metadata(author=None)

        filters = MetadataFilters(author_pattern="test.*")
        result = apply_metadata_filters(metadata, filters)
        assert result is False

    def test_apply_metadata_filters_regex_case_insensitive(self) -> None:
        """Test that regex patterns are case insensitive."""
        metadata = self.create_test_metadata(
            author="John Doe",
            encoding="UTF-8"
        )

        # Author pattern should be case insensitive
        filters = MetadataFilters(author_pattern="john.*")
        assert apply_metadata_filters(metadata, filters) is True

        # Encoding pattern should be case insensitive
        filters = MetadataFilters(encoding_pattern="utf.*")
        assert apply_metadata_filters(metadata, filters) is True

    def test_apply_metadata_filters_regex_invalid_pattern(self) -> None:
        """Test applying filters with invalid regex pattern."""
        metadata = self.create_test_metadata(author="testuser")

        # Invalid regex pattern should raise exception
        filters = MetadataFilters(author_pattern="[invalid")

        with pytest.raises(re.error):
            apply_metadata_filters(metadata, filters)

    def test_apply_metadata_filters_boundary_values(self) -> None:
        """Test applying filters with boundary values."""
        metadata = self.create_test_metadata(
            size=1000,
            line_count=50,
            mtime=1000.0
        )

        # Exact boundary values should pass
        filters = MetadataFilters(
            min_size=1000,
            max_size=1000,
            min_lines=50,
            max_lines=50,
            modified_after=1000.0,
            modified_before=1000.0
        )

        assert apply_metadata_filters(metadata, filters) is True

    def test_apply_metadata_filters_empty_language_set(self) -> None:
        """Test applying filters with empty language set."""
        metadata = self.create_test_metadata(language=Language.PYTHON)

        # Empty language set should not match anything
        filters = MetadataFilters(languages=set())
        result = apply_metadata_filters(metadata, filters)
        assert result is False

    def test_apply_metadata_filters_multiple_languages(self) -> None:
        """Test applying filters with multiple languages."""
        metadata = self.create_test_metadata(language=Language.PYTHON)

        # Should match when language is in set
        filters = MetadataFilters(languages={Language.PYTHON, Language.JAVASCRIPT})
        assert apply_metadata_filters(metadata, filters) is True

        # Should not match when language is not in set
        filters = MetadataFilters(languages={Language.JAVA, Language.GO})
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_all_none_filters(self) -> None:
        """Test applying filters when all filter values are None."""
        metadata = self.create_test_metadata()

        filters = MetadataFilters()  # All fields are None by default
        result = apply_metadata_filters(metadata, filters)
        assert result is True  # Should pass when no filters are applied
