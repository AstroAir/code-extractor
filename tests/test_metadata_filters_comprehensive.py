"""
Comprehensive tests for metadata_filters module.

This module tests the metadata filtering functionality including
file size, date, author, encoding, and language filters.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysearch.utils.metadata_filters import (
    apply_metadata_filters,
    create_metadata_filters,
    get_file_author,
)
from pysearch import FileMetadata, Language, MetadataFilters


class TestMetadataFilters:
    """Test metadata filtering functionality."""

    def create_sample_metadata(self, **overrides) -> FileMetadata:
        """Create sample file metadata for testing."""
        # Create a basic metadata object
        metadata = FileMetadata(
            path=Path("test.py"),
            size=1024,
            mtime=time.time(),
            language=Language.PYTHON,
            encoding="utf-8",
            line_count=50,
            author="testuser",
            created_date=time.time() - 3600,
            modified_date=time.time(),
        )

        # Apply overrides
        for key, value in overrides.items():
            setattr(metadata, key, value)

        return metadata

    @patch("subprocess.run")
    def test_get_file_author_git_success(self, mock_run):
        """Test getting file author via git."""
        mock_run.return_value = MagicMock(returncode=0, stdout="John Doe\n")

        result = get_file_author(Path("test.py"))
        assert result == "John Doe"

    @patch("subprocess.run")
    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_git_failure(self, mock_run):
        """Test getting file author when git fails."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        with patch("pwd.getpwuid") as mock_pwd:
            mock_pwd.return_value = MagicMock(pw_name="testuser")
            with patch.object(Path, "stat") as mock_stat:
                mock_stat.return_value = MagicMock(st_uid=1000)

                result = get_file_author(Path("test.py"))
                assert result == "testuser"

    @patch("subprocess.run")
    @pytest.mark.skipif(sys.platform == "win32", reason="pwd module not available on Windows")
    def test_get_file_author_no_git_no_pwd(self, mock_run):
        """Test getting file author when both git and pwd fail."""
        mock_run.side_effect = FileNotFoundError()

        with patch("pwd.getpwuid", side_effect=ImportError()):
            result = get_file_author(Path("test.py"))
            assert result is None

    def test_apply_metadata_filters_size_min(self):
        """Test applying minimum size filter."""
        metadata = self.create_sample_metadata(size=2048)
        filters = MetadataFilters(min_size=1024)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(min_size=4096)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_size_max(self):
        """Test applying maximum size filter."""
        metadata = self.create_sample_metadata(size=1024)
        filters = MetadataFilters(max_size=2048)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(max_size=512)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_lines_min(self):
        """Test applying minimum lines filter."""
        metadata = self.create_sample_metadata(line_count=100)
        filters = MetadataFilters(min_lines=50)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(min_lines=200)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_lines_max(self):
        """Test applying maximum lines filter."""
        metadata = self.create_sample_metadata(line_count=50)
        filters = MetadataFilters(max_lines=100)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(max_lines=25)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_language(self):
        """Test applying language filter."""
        metadata = self.create_sample_metadata(language=Language.PYTHON)
        filters = MetadataFilters(languages={Language.PYTHON})

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(languages={Language.JAVASCRIPT})
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_author_pattern(self):
        """Test applying author pattern filter."""
        metadata = self.create_sample_metadata(author="john.doe")
        filters = MetadataFilters(author_pattern="john.*")

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(author_pattern="jane.*")
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_encoding_pattern(self):
        """Test applying encoding pattern filter."""
        metadata = self.create_sample_metadata(encoding="utf-8")
        filters = MetadataFilters(encoding_pattern="utf.*")

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(encoding_pattern="ascii")
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_modified_after(self):
        """Test applying modified after filter."""
        now = time.time()
        metadata = self.create_sample_metadata(mtime=now)
        filters = MetadataFilters(modified_after=now - 3600)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(modified_after=now + 3600)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_modified_before(self):
        """Test applying modified before filter."""
        now = time.time()
        metadata = self.create_sample_metadata(mtime=now - 3600)
        filters = MetadataFilters(modified_before=now)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(modified_before=now - 7200)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_created_after(self):
        """Test applying created after filter."""
        now = time.time()
        metadata = self.create_sample_metadata(created_date=now)
        filters = MetadataFilters(created_after=now - 3600)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(created_after=now + 3600)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_created_before(self):
        """Test applying created before filter."""
        now = time.time()
        metadata = self.create_sample_metadata(created_date=now - 3600)
        filters = MetadataFilters(created_before=now)

        assert apply_metadata_filters(metadata, filters) is True

        filters = MetadataFilters(created_before=now - 7200)
        assert apply_metadata_filters(metadata, filters) is False

    def test_apply_metadata_filters_multiple(self):
        """Test applying multiple filters together."""
        metadata = self.create_sample_metadata(
            size=2048, line_count=100, language=Language.PYTHON, author="john.doe"
        )

        filters = MetadataFilters(
            min_size=1024,
            max_size=4096,
            min_lines=50,
            max_lines=200,
            languages={Language.PYTHON},
            author_pattern="john.*",
        )

        assert apply_metadata_filters(metadata, filters) is True

        # Change one filter to make it fail
        filters.languages = {Language.JAVASCRIPT}
        assert apply_metadata_filters(metadata, filters) is False

    def test_create_metadata_filters_basic(self):
        """Test creating metadata filters with basic options."""
        filters = create_metadata_filters(
            min_size="1KB", max_size="1MB", min_lines=10, max_lines=1000
        )

        assert filters.min_size == 1024
        assert filters.max_size == 1024 * 1024
        assert filters.min_lines == 10
        assert filters.max_lines == 1000

    def test_create_metadata_filters_with_dates(self):
        """Test creating metadata filters with date options."""
        filters = create_metadata_filters(
            modified_after="1d",
            modified_before="2023-12-01",
            created_after="1w",
            created_before="2023-11-01",
        )

        assert filters.modified_after is not None
        assert filters.modified_before is not None
        assert filters.created_after is not None
        assert filters.created_before is not None

    def test_create_metadata_filters_with_patterns(self):
        """Test creating metadata filters with pattern options."""
        filters = create_metadata_filters(
            author_pattern="john.*",
            encoding_pattern="utf.*",
            languages={Language.PYTHON, Language.JAVASCRIPT},
        )

        assert filters.author_pattern == "john.*"
        assert filters.encoding_pattern == "utf.*"
        assert filters.languages == {Language.PYTHON, Language.JAVASCRIPT}

    def test_create_metadata_filters_invalid_size(self):
        """Test creating metadata filters with invalid size."""
        with pytest.raises(ValueError):
            create_metadata_filters(min_size="invalid")

    def test_create_metadata_filters_invalid_date(self):
        """Test creating metadata filters with invalid date."""
        with pytest.raises(ValueError):
            create_metadata_filters(modified_after="invalid")
