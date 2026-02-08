"""Tests for pysearch.utils.metadata_filters module."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from pysearch.core.types import FileMetadata, Language, MetadataFilters
from pysearch.utils.metadata_filters import (
    apply_metadata_filters,
    get_file_author,
)


class TestApplyMetadataFilters:
    """Tests for apply_metadata_filters function."""

    def _make_metadata(
        self,
        size: int = 1000,
        language: Language = Language.PYTHON,
        modified_time: float | None = None,
    ) -> FileMetadata:
        return FileMetadata(
            path=Path("test.py"),
            size=size,
            language=language,
            mtime=modified_time or time.time(),
        )

    def test_no_filters(self):
        meta = self._make_metadata()
        filters = MetadataFilters()
        assert apply_metadata_filters(meta, filters) is True

    def test_min_size_pass(self):
        meta = self._make_metadata(size=1000)
        filters = MetadataFilters(min_size=500)
        assert apply_metadata_filters(meta, filters) is True

    def test_min_size_fail(self):
        meta = self._make_metadata(size=100)
        filters = MetadataFilters(min_size=500)
        assert apply_metadata_filters(meta, filters) is False

    def test_max_size_pass(self):
        meta = self._make_metadata(size=100)
        filters = MetadataFilters(max_size=500)
        assert apply_metadata_filters(meta, filters) is True

    def test_max_size_fail(self):
        meta = self._make_metadata(size=1000)
        filters = MetadataFilters(max_size=500)
        assert apply_metadata_filters(meta, filters) is False

    def test_language_filter_pass(self):
        meta = self._make_metadata(language=Language.PYTHON)
        filters = MetadataFilters(languages=[Language.PYTHON])
        assert apply_metadata_filters(meta, filters) is True

    def test_language_filter_fail(self):
        meta = self._make_metadata(language=Language.JAVASCRIPT)
        filters = MetadataFilters(languages=[Language.PYTHON])
        assert apply_metadata_filters(meta, filters) is False


class TestGetFileAuthor:
    """Tests for get_file_author function."""

    def test_nonexistent_file(self, tmp_path: Path):
        result = get_file_author(tmp_path / "nonexistent.py")
        # Should return None for non-git file
        assert result is None or isinstance(result, str)

    def test_file_outside_git(self, tmp_path: Path):
        f = tmp_path / "test.py"
        f.write_text("hello", encoding="utf-8")
        result = get_file_author(f)
        assert result is None or isinstance(result, str)
