"""Tests for pysearch.indexing.advanced.base module."""

from __future__ import annotations

import pytest

from pysearch.indexing.advanced.base import CodebaseIndex


class TestCodebaseIndex:
    """Tests for CodebaseIndex abstract base class."""

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            CodebaseIndex()

    def test_subclass_must_implement(self):
        class IncompleteIndex(CodebaseIndex):
            pass

        with pytest.raises(TypeError):
            IncompleteIndex()

    def test_concrete_subclass(self):
        class ConcreteIndex(CodebaseIndex):
            @property
            def artifact_id(self) -> str:
                return "test_index"

            @property
            def relative_expected_time(self) -> float:
                return 1.0

            async def update(self, tag, results, mark_complete, cache_manager):
                pass

            async def retrieve(self, query, tags):
                return []

        idx = ConcreteIndex()
        assert idx.artifact_id == "test_index"
        assert idx.relative_expected_time == 1.0
