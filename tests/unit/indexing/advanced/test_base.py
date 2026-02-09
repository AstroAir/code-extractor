"""Tests for pysearch.indexing.advanced.base module."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import pytest

from pysearch.indexing.advanced.base import CodebaseIndex


def _make_concrete_index_class():
    """Create a properly-implemented concrete subclass of CodebaseIndex."""

    class ConcreteIndex(CodebaseIndex):
        @property
        def artifact_id(self) -> str:
            return "test_index"

        @property
        def relative_expected_time(self) -> float:
            return 1.0

        async def update(self, tag, results, mark_complete, repo_name=None):
            yield  # async generator

        async def retrieve(self, query, tag, limit=50, **kwargs):
            return []

    return ConcreteIndex


class TestCodebaseIndex:
    """Tests for CodebaseIndex abstract base class."""

    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            CodebaseIndex()

    def test_incomplete_subclass_raises(self):
        class IncompleteIndex(CodebaseIndex):
            pass

        with pytest.raises(TypeError):
            IncompleteIndex()

    def test_missing_artifact_id_raises(self):
        class NoArtifactId(CodebaseIndex):
            @property
            def relative_expected_time(self) -> float:
                return 1.0

            async def update(self, tag, results, mark_complete, repo_name=None):
                yield

            async def retrieve(self, query, tag, limit=50, **kwargs):
                return []

        with pytest.raises(TypeError):
            NoArtifactId()

    def test_missing_relative_expected_time_raises(self):
        class NoTime(CodebaseIndex):
            @property
            def artifact_id(self) -> str:
                return "x"

            async def update(self, tag, results, mark_complete, repo_name=None):
                yield

            async def retrieve(self, query, tag, limit=50, **kwargs):
                return []

        with pytest.raises(TypeError):
            NoTime()

    def test_concrete_subclass_properties(self):
        ConcreteIndex = _make_concrete_index_class()
        idx = ConcreteIndex()
        assert idx.artifact_id == "test_index"
        assert idx.relative_expected_time == 1.0

    async def test_update_is_async_generator(self):
        ConcreteIndex = _make_concrete_index_class()
        idx = ConcreteIndex()
        gen = idx.update(tag=None, results=None, mark_complete=None)
        assert isinstance(gen, AsyncGenerator)
        # Exhaust the generator
        items = [item async for item in gen]
        assert isinstance(items, list)

    async def test_retrieve_returns_list(self):
        ConcreteIndex = _make_concrete_index_class()
        idx = ConcreteIndex()
        result = await idx.retrieve(query="test", tag=None, limit=10)
        assert result == []

    async def test_retrieve_accepts_kwargs(self):
        class KwargsIndex(CodebaseIndex):
            @property
            def artifact_id(self) -> str:
                return "kwargs_idx"

            @property
            def relative_expected_time(self) -> float:
                return 0.5

            async def update(self, tag, results, mark_complete, repo_name=None):
                yield

            async def retrieve(self, query, tag, limit=50, **kwargs):
                return [{"extra": kwargs.get("extra", None)}]

        idx = KwargsIndex()
        result = await idx.retrieve("q", None, limit=5, extra="val")
        assert result == [{"extra": "val"}]
