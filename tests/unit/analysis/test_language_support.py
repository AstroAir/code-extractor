"""Tests for pysearch.analysis.language_support module."""

from __future__ import annotations

import pytest

from pysearch.analysis.language_support import (
    LanguageProcessor,
    LanguageRegistry,
    language_registry,
)
from pysearch.core.types import Language


class TestLanguageRegistry:
    """Tests for LanguageRegistry class."""

    def test_singleton_registry(self):
        assert language_registry is not None

    def test_get_processor_python(self):
        processor = language_registry.get_processor("python")
        assert processor is not None

    def test_get_processor_unknown(self):
        processor = language_registry.get_processor("nonexistent_language_xyz")
        # Should return None or a default processor
        assert processor is None or isinstance(processor, LanguageProcessor)

    def test_supported_languages(self):
        langs = language_registry.get_supported_languages()
        assert isinstance(langs, (list, set))
        # May be empty if tree-sitter not installed

    def test_register_processor(self):
        from unittest.mock import MagicMock
        registry = LanguageRegistry()
        mock_processor = MagicMock(spec=LanguageProcessor)
        registry.register_processor("test_lang", mock_processor)
        assert registry.get_processor("test_lang") is mock_processor
