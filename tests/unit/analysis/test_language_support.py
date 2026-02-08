"""Tests for pysearch.analysis.language_support module."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pysearch.analysis.language_support import (
    CodeChunk,
    LanguageConfig,
    LanguageProcessor,
    LanguageRegistry,
    TreeSitterProcessor,
    language_registry,
)
from pysearch.core.types import EntityType, Language


# ---------------------------------------------------------------------------
# CodeChunk
# ---------------------------------------------------------------------------
class TestCodeChunk:
    """Tests for CodeChunk dataclass."""

    def test_creation(self):
        chunk = CodeChunk(
            content="def foo(): pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
        )
        assert chunk.content == "def foo(): pass"
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.language == Language.PYTHON
        assert chunk.chunk_type == "code"

    def test_defaults(self):
        chunk = CodeChunk(content="x", start_line=1, end_line=1, language=Language.PYTHON)
        assert chunk.entity_name is None
        assert chunk.entity_type is None
        assert chunk.complexity_score == 0.0
        assert chunk.dependencies == []

    def test_with_entity_info(self):
        chunk = CodeChunk(
            content="class Foo: pass",
            start_line=1,
            end_line=1,
            language=Language.PYTHON,
            chunk_type="class",
            entity_name="Foo",
            entity_type=EntityType.CLASS,
            complexity_score=0.5,
            dependencies=["bar"],
        )
        assert chunk.entity_name == "Foo"
        assert chunk.entity_type == EntityType.CLASS
        assert chunk.complexity_score == 0.5
        assert chunk.dependencies == ["bar"]


# ---------------------------------------------------------------------------
# LanguageConfig
# ---------------------------------------------------------------------------
class TestLanguageConfig:
    """Tests for LanguageConfig dataclass."""

    def test_defaults(self):
        cfg = LanguageConfig()
        assert cfg.max_chunk_size == 1000
        assert cfg.respect_boundaries is True
        assert cfg.include_comments is True
        assert cfg.include_docstrings is True
        assert cfg.include_imports is True
        assert cfg.min_chunk_size == 50

    def test_custom(self):
        cfg = LanguageConfig(max_chunk_size=500, respect_boundaries=False, min_chunk_size=10)
        assert cfg.max_chunk_size == 500
        assert cfg.respect_boundaries is False
        assert cfg.min_chunk_size == 10


# ---------------------------------------------------------------------------
# TreeSitterProcessor
# ---------------------------------------------------------------------------
class TestTreeSitterProcessor:
    """Tests for TreeSitterProcessor class."""

    def test_init(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        assert proc.language == Language.PYTHON
        assert proc.config is cfg

    def test_extract_entities_no_parser(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        # If tree-sitter not available, parser is None → returns []
        if proc.parser is None:
            entities = proc.extract_entities("def foo(): pass")
            assert entities == []

    def test_analyze_dependencies_fallback(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        if proc.parser is None:
            deps = proc.analyze_dependencies("import os\nfrom sys import path")
            assert "os" in deps
            assert "sys" in deps

    def test_calculate_complexity_basic_fallback(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        if proc.parser is None:
            score = proc.calculate_complexity("if x:\n    for i in range(10):\n        pass")
            assert 0.0 <= score <= 1.0

    async def test_basic_chunk(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        content = "line1\nline2\nline3\nline4\nline5"
        chunks = []
        async for chunk in proc._basic_chunk(content, max_chunk_size=12):
            chunks.append(chunk)
        assert len(chunks) >= 1
        assert all(isinstance(c, CodeChunk) for c in chunks)
        # All content should be represented
        reconstructed = "\n".join(c.content for c in chunks)
        assert reconstructed == content

    async def test_chunk_code_fallback(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        # Force no parser to test fallback
        proc.parser = None
        content = "a = 1\nb = 2\nc = 3"
        chunks = []
        async for chunk in proc.chunk_code(content, max_chunk_size=1000):
            chunks.append(chunk)
        assert len(chunks) >= 1

    def test_analyze_dependencies_regex_python(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        deps = proc._analyze_dependencies_regex("import os\nfrom pathlib import Path")
        assert "os" in deps
        assert "pathlib" in deps

    def test_analyze_dependencies_regex_javascript(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.JAVASCRIPT, cfg)
        deps = proc._analyze_dependencies_regex(
            "import React from 'react';\nconst fs = require('fs');"
        )
        assert "react" in deps
        assert "fs" in deps

    def test_calculate_complexity_basic(self):
        cfg = LanguageConfig()
        proc = TreeSitterProcessor(Language.PYTHON, cfg)
        score = proc._calculate_complexity_basic(
            "if x:\n    for i in range(10):\n        while True:\n            pass"
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.0


# ---------------------------------------------------------------------------
# LanguageRegistry
# ---------------------------------------------------------------------------
class TestLanguageRegistry:
    """Tests for LanguageRegistry class."""

    def test_global_registry_exists(self):
        assert language_registry is not None
        assert isinstance(language_registry, LanguageRegistry)

    def test_get_processor_python(self):
        processor = language_registry.get_processor(Language.PYTHON)
        assert processor is not None
        assert isinstance(processor, LanguageProcessor)

    def test_get_processor_javascript(self):
        processor = language_registry.get_processor(Language.JAVASCRIPT)
        assert processor is not None

    def test_get_processor_nonexistent(self):
        processor = language_registry.get_processor(Language.UNKNOWN)
        assert processor is None

    def test_supported_languages(self):
        langs = language_registry.get_supported_languages()
        assert isinstance(langs, set)
        assert Language.PYTHON in langs

    def test_supported_languages_include_treesitter_langs(self):
        langs = language_registry.get_supported_languages()
        # Tree-sitter languages are always registered
        tree_sitter_langs = {
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.JAVA,
            Language.C,
            Language.CPP,
            Language.GO,
            Language.RUST,
        }
        # All tree-sitter languages must be present
        assert tree_sitter_langs.issubset(langs)
        # RegexProcessor languages should also be registered
        assert len(langs) > len(tree_sitter_langs)

    def test_register_processor(self):
        registry = LanguageRegistry()
        mock_processor = MagicMock(spec=LanguageProcessor)
        registry.register_processor(Language.UNKNOWN, mock_processor)
        assert registry.get_processor(Language.UNKNOWN) is mock_processor

    def test_new_registry_independent(self):
        registry = LanguageRegistry()
        assert Language.PYTHON in registry.get_supported_languages()
