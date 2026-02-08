"""Tests for pysearch.search.semantic_advanced module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.search.semantic_advanced import (
    CodeSemanticAnalyzer,
    SemanticConcept,
    SemanticEmbedding,
    SemanticMatch,
    SemanticSearchEngine,
)
from pysearch.core.types import SearchItem


class TestSemanticConcept:
    """Tests for SemanticConcept dataclass."""

    def test_creation(self):
        c = SemanticConcept(
            name="connect",
            category="function",
            context="database connection",
            confidence=0.9,
            line_number=5,
        )
        assert c.name == "connect"
        assert c.category == "function"
        assert c.confidence == 0.9

    def test_defaults(self):
        c = SemanticConcept(
            name="x", category="var", context="", confidence=1.0, line_number=1,
        )
        assert c.metadata == {}


class TestSemanticEmbedding:
    """Tests for SemanticEmbedding class."""

    def test_init(self):
        emb = SemanticEmbedding()
        assert emb is not None

    def test_fit_and_transform(self):
        emb = SemanticEmbedding()
        docs = ["hello world", "database connection", "web api"]
        emb.fit(docs)
        vector = emb.transform("hello world")
        assert vector is not None

    def test_transform_empty(self):
        emb = SemanticEmbedding()
        emb.fit(["some text"])
        vector = emb.transform("")
        assert vector is not None

    def test_cosine_similarity(self):
        emb = SemanticEmbedding()
        # cosine_similarity expects sparse dict vectors: {index: weight}
        v1 = {0: 1.0, 1: 0.5}
        v2 = {0: 1.0, 1: 0.5}
        v3 = {2: 1.0, 3: 0.5}
        sim_same = emb.cosine_similarity(v1, v2)
        sim_diff = emb.cosine_similarity(v1, v3)
        assert sim_same > sim_diff


class TestCodeSemanticAnalyzer:
    """Tests for CodeSemanticAnalyzer class."""

    def test_init(self):
        analyzer = CodeSemanticAnalyzer()
        assert analyzer is not None

    def test_extract_concepts(self):
        analyzer = CodeSemanticAnalyzer()
        code = "def connect_database():\n    conn = sqlite3.connect('db')\n    return conn"
        concepts = analyzer.extract_concepts(code)
        assert isinstance(concepts, list)

    def test_extract_concepts_empty(self):
        analyzer = CodeSemanticAnalyzer()
        concepts = analyzer.extract_concepts("")
        assert isinstance(concepts, list)


class TestSemanticSearchEngine:
    """Tests for SemanticSearchEngine class."""

    def test_init(self):
        engine = SemanticSearchEngine()
        assert engine is not None

    def test_search_semantic(self):
        engine = SemanticSearchEngine()
        content = "def process_data():\n    db = connect()\n    return db.query('SELECT *')"
        results = engine.search_semantic("database query", content)
        assert isinstance(results, list)
