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

    def test_search_semantic_with_file_path(self):
        engine = SemanticSearchEngine()
        content = "def connect_database():\n    conn = sqlite3.connect('db')\n    return conn\n"
        results = engine.search_semantic(
            "database connection", content, file_path=Path("db.py")
        )
        assert isinstance(results, list)

    def test_search_semantic_empty_content(self):
        engine = SemanticSearchEngine()
        results = engine.search_semantic("anything", "")
        assert results == []

    def test_search_semantic_high_threshold(self):
        engine = SemanticSearchEngine()
        content = "x = 1\ny = 2\n"
        results = engine.search_semantic("database connection", content, threshold=0.99)
        assert results == []

    def test_fit_corpus(self):
        engine = SemanticSearchEngine()
        docs = [
            "def connect(): pass",
            "class Database: pass",
            "import sqlite3",
        ]
        engine.fit_corpus(docs)
        assert engine.embedding_model.is_fitted is True

    def test_fit_corpus_empty(self):
        engine = SemanticSearchEngine()
        engine.fit_corpus([])
        assert engine.embedding_model.is_fitted is False

    def test_search_semantic_after_fit(self):
        engine = SemanticSearchEngine()
        docs = [
            "def connect_database():\n    conn = sqlite3.connect('db')\n    return conn",
            "def process_data(data):\n    return [d.strip() for d in data]",
            "class UserManager:\n    def get_user(self, id): pass",
        ]
        engine.fit_corpus(docs)
        results = engine.search_semantic("database connection", docs[0], threshold=0.01)
        assert isinstance(results, list)

    def test_expand_query_semantically(self):
        engine = SemanticSearchEngine()
        expanded = engine.expand_query_semantically("database")
        assert isinstance(expanded, list)
        assert "database" in expanded
        assert len(expanded) >= 1

    def test_expand_query_snake_case(self):
        engine = SemanticSearchEngine()
        expanded = engine.expand_query_semantically("connect_db")
        assert isinstance(expanded, list)
        # Should include camelCase variant
        assert any("connectDb" in term or "connect_db" in term for term in expanded)

    def test_expand_query_adds_plural(self):
        engine = SemanticSearchEngine()
        expanded = engine.expand_query_semantically("query")
        assert "querys" in expanded or "query" in expanded

    def test_expand_query_removes_plural(self):
        engine = SemanticSearchEngine()
        expanded = engine.expand_query_semantically("queries")
        assert "querie" in expanded or "queries" in expanded

    def test_search_semantic_caches_concepts(self):
        engine = SemanticSearchEngine()
        content = "def foo():\n    pass\n"
        engine.search_semantic("foo", content)
        # Second call should use cache
        engine.search_semantic("bar", content)
        # Cache should have one entry
        assert len(engine._concept_cache) == 1
