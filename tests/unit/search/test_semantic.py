"""Tests for pysearch.search.semantic module."""

from __future__ import annotations

from pysearch.search.semantic import (
    CONCEPT_PATTERNS,
    concept_to_patterns,
    expand_semantic_query,
    semantic_similarity_score,
)


class TestConceptPatterns:
    """Tests for CONCEPT_PATTERNS constant."""

    def test_has_database_patterns(self):
        assert "database" in CONCEPT_PATTERNS

    def test_has_web_patterns(self):
        assert "web" in CONCEPT_PATTERNS

    def test_has_testing_patterns(self):
        assert "testing" in CONCEPT_PATTERNS

    def test_has_async_patterns(self):
        assert "async" in CONCEPT_PATTERNS

    def test_patterns_are_lists(self):
        for _key, patterns in CONCEPT_PATTERNS.items():
            assert isinstance(patterns, list)
            assert len(patterns) > 0


class TestSemanticSimilarityScore:
    """Tests for semantic_similarity_score function."""

    def test_database_code(self):
        code = "conn = sqlite3.connect('app.db')\ncursor = conn.cursor()"
        score = semantic_similarity_score(code, "database connection")
        assert score > 0

    def test_web_code(self):
        code = "response = requests.get('http://api.example.com')"
        score = semantic_similarity_score(code, "http request")
        assert isinstance(score, float)

    def test_unrelated(self):
        code = "x = 1 + 2"
        score = semantic_similarity_score(code, "database connection")
        assert score == 0.0

    def test_empty_content(self):
        score = semantic_similarity_score("", "anything")
        assert score == 0.0

    def test_empty_query(self):
        score = semantic_similarity_score("some code", "")
        assert score == 0.0


class TestConceptToPatterns:
    """Tests for concept_to_patterns function."""

    def test_known_concept(self):
        patterns = concept_to_patterns("database")
        assert isinstance(patterns, list)
        assert len(patterns) > 0

    def test_unknown_concept(self):
        patterns = concept_to_patterns("nonexistent_category")
        # concept_to_patterns generates word-boundary patterns even for unknown concepts
        assert isinstance(patterns, list)


class TestExpandSemanticQuery:
    """Tests for expand_semantic_query function."""

    def test_basic(self):
        expanded = expand_semantic_query("database connection")
        assert isinstance(expanded, list)
        assert len(expanded) > 0

    def test_empty_query(self):
        expanded = expand_semantic_query("")
        assert isinstance(expanded, list)
