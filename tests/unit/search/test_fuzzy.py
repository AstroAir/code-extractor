"""Tests for pysearch.search.fuzzy module."""

from __future__ import annotations

import pytest

from pysearch.search.fuzzy import (
    FuzzyAlgorithm,
    FuzzyMatch,
    calculate_similarity,
    damerau_levenshtein_distance,
    fuzzy_match,
    fuzzy_search_advanced,
    jaro_winkler_similarity,
    levenshtein_distance,
    soundex,
    suggest_corrections,
)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_insertion(self):
        assert levenshtein_distance("abc", "abcd") == 1

    def test_single_deletion(self):
        assert levenshtein_distance("abcd", "abc") == 1

    def test_single_substitution(self):
        assert levenshtein_distance("abc", "adc") == 1

    def test_known_distance(self):
        assert levenshtein_distance("kitten", "sitting") == 3


class TestDamerauLevenshteinDistance:
    """Tests for damerau_levenshtein_distance function."""

    def test_identical(self):
        assert damerau_levenshtein_distance("hello", "hello") == 0

    def test_transposition(self):
        assert damerau_levenshtein_distance("ab", "ba") == 1

    def test_empty(self):
        assert damerau_levenshtein_distance("", "") == 0
        assert damerau_levenshtein_distance("a", "") == 1

    def test_transposition_cheaper(self):
        # Damerau-Levenshtein considers transposition as 1 edit
        d_dl = damerau_levenshtein_distance("ab", "ba")
        d_lev = levenshtein_distance("ab", "ba")
        assert d_dl <= d_lev


class TestFuzzyAlgorithm:
    """Tests for FuzzyAlgorithm enum."""

    def test_values(self):
        assert FuzzyAlgorithm.LEVENSHTEIN == "levenshtein"
        assert FuzzyAlgorithm.DAMERAU_LEVENSHTEIN == "damerau_levenshtein"
        assert FuzzyAlgorithm.JARO_WINKLER == "jaro_winkler"
        assert FuzzyAlgorithm.SOUNDEX == "soundex"


class TestFuzzyMatch:
    """Tests for FuzzyMatch dataclass."""

    def test_creation(self):
        m = FuzzyMatch(
            start=0, end=5, matched_text="hello",
            distance=1, similarity=0.8,
            algorithm=FuzzyAlgorithm.LEVENSHTEIN,
        )
        assert m.matched_text == "hello"
        assert m.similarity == 0.8
        assert m.distance == 1


class TestJaroWinklerSimilarity:
    """Tests for jaro_winkler_similarity function."""

    def test_identical(self):
        assert jaro_winkler_similarity("hello", "hello") == 1.0

    def test_completely_different(self):
        score = jaro_winkler_similarity("abc", "xyz")
        assert score < 0.5

    def test_similar_prefix(self):
        score = jaro_winkler_similarity("hello", "helpo")
        assert score > 0.8

    def test_empty_strings(self):
        assert jaro_winkler_similarity("", "") == 1.0


class TestSoundex:
    """Tests for soundex function."""

    def test_known_codes(self):
        assert soundex("Robert") == soundex("Rupert")

    def test_empty(self):
        result = soundex("")
        assert isinstance(result, str)

    def test_same_word(self):
        assert soundex("Smith") == soundex("Smith")


class TestFuzzyMatchFunction:
    """Tests for fuzzy_match function."""

    def test_exact_match(self):
        matches = fuzzy_match("hello world", "hello", max_distance=0)
        assert len(matches) >= 1

    def test_fuzzy_match_result(self):
        matches = fuzzy_match("hello world", "helo", max_distance=1)
        assert len(matches) >= 1

    def test_no_match(self):
        matches = fuzzy_match("hello", "xyz", max_distance=0)
        assert len(matches) == 0

    def test_with_jaro_winkler_algorithm(self):
        matches = fuzzy_match(
            "hello world", "helo",
            max_distance=1,
            algorithm=FuzzyAlgorithm.JARO_WINKLER,
        )
        assert isinstance(matches, list)

    def test_with_damerau_levenshtein(self):
        matches = fuzzy_match(
            "hello world", "hlelo",
            max_distance=1,
            algorithm=FuzzyAlgorithm.DAMERAU_LEVENSHTEIN,
        )
        assert isinstance(matches, list)


class TestFuzzySearchAdvanced:
    """Tests for fuzzy_search_advanced function."""

    def test_basic(self):
        matches = fuzzy_search_advanced("hello world", "hello")
        assert isinstance(matches, list)

    def test_with_algorithms(self):
        matches = fuzzy_search_advanced(
            "hello world", "helo",
            algorithms=[FuzzyAlgorithm.LEVENSHTEIN],
        )
        assert isinstance(matches, list)

    def test_with_multiple_algorithms(self):
        matches = fuzzy_search_advanced(
            "hello world", "helo",
            algorithms=[FuzzyAlgorithm.LEVENSHTEIN, FuzzyAlgorithm.JARO_WINKLER],
        )
        assert isinstance(matches, list)


class TestMetaphone:
    """Tests for metaphone function."""

    def test_basic(self):
        from pysearch.search.fuzzy import metaphone
        code = metaphone("Robert")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_empty(self):
        from pysearch.search.fuzzy import metaphone
        code = metaphone("")
        assert isinstance(code, str)

    def test_same_sound(self):
        from pysearch.search.fuzzy import metaphone
        code1 = metaphone("Smith")
        code2 = metaphone("Smith")
        assert code1 == code2


class TestFuzzyPattern:
    """Tests for fuzzy_pattern function."""

    def test_basic(self):
        from pysearch.search.fuzzy import fuzzy_pattern
        pattern = fuzzy_pattern("hello", max_distance=1)
        assert isinstance(pattern, str)
        assert len(pattern) > 0

    def test_zero_distance(self):
        from pysearch.search.fuzzy import fuzzy_pattern
        pattern = fuzzy_pattern("test", max_distance=0)
        assert isinstance(pattern, str)


class TestCalculateSimilarity:
    """Tests for calculate_similarity function."""

    def test_identical(self):
        score = calculate_similarity("hello", "hello", FuzzyAlgorithm.LEVENSHTEIN)
        assert score == 1.0

    def test_different(self):
        score = calculate_similarity("abc", "xyz", FuzzyAlgorithm.LEVENSHTEIN)
        assert score < 1.0

    def test_jaro_winkler(self):
        score = calculate_similarity("hello", "hello", FuzzyAlgorithm.JARO_WINKLER)
        assert score == 1.0

    def test_soundex(self):
        score = calculate_similarity("Robert", "Rupert", FuzzyAlgorithm.SOUNDEX)
        assert isinstance(score, float)

    def test_damerau_levenshtein(self):
        score = calculate_similarity("hello", "hello", FuzzyAlgorithm.DAMERAU_LEVENSHTEIN)
        assert score == 1.0

    def test_empty_strings(self):
        score = calculate_similarity("", "", FuzzyAlgorithm.LEVENSHTEIN)
        assert score == 1.0


class TestSuggestCorrections:
    """Tests for suggest_corrections function."""

    def test_basic(self):
        dictionary = ["hello", "world", "help", "heap"]
        suggestions = suggest_corrections("helo", dictionary)
        assert isinstance(suggestions, list)
        # Returns list of (word, score) tuples
        words = [s[0] if isinstance(s, tuple) else s for s in suggestions]
        assert "hello" in words or "help" in words

    def test_exact_match_in_dict(self):
        dictionary = ["hello", "world"]
        suggestions = suggest_corrections("hello", dictionary)
        assert isinstance(suggestions, list)

    def test_empty_dictionary(self):
        suggestions = suggest_corrections("hello", [])
        assert suggestions == []

    def test_max_suggestions(self):
        dictionary = ["hello", "hallo", "hullo", "helly", "helps"]
        suggestions = suggest_corrections("helo", dictionary, max_suggestions=2)
        assert len(suggestions) <= 2
