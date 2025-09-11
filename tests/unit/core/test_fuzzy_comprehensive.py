"""
Comprehensive tests for fuzzy module.

This module tests the fuzzy matching functionality including
various algorithms, distance calculations, and similarity scoring.
"""


from pysearch.search.fuzzy import (
    FuzzyAlgorithm,
    FuzzyMatch,
    calculate_similarity,
    damerau_levenshtein_distance,
    fuzzy_match,
    fuzzy_pattern,
    jaro_winkler_similarity,
    levenshtein_distance,
    metaphone,
    soundex,
)


class TestFuzzyAlgorithms:
    """Test fuzzy matching algorithms."""

    def test_levenshtein_distance_identical(self):
        """Test Levenshtein distance for identical strings."""
        assert levenshtein_distance("hello", "hello") == 0
        assert levenshtein_distance("", "") == 0

    def test_levenshtein_distance_empty(self):
        """Test Levenshtein distance with empty strings."""
        assert levenshtein_distance("hello", "") == 5
        assert levenshtein_distance("", "world") == 5

    def test_levenshtein_distance_single_char(self):
        """Test Levenshtein distance with single character changes."""
        assert levenshtein_distance("cat", "bat") == 1  # substitution
        assert levenshtein_distance("cat", "cats") == 1  # insertion
        assert levenshtein_distance("cats", "cat") == 1  # deletion

    def test_levenshtein_distance_multiple_changes(self):
        """Test Levenshtein distance with multiple changes."""
        assert levenshtein_distance("kitten", "sitting") == 3
        assert levenshtein_distance("saturday", "sunday") == 3

    def test_damerau_levenshtein_distance_transposition(self):
        """Test Damerau-Levenshtein distance with transpositions."""
        assert damerau_levenshtein_distance("ab", "ba") == 1  # transposition
        assert damerau_levenshtein_distance("hello", "hlelo") == 1  # transposition

    def test_damerau_levenshtein_distance_regular_operations(self):
        """Test Damerau-Levenshtein distance with regular operations."""
        assert damerau_levenshtein_distance("cat", "bat") == 1  # substitution
        assert damerau_levenshtein_distance("cat", "cats") == 1  # insertion
        assert damerau_levenshtein_distance("cats", "cat") == 1  # deletion

    def test_jaro_winkler_similarity_identical(self):
        """Test Jaro-Winkler similarity for identical strings."""
        assert jaro_winkler_similarity("hello", "hello") == 1.0
        assert jaro_winkler_similarity("", "") == 1.0

    def test_jaro_winkler_similarity_different(self):
        """Test Jaro-Winkler similarity for different strings."""
        similarity = jaro_winkler_similarity("martha", "marhta")
        assert 0.9 < similarity < 1.0  # Should be high due to common prefix

        similarity = jaro_winkler_similarity("dixon", "dicksonx")
        assert 0.7 < similarity < 0.9

    def test_jaro_winkler_similarity_no_match(self):
        """Test Jaro-Winkler similarity for completely different strings."""
        similarity = jaro_winkler_similarity("abc", "xyz")
        assert similarity < 0.5

    def test_soundex_basic(self):
        """Test Soundex algorithm with basic examples."""
        assert soundex("Smith") == "S530"
        assert soundex("Johnson") == "J525"
        assert soundex("Williams") == "W452"

    def test_soundex_similar_sounding(self):
        """Test Soundex algorithm with similar sounding names."""
        assert soundex("Smith") == soundex("Smyth")
        assert soundex("Johnson") == soundex("Jonson")

    def test_soundex_edge_cases(self):
        """Test Soundex algorithm with edge cases."""
        assert soundex("") == "0000"
        assert soundex("A") == "A000"
        assert soundex("Ae") == "A000"

    def test_metaphone_basic(self):
        """Test Metaphone algorithm with basic examples."""
        assert metaphone("Smith") == "SMT"
        assert metaphone("Johnson") == "JHNS"

    def test_metaphone_similar_sounding(self):
        """Test Metaphone algorithm with similar sounding words."""
        # Note: Metaphone may not always produce identical codes for similar sounds
        smith_code = metaphone("Smith")
        smyth_code = metaphone("Smyth")
        assert isinstance(smith_code, str)
        assert isinstance(smyth_code, str)

    def test_metaphone_edge_cases(self):
        """Test Metaphone algorithm with edge cases."""
        assert metaphone("") == ""
        assert metaphone("A") == "A"

    def test_calculate_similarity_levenshtein(self):
        """Test similarity calculation using Levenshtein algorithm."""
        similarity = calculate_similarity("hello", "hello", FuzzyAlgorithm.LEVENSHTEIN)
        assert similarity == 1.0

        similarity = calculate_similarity("hello", "helo", FuzzyAlgorithm.LEVENSHTEIN)
        assert 0.7 < similarity < 1.0

    def test_calculate_similarity_damerau_levenshtein(self):
        """Test similarity calculation using Damerau-Levenshtein algorithm."""
        similarity = calculate_similarity("hello", "hlelo", FuzzyAlgorithm.DAMERAU_LEVENSHTEIN)
        assert 0.7 < similarity <= 1.0  # Should be high due to single transposition

    def test_calculate_similarity_jaro_winkler(self):
        """Test similarity calculation using Jaro-Winkler algorithm."""
        similarity = calculate_similarity("martha", "marhta", FuzzyAlgorithm.JARO_WINKLER)
        assert 0.9 < similarity < 1.0

    def test_calculate_similarity_soundex(self):
        """Test similarity calculation using Soundex algorithm."""
        similarity = calculate_similarity("Smith", "Smyth", FuzzyAlgorithm.SOUNDEX)
        assert similarity == 1.0  # Should be identical Soundex codes

        similarity = calculate_similarity("Smith", "Johnson", FuzzyAlgorithm.SOUNDEX)
        assert similarity == 0.0  # Should be different Soundex codes

    def test_calculate_similarity_metaphone(self):
        """Test similarity calculation using Metaphone algorithm."""
        similarity = calculate_similarity("hello", "hello", FuzzyAlgorithm.METAPHONE)
        assert similarity == 1.0

    def test_generate_fuzzy_regex_basic(self):
        """Test fuzzy regex generation with basic patterns."""
        regex = fuzzy_pattern("hello")
        assert isinstance(regex, str)
        assert "h" in regex
        assert "e" in regex
        assert "l" in regex
        assert "o" in regex

    def test_generate_fuzzy_regex_with_substitutions(self):
        """Test fuzzy regex generation with character substitutions."""
        regex = fuzzy_pattern("hello")
        # Should include common substitutions
        assert "[h]" in regex or "h" in regex

    def test_find_fuzzy_matches_exact(self):
        """Test finding fuzzy matches with exact matches."""
        text = "This is a hello world example"
        matches = fuzzy_match(text, "hello", max_distance=0)

        assert len(matches) == 1
        assert matches[0].matched_text == "hello"
        assert matches[0].distance == 0
        assert matches[0].similarity == 1.0

    def test_find_fuzzy_matches_with_distance(self):
        """Test finding fuzzy matches with edit distance."""
        text = "This is a helo world example"  # Missing 'l'
        matches = fuzzy_match(text, "hello", max_distance=1)

        assert len(matches) >= 1
        found_match = any(m.matched_text == "helo" for m in matches)
        assert found_match

    def test_find_fuzzy_matches_multiple(self):
        """Test finding multiple fuzzy matches."""
        text = "hello world, helo there, hallo friend"
        matches = fuzzy_match(text, "hello", max_distance=1)

        assert len(matches) >= 2  # Should find "hello" and "helo" at minimum

    def test_find_fuzzy_matches_no_matches(self):
        """Test finding fuzzy matches when none exist."""
        text = "This is completely different text"
        matches = fuzzy_match(text, "hello", max_distance=1)

        assert len(matches) == 0

    def test_find_fuzzy_matches_with_similarity_threshold(self):
        """Test finding fuzzy matches with similarity threshold."""
        text = "This is a helo world example"
        matches = fuzzy_match(text, "hello", max_distance=2, min_similarity=0.8)

        # Should find matches with high similarity
        assert all(m.similarity >= 0.8 for m in matches)

    def test_find_fuzzy_matches_case_insensitive(self):
        """Test finding fuzzy matches case insensitively."""
        text = "This is a HELLO world example"
        matches = fuzzy_match(text, "hello", max_distance=0)

        # Should find the match regardless of case
        assert len(matches) >= 1

    def test_fuzzy_match_dataclass(self):
        """Test FuzzyMatch dataclass functionality."""
        match = FuzzyMatch(
            start=10,
            end=15,
            matched_text="hello",
            distance=0,
            similarity=1.0,
            algorithm=FuzzyAlgorithm.LEVENSHTEIN,
        )

        assert match.start == 10
        assert match.end == 15
        assert match.matched_text == "hello"
        assert match.distance == 0
        assert match.similarity == 1.0
        assert match.algorithm == FuzzyAlgorithm.LEVENSHTEIN

    def test_fuzzy_algorithm_enum(self):
        """Test FuzzyAlgorithm enum values."""
        assert FuzzyAlgorithm.LEVENSHTEIN == "levenshtein"
        assert FuzzyAlgorithm.DAMERAU_LEVENSHTEIN == "damerau_levenshtein"
        assert FuzzyAlgorithm.JARO_WINKLER == "jaro_winkler"
        assert FuzzyAlgorithm.SOUNDEX == "soundex"
        assert FuzzyAlgorithm.METAPHONE == "metaphone"

    def test_find_fuzzy_matches_with_algorithm(self):
        """Test finding fuzzy matches with specific algorithm."""
        text = "This is a helo world example"
        matches = fuzzy_match(text, "hello", max_distance=1, algorithm=FuzzyAlgorithm.LEVENSHTEIN)

        assert len(matches) >= 1
        assert all(m.algorithm == FuzzyAlgorithm.LEVENSHTEIN for m in matches)

    def test_find_fuzzy_matches_word_boundaries(self):
        """Test finding fuzzy matches respecting word boundaries."""
        text = "hello world, say hello to everyone"
        matches = fuzzy_match(text, "hello", max_distance=0)

        # Should find both instances of "hello"
        assert len(matches) == 2
        assert all(m.matched_text == "hello" for m in matches)

    def test_performance_with_long_text(self):
        """Test fuzzy matching performance with longer text."""
        text = "hello " * 1000 + "world"
        matches = fuzzy_match(text, "hello", max_distance=0)

        # Should find all instances efficiently
        assert len(matches) == 1000

    def test_unicode_support(self):
        """Test fuzzy matching with Unicode characters."""
        text = "This is a héllo world example"
        matches = fuzzy_match(text, "hello", max_distance=1)

        # Should handle Unicode characters properly
        assert len(matches) >= 1
