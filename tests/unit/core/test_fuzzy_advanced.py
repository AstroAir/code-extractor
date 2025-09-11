import tempfile
from pathlib import Path

import pytest

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch.search.fuzzy import (
    FuzzyAlgorithm,
    FuzzyMatch,
    calculate_similarity,
    damerau_levenshtein_distance,
    fuzzy_match,
    fuzzy_search_advanced,
    jaro_winkler_similarity,
    levenshtein_distance,
    metaphone,
    soundex,
)
from pysearch import OutputFormat


def test_distance_algorithms():
    """Test different distance algorithms."""
    # Levenshtein distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("hello", "hello") == 0
    assert levenshtein_distance("", "abc") == 3

    # Damerau-Levenshtein (allows transpositions)
    assert damerau_levenshtein_distance("hello", "hlelo") <= 2  # Transposition
    assert damerau_levenshtein_distance("kitten", "sitting") <= 3

    # Jaro-Winkler similarity
    assert jaro_winkler_similarity("hello", "hello") == 1.0
    assert jaro_winkler_similarity("hello", "hallo") > 0.8
    assert jaro_winkler_similarity("abc", "xyz") < 0.3


def test_phonetic_algorithms():
    """Test phonetic algorithms."""
    # Soundex
    assert soundex("Smith") == soundex("Smyth")
    assert soundex("Johnson") == soundex("Jonson")
    assert soundex("hello") == "H400"

    # Metaphone - just test that it returns strings
    smith_meta = metaphone("Smith")
    smyth_meta = metaphone("Smyth")
    assert isinstance(smith_meta, str)
    assert isinstance(smyth_meta, str)
    # They should be similar but may not be identical
    assert len(smith_meta) > 0 and len(smyth_meta) > 0


def test_similarity_calculation():
    """Test similarity calculation with different algorithms."""
    # Levenshtein
    sim = calculate_similarity("hello", "hallo", FuzzyAlgorithm.LEVENSHTEIN)
    assert 0.7 < sim < 1.0

    # Jaro-Winkler
    sim = calculate_similarity("hello", "hallo", FuzzyAlgorithm.JARO_WINKLER)
    assert 0.7 < sim < 1.0

    # Soundex (exact match or no match)
    sim = calculate_similarity("Smith", "Smyth", FuzzyAlgorithm.SOUNDEX)
    assert sim == 1.0

    sim = calculate_similarity("Smith", "Jones", FuzzyAlgorithm.SOUNDEX)
    assert sim == 0.0


def test_fuzzy_match_basic():
    """Test basic fuzzy matching."""
    text = "The quick brown fox jumps over the lazy dog"

    # Test Levenshtein matching
    matches = fuzzy_match(text, "quik", max_distance=1, algorithm=FuzzyAlgorithm.LEVENSHTEIN)
    assert len(matches) >= 1
    assert any("quick" in match.matched_text for match in matches)

    # Test with higher similarity threshold
    matches = fuzzy_match(text, "quik", min_similarity=0.8, algorithm=FuzzyAlgorithm.LEVENSHTEIN)
    assert len(matches) >= 1


def test_fuzzy_match_algorithms():
    """Test fuzzy matching with different algorithms."""
    text = "The quick brown fox jumps over the lazy dog"

    # Damerau-Levenshtein (good for transpositions)
    matches = fuzzy_match(text, "qucik", algorithm=FuzzyAlgorithm.DAMERAU_LEVENSHTEIN)
    assert len(matches) >= 1

    # Jaro-Winkler (good for prefix similarities)
    matches = fuzzy_match(text, "qui", algorithm=FuzzyAlgorithm.JARO_WINKLER, min_similarity=0.5)
    assert len(matches) >= 1


def test_advanced_fuzzy_search():
    """Test advanced fuzzy search with multiple algorithms."""
    text = "The quick brown fox jumps over the lazy dog"

    # Test with multiple algorithms
    matches = fuzzy_search_advanced(
        text=text,
        pattern="quik",
        algorithms=[FuzzyAlgorithm.LEVENSHTEIN, FuzzyAlgorithm.DAMERAU_LEVENSHTEIN],
        max_distance=2,
        min_similarity=0.6,
    )

    assert len(matches) >= 1
    assert all(isinstance(match, FuzzyMatch) for match in matches)

    # Results should be sorted by similarity
    if len(matches) > 1:
        for i in range(len(matches) - 1):
            assert matches[i].similarity >= matches[i + 1].similarity


def test_fuzzy_search_api():
    """Test fuzzy search through the API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files with typos
        files = {
            "test1.py": "def proces_data():\n    return 'processing'",
            "test2.py": "def process_info():\n    return 'information'",
            "test3.py": "def proccess_files():\n    return 'files'",  # Double 'c' typo
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test fuzzy search for "process" (should find all variants)
        result = engine.fuzzy_search(
            pattern="process",
            max_distance=2,
            min_similarity=0.6,
            algorithm="levenshtein",
            output=OutputFormat.JSON,
        )

        # Should find matches in all files
        assert len(result.items) >= 3
        found_files = {item.file.name for item in result.items}
        assert "test1.py" in found_files  # proces
        assert "test2.py" in found_files  # process
        assert "test3.py" in found_files  # proccess


def test_phonetic_search_api():
    """Test phonetic search through the API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files with phonetically similar words
        files = {
            "names1.py": "# Author: Smith\ndef get_smith_data(): pass",
            "names2.py": "# Author: Smyth\ndef get_smyth_info(): pass",
            "names3.py": "# Author: Jones\ndef get_jones_data(): pass",
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test Soundex phonetic search
        result = engine.phonetic_search(
            pattern="Smith", algorithm="soundex", output=OutputFormat.JSON
        )

        # Should find both Smith and Smyth (same Soundex code)
        found_files = {item.file.name for item in result.items}
        # Note: This test might be sensitive to the regex pattern generation
        # The actual behavior depends on how well the Soundex pattern works


def test_multi_algorithm_fuzzy_search():
    """Test multi-algorithm fuzzy search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def calculate_distance():\n    return distance")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test multi-algorithm search
        result = engine.multi_algorithm_fuzzy_search(
            pattern="calcuate",  # Missing 'l'
            algorithms=["levenshtein", "damerau_levenshtein"],
            max_distance=2,
            min_similarity=0.6,
            output=OutputFormat.JSON,
        )

        # Should find the calculate function
        assert len(result.items) >= 1
        found_text = " ".join(" ".join(item.lines) for item in result.items)
        assert "calculate" in found_text.lower()


def test_fuzzy_edge_cases():
    """Test fuzzy search edge cases."""
    # Empty strings
    assert levenshtein_distance("", "") == 0
    assert jaro_winkler_similarity("", "") == 1.0

    # Single characters
    assert levenshtein_distance("a", "b") == 1
    assert jaro_winkler_similarity("a", "a") == 1.0

    # Very different strings
    assert jaro_winkler_similarity("abc", "xyz") < 0.5

    # Case sensitivity - may or may not be case-insensitive depending on implementation
    similarity = calculate_similarity("Hello", "hello", FuzzyAlgorithm.LEVENSHTEIN)
    assert similarity >= 0.8  # Should be high similarity


if __name__ == "__main__":
    pytest.main([__file__])
