from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class FuzzyAlgorithm(str, Enum):
    """Available fuzzy matching algorithms."""

    LEVENSHTEIN = "levenshtein"
    DAMERAU_LEVENSHTEIN = "damerau_levenshtein"
    JARO_WINKLER = "jaro_winkler"
    SOUNDEX = "soundex"
    METAPHONE = "metaphone"


@dataclass
class FuzzyMatch:
    """Represents a fuzzy match result."""

    start: int
    end: int
    matched_text: str
    distance: int
    similarity: float  # 0.0 to 1.0
    algorithm: FuzzyAlgorithm


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Damerau-Levenshtein distance (allows transpositions).
    More accurate for typos involving character swaps.
    """
    len1, len2 = len(s1), len(s2)

    # Create a dictionary for character positions
    da = {}
    for char in s1 + s2:
        da[char] = 0

    # Create the distance matrix
    max_dist = len1 + len2
    H = [[max_dist for _ in range(len2 + 2)] for _ in range(len1 + 2)]

    H[0][0] = max_dist
    for i in range(0, len1 + 1):
        H[i + 1][0] = max_dist
        H[i + 1][1] = i
    for j in range(0, len2 + 1):
        H[0][j + 1] = max_dist
        H[1][j + 1] = j

    for i in range(1, len1 + 1):
        db = 0
        for j in range(1, len2 + 1):
            k = da[s2[j - 1]]
            l = db
            if s1[i - 1] == s2[j - 1]:
                cost = 0
                db = j
            else:
                cost = 1

            H[i + 1][j + 1] = min(
                H[i][j] + cost,  # substitution
                H[i + 1][j] + 1,  # insertion
                H[i][j + 1] + 1,  # deletion
                H[k][l] + (i - k - 1) + 1 + (j - l - 1),  # transposition
            )

        da[s1[i - 1]] = i

    return H[len1 + 1][len2 + 1]


def jaro_winkler_similarity(s1: str, s2: str) -> float:
    """
    Calculate Jaro-Winkler similarity (0.0 to 1.0).
    Good for strings with common prefixes.
    """
    if s1 == s2:
        return 1.0

    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0

    # Calculate the match window
    match_window = max(len1, len2) // 2 - 1
    if match_window < 0:
        match_window = 0

    # Initialize match arrays
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - match_window)
        end = min(i + match_window + 1, len2)

        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    # Calculate Jaro similarity
    jaro = (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches) / 3

    # Calculate common prefix length (up to 4 characters)
    prefix_len = 0
    for i in range(min(len1, len2, 4)):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break

    # Calculate Jaro-Winkler similarity
    return jaro + (0.1 * prefix_len * (1 - jaro))


def soundex(s: str) -> str:
    """
    Generate Soundex code for phonetic matching.
    Useful for matching words that sound similar.
    """
    if not s:
        return "0000"

    s = s.upper()
    soundex_code = s[0]

    # Mapping of letters to numbers
    mapping = {
        "B": "1",
        "F": "1",
        "P": "1",
        "V": "1",
        "C": "2",
        "G": "2",
        "J": "2",
        "K": "2",
        "Q": "2",
        "S": "2",
        "X": "2",
        "Z": "2",
        "D": "3",
        "T": "3",
        "L": "4",
        "M": "5",
        "N": "5",
        "R": "6",
    }

    for char in s[1:]:
        if char in mapping:
            code = mapping[char]
            if code != soundex_code[-1]:  # Avoid consecutive duplicates
                soundex_code += code
        # Vowels and other letters are ignored

    # Pad with zeros or truncate to 4 characters
    soundex_code = (soundex_code + "000")[:4]
    return soundex_code


def metaphone(s: str) -> str:
    """
    Generate Double Metaphone code for phonetic matching.
    Simplified version for basic phonetic similarity.
    """
    if not s:
        return ""

    s = s.upper()
    result = ""

    # Simple metaphone rules (simplified)
    replacements = [
        ("PH", "F"),
        ("GH", "F"),
        ("CK", "K"),
        ("SCH", "SK"),
        ("QU", "KW"),
        ("TH", "T"),
        ("SH", "S"),
        ("CH", "K"),
        ("WH", "W"),
        ("X", "KS"),
    ]

    for old, new in replacements:
        s = s.replace(old, new)

    # Remove vowels except at the beginning
    if s:
        result = s[0]
        for char in s[1:]:
            if char not in "AEIOU":
                result += char

    return result[:4]  # Limit to 4 characters


def calculate_similarity(s1: str, s2: str, algorithm: FuzzyAlgorithm) -> float:
    """Calculate similarity score (0.0 to 1.0) using specified algorithm."""
    if algorithm == FuzzyAlgorithm.LEVENSHTEIN:
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    elif algorithm == FuzzyAlgorithm.DAMERAU_LEVENSHTEIN:
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = damerau_levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)

    elif algorithm == FuzzyAlgorithm.JARO_WINKLER:
        return jaro_winkler_similarity(s1, s2)

    elif algorithm == FuzzyAlgorithm.SOUNDEX:
        return 1.0 if soundex(s1) == soundex(s2) else 0.0

    elif algorithm == FuzzyAlgorithm.METAPHONE:
        return 1.0 if metaphone(s1) == metaphone(s2) else 0.0

    else:
        # Default to Levenshtein
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 1.0
        distance = levenshtein_distance(s1, s2)
        return 1.0 - (distance / max_len)


def fuzzy_pattern(
    pattern: str, max_distance: int = 2, algorithm: FuzzyAlgorithm = FuzzyAlgorithm.LEVENSHTEIN
) -> str:
    """
    Generate a regex pattern for fuzzy matching with edit distance.
    Enhanced with better character substitutions and algorithm-specific patterns.
    """
    if max_distance <= 0:
        return re.escape(pattern)

    # Enhanced character substitutions for common typos
    substitutions = {
        "a": "[a@4]",
        "e": "[e3]",
        "i": "[i1l!]",
        "o": "[o0]",
        "s": "[s5$z]",
        "t": "[t7+]",
        "l": "[l1i!]",
        "g": "[g9q]",
        "b": "[b6]",
        "c": "[ck]",
        "k": "[ck]",
        "f": "[fph]",
        "ph": "[f]",
        "v": "[vw]",
        "w": "[vw]",
        "y": "[yi]",
        "z": "[zs]",
        "x": "[xks]",
        "qu": "[kw]",
        "th": "[t]",
        "sh": "[s]",
        "ch": "[k]",
    }

    # Algorithm-specific pattern generation
    if algorithm == FuzzyAlgorithm.SOUNDEX:
        # For Soundex, focus on phonetic similarities
        return _generate_soundex_pattern(pattern)
    elif algorithm == FuzzyAlgorithm.METAPHONE:
        # For Metaphone, use phonetic substitutions
        return _generate_metaphone_pattern(pattern)

    # Default: Enhanced Levenshtein/Damerau-Levenshtein pattern
    pattern_lower = pattern.lower()

    # Handle multi-character substitutions first
    for old, new in [
        ("ph", "[fph]"),
        ("qu", "[qkw]"),
        ("th", "[t]"),
        ("sh", "[s]"),
        ("ch", "[kc]"),
    ]:
        pattern_lower = pattern_lower.replace(old, new)

    # Build fuzzy pattern with character classes
    fuzzy_chars = []
    i = 0
    while i < len(pattern_lower):
        char = pattern_lower[i]
        if char in substitutions:
            fuzzy_chars.append(substitutions[char])
        else:
            fuzzy_chars.append(re.escape(char))
        i += 1

    # Generate pattern based on max_distance
    if max_distance == 1:
        # Allow single character insertions, deletions, or substitutions
        parts = []
        for i, char_class in enumerate(fuzzy_chars):
            if i > 0:
                parts.append("[a-zA-Z0-9_]?")  # Optional insertion
            parts.append(f"({char_class})?")  # Optional character (deletion)
            parts.append(char_class)  # Required character
        return "".join(parts)

    elif max_distance >= 2:
        # More flexible pattern for higher edit distances
        pattern_parts = []
        for i, char_class in enumerate(fuzzy_chars):
            # Make some characters optional (deletions)
            if i % 2 == 0:
                pattern_parts.append(f"({char_class})?")
            else:
                pattern_parts.append(char_class)

            # Allow insertions between characters
            if i < len(fuzzy_chars) - 1:
                pattern_parts.append("[a-zA-Z0-9_]{0,2}")  # Up to 2 extra chars

        return "".join(pattern_parts)

    return "".join(fuzzy_chars)


def _generate_soundex_pattern(pattern: str) -> str:
    """Generate regex pattern for Soundex-based matching."""
    soundex_code = soundex(pattern)

    # Create pattern that matches words with same Soundex code
    # This is a simplified approach - in practice, you'd pre-compute Soundex codes
    first_char = pattern[0].lower() if pattern else ""

    # Pattern that starts with same letter and has similar consonant structure
    consonant_groups = {
        "1": "[bfpv]",
        "2": "[cgjkqsxz]",
        "3": "[dt]",
        "4": "[l]",
        "5": "[mn]",
        "6": "[r]",
    }

    pattern_parts = [f"[{first_char.upper()}{first_char.lower()}]"]
    for code in soundex_code[1:]:
        if code in consonant_groups:
            pattern_parts.append(f"{consonant_groups[code]}*")
        elif code == "0":
            pattern_parts.append("[aeiou]*")

    return "".join(pattern_parts)


def _generate_metaphone_pattern(pattern: str) -> str:
    """Generate regex pattern for Metaphone-based matching."""
    metaphone_code = metaphone(pattern)

    # Create pattern based on metaphone transformations
    pattern_lower = pattern.lower()

    # Apply metaphone-like transformations
    transformations = [
        ("ph", "f"),
        ("gh", "f"),
        ("ck", "k"),
        ("qu", "kw"),
        ("th", "t"),
        ("sh", "s"),
        ("ch", "k"),
        ("wh", "w"),
        ("x", "ks"),
    ]

    for old, new in transformations:
        pattern_lower = pattern_lower.replace(old, f"({old}|{new})")

    # Remove vowels except at the beginning (metaphone style)
    if pattern_lower:
        result = pattern_lower[0]
        for char in pattern_lower[1:]:
            if char not in "aeiou":
                result += char
            else:
                result += "[aeiou]?"  # Optional vowels
    else:
        result = pattern_lower

    return result


def fuzzy_match(
    text: str,
    pattern: str,
    max_distance: int = 2,
    min_similarity: float = 0.6,
    algorithm: FuzzyAlgorithm = FuzzyAlgorithm.LEVENSHTEIN,
    word_boundaries: bool = True,
) -> list[FuzzyMatch]:
    """
    Find fuzzy matches in text using specified algorithm.

    Args:
        text: Text to search in
        pattern: Pattern to search for
        max_distance: Maximum edit distance (for distance-based algorithms)
        min_similarity: Minimum similarity score (0.0 to 1.0)
        algorithm: Fuzzy matching algorithm to use
        word_boundaries: Whether to match only at word boundaries

    Returns:
        List of FuzzyMatch objects sorted by similarity (best first)
    """
    matches: list[FuzzyMatch] = []
    pattern_lower = pattern.lower()

    # Choose regex pattern based on word boundaries
    if word_boundaries:
        word_pattern = r"\b\w+\b"
    else:
        word_pattern = r"\w+"

    for match in re.finditer(word_pattern, text):
        word = match.group()
        word_lower = word.lower()

        # Calculate similarity using specified algorithm
        similarity = calculate_similarity(word_lower, pattern_lower, algorithm)

        # For distance-based algorithms, also check max_distance
        if algorithm in [FuzzyAlgorithm.LEVENSHTEIN, FuzzyAlgorithm.DAMERAU_LEVENSHTEIN]:
            if algorithm == FuzzyAlgorithm.LEVENSHTEIN:
                distance = levenshtein_distance(word_lower, pattern_lower)
            else:
                distance = damerau_levenshtein_distance(word_lower, pattern_lower)

            if distance > max_distance:
                continue
        else:
            # For similarity-based algorithms, estimate distance
            distance = int((1.0 - similarity) * max(len(word), len(pattern)))

        # Check minimum similarity threshold
        if similarity >= min_similarity:
            matches.append(
                FuzzyMatch(
                    start=match.start(),
                    end=match.end(),
                    matched_text=word,
                    distance=distance,
                    similarity=similarity,
                    algorithm=algorithm,
                )
            )

    # Sort by similarity (best first), then by distance (lowest first)
    matches.sort(key=lambda m: (-m.similarity, m.distance))
    return matches


def fuzzy_search_advanced(
    text: str,
    pattern: str,
    algorithms: list[FuzzyAlgorithm] | None = None,
    max_distance: int = 2,
    min_similarity: float = 0.6,
    combine_results: bool = True,
) -> list[FuzzyMatch]:
    """
    Advanced fuzzy search using multiple algorithms.

    Args:
        text: Text to search in
        pattern: Pattern to search for
        algorithms: List of algorithms to use (default: all)
        max_distance: Maximum edit distance
        min_similarity: Minimum similarity score
        combine_results: Whether to combine and deduplicate results

    Returns:
        List of FuzzyMatch objects
    """
    if algorithms is None:
        algorithms = list(FuzzyAlgorithm)

    all_matches: list[FuzzyMatch] = []

    for algorithm in algorithms:
        matches = fuzzy_match(
            text=text,
            pattern=pattern,
            max_distance=max_distance,
            min_similarity=min_similarity,
            algorithm=algorithm,
        )
        all_matches.extend(matches)

    if not combine_results:
        return all_matches

    # Combine and deduplicate results
    # Group matches by position and take the best similarity
    position_matches: dict[tuple[int, int], FuzzyMatch] = {}

    for match in all_matches:
        key = (match.start, match.end)
        if key not in position_matches or match.similarity > position_matches[key].similarity:
            position_matches[key] = match

    # Sort combined results
    combined_matches = list(position_matches.values())
    combined_matches.sort(key=lambda m: (-m.similarity, m.distance))

    return combined_matches


def suggest_corrections(
    word: str,
    dictionary: list[str],
    max_suggestions: int = 5,
    algorithm: FuzzyAlgorithm = FuzzyAlgorithm.DAMERAU_LEVENSHTEIN,
) -> list[tuple[str, float]]:
    """
    Suggest spelling corrections for a word based on a dictionary.

    Args:
        word: Word to correct
        dictionary: List of valid words
        max_suggestions: Maximum number of suggestions
        algorithm: Algorithm to use for similarity calculation

    Returns:
        List of (suggestion, similarity) tuples sorted by similarity
    """
    suggestions: list[tuple[str, float]] = []
    word_lower = word.lower()

    for dict_word in dictionary:
        similarity = calculate_similarity(word_lower, dict_word.lower(), algorithm)
        if similarity > 0.3:  # Minimum threshold for suggestions
            suggestions.append((dict_word, similarity))

    # Sort by similarity and return top suggestions
    suggestions.sort(key=lambda x: x[1], reverse=True)
    return suggestions[:max_suggestions]


def fuzzy_regex_pattern(
    pattern: str, max_distance: int = 2, algorithm: FuzzyAlgorithm = FuzzyAlgorithm.LEVENSHTEIN
) -> str:
    """
    Generate an optimized regex pattern for fuzzy matching.
    This is an alias for fuzzy_pattern with better naming.
    """
    return fuzzy_pattern(pattern, max_distance, algorithm)
