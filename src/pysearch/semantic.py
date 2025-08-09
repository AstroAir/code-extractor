"""
Lightweight semantic search module for pysearch.

This module provides semantic search capabilities without requiring external models
or heavy dependencies. It uses pattern-based concept recognition to identify
semantically related code even when exact text matches are not present.

Key Features:
    - Lightweight semantic matching using regex patterns
    - Predefined concept categories for common programming domains
    - Fast concept recognition with O(1) pattern matching
    - No external dependencies (no transformers, embeddings, etc.)
    - Optimized for code-specific semantic relationships
    - Extensible pattern system for custom concepts

Concept Categories:
    - Database: Connection handling, queries, transactions
    - Web: HTTP operations, APIs, web frameworks
    - Testing: Test functions, assertions, mocking
    - Async: Asynchronous programming patterns
    - Logging: Logging and debugging functionality
    - File: File operations and path handling
    - Error: Error handling and exceptions
    - Config: Configuration and settings
    - Security: Authentication, encryption, permissions

Functions:
    semantic_similarity_score: Calculate semantic similarity between text and query
    extract_semantic_features: Extract semantic features from code content
    get_concept_patterns: Get regex patterns for specific concept categories

Example:
    Basic semantic similarity:
        >>> from pysearch.semantic import semantic_similarity_score
        >>>
        >>> code = '''
        ... def connect_database():
        ...     conn = sqlite3.connect('app.db')
        ...     return conn
        ... '''
        >>>
        >>> score = semantic_similarity_score(code, "database connection")
        >>> print(f"Semantic similarity: {score:.2f}")

    Concept pattern matching:
        >>> from pysearch.semantic import get_concept_patterns
        >>>
        >>> db_patterns = get_concept_patterns("database")
        >>> print("Database patterns:", db_patterns)
"""

from __future__ import annotations

import re

# Semantic concept mappings for Python code
CONCEPT_PATTERNS = {
    "database": [
        r"\b(db|database|conn|connection|cursor|execute|query|select|insert|update|delete)\b",
        r"\b(sql|mysql|postgres|sqlite|mongodb|redis)\b",
        r"\b(session|transaction|commit|rollback)\b",
    ],
    "web": [
        r"\b(http|https|request|response|get|post|put|delete|patch)\b",
        r"\b(flask|django|fastapi|tornado|bottle)\b",
        r"\b(route|endpoint|api|rest|json|xml)\b",
        r"\b(middleware|cors|auth|session|cookie)\b",
    ],
    "testing": [
        r"\b(test|assert|mock|patch|fixture|setUp|tearDown)\b",
        r"\b(pytest|unittest|nose|doctest)\b",
        r"\b(should|expect|verify|check|validate)\b",
    ],
    "async": [
        r"\b(async|await|asyncio|coroutine|future|task)\b",
        r"\b(async def|await\s+\w+)\b",
        r"\b(gather|create_task|run|get_event_loop)\b",
    ],
    "logging": [
        r"\b(log|logger|logging|debug|info|warn|warning|error|critical)\b",
        r"\b(getLogger|basicConfig|handler|formatter)\b",
    ],
    "file": [
        r"\b(file|open|read|write|close|path|os\.path|pathlib)\b",
        r"\b(with\s+open|\w+\.read|\w+\.write)\b",
        r"\b(exists|isfile|isdir|mkdir|rmdir)\b",
    ],
    "error": [
        r"\b(error|exception|try|catch|except|finally|raise|throw)\b",
        r"\b(ValueError|TypeError|KeyError|IndexError|AttributeError)\b",
        r"\b(handle|rescue|recover|fail|failure)\b",
    ],
    "config": [
        r"\b(config|configuration|settings|options|params|parameters)\b",
        r"\b(env|environment|ENV|dotenv)\b",
        r"\b(yaml|json|toml|ini|cfg)\b",
    ],
    "security": [
        r"\b(auth|authentication|authorization|login|password|token|jwt)\b",
        r"\b(hash|encrypt|decrypt|cipher|ssl|tls|https)\b",
        r"\b(permission|role|user|access|grant|deny)\b",
    ],
    "data": [
        r"\b(data|dataset|dataframe|array|list|dict|json|csv|xml)\b",
        r"\b(pandas|numpy|serialize|deserialize|parse|format)\b",
        r"\b(transform|process|filter|map|reduce)\b",
    ],
}


def concept_to_patterns(concept: str) -> list[str]:
    """
    Convert a semantic concept to regex patterns for matching.

    This function maps high-level concepts to specific regex patterns that can
    identify related code constructs. It supports both predefined concept categories
    and automatic pattern generation for unknown concepts.

    Args:
        concept: Semantic concept to convert (e.g., "database", "web", "testing")

    Returns:
        List of regex patterns that match the concept

    Example:
        >>> patterns = concept_to_patterns("database")
        >>> print(patterns)
        ['\\b(db|database|conn|connection|cursor|execute|query)\\b', ...]

        >>> patterns = concept_to_patterns("custom_function")
        >>> print(patterns)
        ['\\bcustom_function\\b', '\\bcustom_functions\\b', ...]

    Note:
        For unknown concepts, generates patterns including:
        - Exact match with word boundaries
        - Common variations (plural, past tense, etc.)
        - Case variations (snake_case, kebab-case)
    """
    concept_lower = concept.lower()

    # Direct match
    if concept_lower in CONCEPT_PATTERNS:
        return CONCEPT_PATTERNS[concept_lower]

    # Partial matches
    patterns = []
    for key, pattern_list in CONCEPT_PATTERNS.items():
        if concept_lower in key or key in concept_lower:
            patterns.extend(pattern_list)

    # If no semantic match, treat as literal pattern
    if not patterns:
        # Try to infer patterns from the concept
        patterns = [rf"\b{re.escape(concept)}\b"]

        # Add common variations
        variations = [
            concept + "s",  # plural
            concept + "ed",  # past tense
            concept + "ing",  # present participle
            concept.replace("_", ""),  # snake_case -> camelCase
            concept.replace("-", ""),  # kebab-case -> camelCase
        ]

        for var in variations:
            if var != concept:
                patterns.append(rf"\b{re.escape(var)}\b")

    return patterns


def expand_semantic_query(query: str) -> list[str]:
    """
    Expand a query string into comprehensive semantic patterns.

    This function breaks down a multi-word query into individual concepts
    and generates regex patterns for each, creating a comprehensive set
    of patterns for semantic matching.

    Args:
        query: Query string containing one or more concepts

    Returns:
        List of unique regex patterns covering all concepts in the query

    Example:
        >>> patterns = expand_semantic_query("database connection error")
        >>> print(f"Generated {len(patterns)} patterns")
        >>> # Includes patterns for database, connection, and error concepts
    """
    words = re.findall(r"\w+", query.lower())
    all_patterns = []

    for word in words:
        patterns = concept_to_patterns(word)
        all_patterns.extend(patterns)

    return list(set(all_patterns))  # Remove duplicates


def semantic_similarity_score(text: str, concept: str) -> float:
    """
    Calculate semantic similarity score between text and concept.

    This function computes a similarity score by checking how many of the
    concept's associated patterns match in the given text. The score ranges
    from 0.0 (no matches) to 1.0 (all patterns match).

    Args:
        text: Text content to analyze for semantic similarity
        concept: Concept to match against (e.g., "database", "web API")

    Returns:
        Similarity score between 0.0 and 1.0

    Example:
        >>> code = '''
        ... def connect_database():
        ...     conn = sqlite3.connect('app.db')
        ...     cursor = conn.cursor()
        ...     return conn
        ... '''
        >>> score = semantic_similarity_score(code, "database")
        >>> print(f"Database similarity: {score:.2f}")

    Note:
        The score is calculated as: (matching_patterns / total_patterns)
        Higher scores indicate stronger semantic relationship.
    """
    patterns = concept_to_patterns(concept)
    matches = 0
    total_patterns = len(patterns)

    if total_patterns == 0:
        return 0.0

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    return matches / total_patterns
