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
    """Convert a semantic concept to regex patterns."""
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
    """Expand a query into semantic patterns."""
    words = re.findall(r"\w+", query.lower())
    all_patterns = []

    for word in words:
        patterns = concept_to_patterns(word)
        all_patterns.extend(patterns)

    return list(set(all_patterns))  # Remove duplicates


def semantic_similarity_score(text: str, concept: str) -> float:
    """Calculate semantic similarity score between text and concept."""
    patterns = concept_to_patterns(concept)
    matches = 0
    total_patterns = len(patterns)

    if total_patterns == 0:
        return 0.0

    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    return matches / total_patterns
