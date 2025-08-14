"""
Code analysis and understanding capabilities.

This module provides sophisticated code analysis features:
- Dependency analysis and graph generation
- Language detection and support
- Content addressing and hashing
- GraphRAG functionality for code understanding
- Enhanced language-specific processing

The analysis module enables deep understanding of code structure
and relationships, supporting advanced search and navigation features.
"""

from .language_detection import detect_language, get_supported_languages

__all__ = [
    # Language detection
    "detect_language",
    "get_supported_languages",
]
