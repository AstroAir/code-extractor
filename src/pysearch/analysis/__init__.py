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

from .content_addressing import (
    ContentAddress,
    ContentAddressedIndexer,
    GlobalCacheManager,
    IndexingProgressUpdate,
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)
from .dependency_analysis import (
    CircularDependencyDetector,
    DependencyAnalyzer,
    DependencyEdge,
    DependencyGraph,
    DependencyMetrics,
    ImportNode,
)
from .language_detection import (
    detect_language,
    get_language_extensions,
    get_supported_languages,
    is_text_file,
)
from .language_support import (
    CodeChunk,
    LanguageConfig,
    LanguageProcessor,
    LanguageRegistry,
    TreeSitterProcessor,
    language_registry,
)

__all__ = [
    # Language detection
    "detect_language",
    "get_supported_languages",
    "get_language_extensions",
    "is_text_file",
    # Dependency analysis
    "DependencyAnalyzer",
    "DependencyGraph",
    "DependencyEdge",
    "DependencyMetrics",
    "ImportNode",
    "CircularDependencyDetector",
    # Content addressing
    "ContentAddress",
    "ContentAddressedIndexer",
    "GlobalCacheManager",
    "IndexTag",
    "PathAndCacheKey",
    "RefreshIndexResults",
    "IndexingProgressUpdate",
    # Language support
    "LanguageProcessor",
    "TreeSitterProcessor",
    "LanguageRegistry",
    "LanguageConfig",
    "CodeChunk",
    "language_registry",
]
