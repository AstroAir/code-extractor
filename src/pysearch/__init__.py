"""
pysearch: High-performance, context-aware search engine for Python codebases.

This package provides a comprehensive search solution for Python projects, supporting
text/regex/AST/semantic search with both CLI and programmable API interfaces. It's
designed for engineering-grade retrieval in large multi-module projects with
performance optimizations and intelligent caching.

Key Features:
    - **Multiple Search Modes**: Text, regex, AST structural, and semantic search
    - **Code Block Matching**: Functions, classes, decorators, imports, strings/comments
    - **Context-Aware Output**: Configurable context lines around matches
    - **Project-Wide Indexing**: Efficient caching for large codebases with incremental updates
    - **Advanced Filtering**: Include/exclude patterns, AST filters, metadata filters
    - **Performance Optimized**: Parallel processing, directory pruning, smart caching
    - **Rich Output Formats**: Plain text, JSON, highlighted console output
    - **Result Ranking**: Configurable scoring and similarity clustering
    - **Dual Interfaces**: Both CLI and programmatic API access
    - **Error Handling**: Comprehensive error collection and reporting
    - **Extensible**: Plugin-ready architecture for custom matchers and filters

Main Classes:
    PySearch: Main search engine class that orchestrates all operations
    SearchConfig: Configuration object for search parameters and performance tuning
    Query: Search query specification with filters and options
    SearchResult: Search results container with items and performance statistics
    ASTFilters: AST-based filtering for structural code matching
    MetadataFilters: File metadata-based filtering (size, date, language, etc.)

Core Modules:
    api: Main search engine and high-level interfaces
    config: Configuration management and validation
    types: Core data types and enumerations
    matchers: Pattern matching implementations (text, regex, AST, semantic)
    indexer: File indexing and caching system
    scorer: Result ranking and similarity algorithms
    formatter: Output formatting and highlighting
    cli: Command-line interface implementation

Example Usage:
    Basic API usage:
        >>> from pysearch import PySearch, SearchConfig
        >>> config = SearchConfig(paths=["."], include=["**/*.py"])
        >>> engine = PySearch(config)
        >>> results = engine.search(pattern="def main", regex=True)
        >>> for item in results.items:
        ...     print(f"{item.file}: {len(item.lines)} lines")

    Advanced query with filters:
        >>> from pysearch.types import Query, ASTFilters
        >>> filters = ASTFilters(func_name=".*handler", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, filters=filters)
        >>> results = engine.run(query)

    CLI usage:
        $ pysearch find --pattern "def main" --path . --regex --context 2
        $ pysearch find --pattern "class.*Test" --regex --ast --filter-decorator "pytest.*"

Performance Features:
    - Parallel file processing with configurable worker threads
    - Incremental indexing based on file modification times
    - Optional strict hash checking for exact change detection
    - Directory pruning to skip excluded paths during traversal
    - In-memory caching with TTL for frequently accessed files
    - Smart result deduplication and similarity clustering

Use Cases:
    - Code refactoring and analysis
    - Finding deprecated API usage
    - Security auditing and vulnerability scanning
    - Technical debt identification
    - Dependency analysis and import tracking
    - Code quality assessment
    - Documentation and comment analysis

For detailed documentation, examples, and advanced usage patterns, see:
    - docs/usage.md for getting started
    - docs/configuration.md for configuration options
    - docs/architecture.md for implementation details
    - examples/ directory for practical examples
    - Project repository: https://github.com/your-org/pysearch
"""

from .api import PySearch
from .config import SearchConfig
from .error_handling import (
    EncodingError,
    FileAccessError,
    ParsingError,
    PermissionError,
    SearchError,
)
from .history import SearchHistory
from .language_detection import detect_language, get_supported_languages
from .logging_config import configure_logging, disable_logging, enable_debug_logging, get_logger
from .metadata_filters import create_metadata_filters
from .types import (
    ASTFilters,
    FileMetadata,
    Language,
    MatchSpan,
    MetadataFilters,
    OutputFormat,
    Query,
    SearchItem,
    SearchResult,
    SearchStats,
)

# Package metadata
__version__ = "0.1.0"
__author__ = "Kilo Code"
__email__ = "contact@kilocode.dev"
__license__ = "MIT"
__description__ = "High-performance, context-aware search engine for Python codebases"
__url__ = "https://github.com/your-org/pysearch"

# Public API
__all__ = [
    # Main classes
    "PySearch",
    "SearchConfig",
    "SearchHistory",
    # Data types
    "OutputFormat",
    "SearchItem",
    "SearchStats",
    "SearchResult",
    "Query",
    "ASTFilters",
    "MatchSpan",
    "Language",
    "FileMetadata",
    "MetadataFilters",
    # Utility functions
    "detect_language",
    "get_supported_languages",
    "create_metadata_filters",
    # Logging and configuration
    "configure_logging",
    "get_logger",
    "enable_debug_logging",
    "disable_logging",
    # Exception classes
    "SearchError",
    "FileAccessError",
    "PermissionError",
    "EncodingError",
    "ParsingError",
    # Package metadata
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
]
