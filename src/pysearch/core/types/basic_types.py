"""
Basic type definitions for pysearch core functionality.

This module contains the fundamental data types, enumerations, and data classes
used for basic search operations, configuration, and results.

Key Types:
    OutputFormat: Enumeration of supported output formats
    Language: Enumeration of supported programming languages
    ASTFilters: Configuration for AST-based filtering
    FileMetadata: Extended file metadata for advanced filtering
    SearchItem: Individual search result item with context
    SearchResult: Complete search results with statistics
    Query: Search query specification
    MetadataFilters: Advanced metadata-based filters

Example:
    Creating a search query:
        >>> from pysearch.core.types.basic_types import Query, ASTFilters, OutputFormat
        >>>
        >>> # Basic text search
        >>> query = Query(pattern="def main", use_regex=True)
        >>>
        >>> # AST-based search with filters
        >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)

    Working with results:
        >>> from pysearch.core.types.basic_types import SearchResult, SearchItem
        >>>
        >>> # Process search results
        >>> for item in results.items:
        ...     print(f"Found in {item.file} at lines {item.start_line}-{item.end_line}")
        ...     for line in item.lines:
        ...         print(f"  {line}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

class Language(str, Enum):
    """Supported programming languages for syntax-aware processing."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SHELL = "shell"
    POWERSHELL = "powershell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    XML = "xml"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    DOCKERFILE = "dockerfile"
    MAKEFILE = "makefile"
    UNKNOWN = "unknown"


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    HIGHLIGHT = "highlight"


@dataclass(slots=True)
class ASTFilters:
    r"""
    Configuration for AST-based filtering during search operations.

    This class defines regex patterns for filtering search results based on
    Abstract Syntax Tree (AST) elements like function names, class names,
    decorators, and import statements.

    Attributes:
        func_name: Regex pattern to match function names
        class_name: Regex pattern to match class names
        decorator: Regex pattern to match decorator names
        imported: Regex pattern to match imported symbols (including module prefix)

    Example:
        >>> from pysearch.core.types.basic_types import ASTFilters
        >>>
        >>> # Filter for handler functions with cache decorators
        >>> filters = ASTFilters(
        ...     func_name=r".*_handler$",
        ...     decorator=r"(lru_cache|cache)"
        ... )
        >>>
        >>> # Filter for controller classes
        >>> filters = ASTFilters(class_name=r".*Controller$")
        >>>
        >>> # Filter for specific imports
        >>> filters = ASTFilters(imported=r"requests\.(get|post)")
    """

    func_name: str | None = None
    class_name: str | None = None
    decorator: str | None = None
    imported: str | None = None


@dataclass(slots=True)
class FileMetadata:
    """Extended file metadata for advanced filtering."""

    path: Path
    size: int
    mtime: float
    language: Language
    encoding: str = "utf-8"
    line_count: int | None = None
    author: str | None = None  # From git blame or file attributes
    created_date: float | None = None
    modified_date: float | None = None
    content_hash: str | None = None  # SHA1 hash of file content


# match_spans: list of (line_index, (start_col, end_col))
MatchSpan = tuple[int, tuple[int, int]]


@dataclass(slots=True)
class SearchItem:
    """
    Represents a single search result item with context.

    This class contains information about a matched location in a file,
    including the file path, line range, content lines, and precise
    match positions within those lines.

    Attributes:
        file: Path to the file containing the match
        start_line: Starting line number (1-based) of the match context
        end_line: Ending line number (1-based) of the match context
        lines: List of content lines including context around the match
        match_spans: List of precise match positions as (line_index, (start_col, end_col))

    Example:
        >>> item = SearchItem(
        ...     file=Path("example.py"),
        ...     start_line=10,
        ...     end_line=12,
        ...     lines=["    # Context line", "    def main():", "        pass"],
        ...     match_spans=[(1, (4, 12))]  # "def main" at line 1, cols 4-12
        ... )
    """

    file: Path
    start_line: int
    end_line: int
    lines: list[str]
    match_spans: list[MatchSpan] = field(default_factory=list)


@dataclass(slots=True)
class SearchStats:
    """
    Performance and result statistics for a search operation.

    Attributes:
        files_scanned: Total number of files examined during search
        files_matched: Number of files containing at least one match
        items: Total number of search result items found
        elapsed_ms: Total search time in milliseconds
        indexed_files: Number of files processed by the indexer
    """

    files_scanned: int = 0
    files_matched: int = 0
    items: int = 0
    elapsed_ms: float = 0.0
    indexed_files: int = 0


@dataclass(slots=True)
class SearchResult:
    """
    Complete search results including items and performance statistics.

    This is the main result object returned by search operations, containing
    both the matched items and metadata about the search performance.

    Attributes:
        items: List of SearchItem objects representing matches
        stats: SearchStats object with performance metrics

    Example:
        >>> result = SearchResult(
        ...     items=[item1, item2, item3],
        ...     stats=SearchStats(files_scanned=100, files_matched=3, items=3, elapsed_ms=45.2)
        ... )
        >>> print(f"Found {len(result.items)} matches in {result.stats.elapsed_ms}ms")
    """

    items: list[SearchItem] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)


@dataclass(slots=True)
class MetadataFilters:
    """Advanced metadata-based filters for search."""

    min_size: int | None = None  # Minimum file size in bytes
    max_size: int | None = None  # Maximum file size in bytes
    modified_after: float | None = None  # Unix timestamp
    modified_before: float | None = None  # Unix timestamp
    created_after: float | None = None  # Unix timestamp
    created_before: float | None = None  # Unix timestamp
    min_lines: int | None = None  # Minimum line count
    max_lines: int | None = None  # Maximum line count
    author_pattern: str | None = None  # Regex pattern for author name
    encoding_pattern: str | None = None  # Regex pattern for file encoding
    languages: set[Language] | None = None  # Specific languages to include


@dataclass(slots=True)
class Query:
    r"""
    Search query specification with all search parameters.

    This class encapsulates all the parameters needed to execute a search,
    including the pattern, search modes, filters, and output preferences.

    Attributes:
        pattern: The search pattern (text, regex, or semantic query)
        use_regex: Whether to interpret pattern as regular expression
        use_ast: Whether to use AST-based structural matching
        context: Number of context lines to include around matches
        output: Output format for results
        filters: AST-based filters for structural matching
        metadata_filters: File metadata-based filters
        search_docstrings: Whether to search in docstrings
        search_comments: Whether to search in comments
        search_strings: Whether to search in string literals

    Examples:
        Simple text search:
            >>> query = Query(pattern="def main")
            >>> # Searches for literal text "def main"

        Regex search:
            >>> query = Query(pattern=r"def \w+_handler", use_regex=True)
            >>> # Searches for functions ending with "_handler"

        AST search with filters:
            >>> from pysearch.core.types.basic_types import ASTFilters
            >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
            >>> query = Query(pattern="def", use_ast=True, filters=filters)
            >>> # Finds function definitions matching the AST filters

        Semantic search:
            >>> query = Query(pattern="database connection", use_semantic=True)
            >>> # Finds code semantically related to database connections

        Complex query with metadata filters:
            >>> from pysearch.core.types.basic_types import MetadataFilters, Language
            >>> metadata = MetadataFilters(min_lines=50, languages={Language.PYTHON})
            >>> query = Query(
            ...     pattern="class.*Test",
            ...     use_regex=True,
            ...     context=5,
            ...     metadata_filters=metadata,
            ...     search_docstrings=True,
            ...     search_comments=False
            ... )
            >>> # Searches for test classes in substantial Python files
    """

    pattern: str
    use_regex: bool = False
    use_ast: bool = False
    use_semantic: bool = False
    context: int = 2
    output: OutputFormat = OutputFormat.TEXT
    filters: ASTFilters | None = None
    metadata_filters: MetadataFilters | None = None
    search_docstrings: bool = True
    search_comments: bool = True
    search_strings: bool = True

    # Count-only and per-file limits
    count_only: bool = False
    max_per_file: int | None = None

    # Boolean query support
    use_boolean: bool = False


class BooleanOperator(str, Enum):
    """Boolean operators for query composition."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass(slots=True)
class BooleanQuery:
    """Represents a boolean query with logical operators."""

    operator: BooleanOperator | None = None
    left: BooleanQuery | None = None
    right: BooleanQuery | None = None
    term: str | None = None  # For leaf nodes


@dataclass(slots=True)
class CountResult:
    """Result for count-only searches."""

    total_matches: int
    files_matched: int
    stats: SearchStats
