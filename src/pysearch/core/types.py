"""
Core type definitions for pysearch.

This module provides backward compatibility by re-exporting all types from the
new modular structure. All types are now organized in the types/ package but
can still be imported from this module for compatibility.

For new code, consider importing directly from the specific type modules:
- from pysearch.core.types.basic_types import Query, SearchResult
- from pysearch.core.types.graphrag_types import GraphRAGQuery, CodeEntity

Example:
    Creating a search query:
        >>> from pysearch.core.types import Query, ASTFilters, OutputFormat
        >>>
        >>> # Basic text search
        >>> query = Query(pattern="def main", use_regex=True)
        >>>
        >>> # AST-based search with filters
        >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, filters=filters)

    Working with results:
        >>> from pysearch.core.types import SearchResult, SearchItem
        >>>
        >>> # Process search results
        >>> for item in results.items:
        ...     print(f"Found in {item.file} at lines {item.start_line}-{item.end_line}")
        ...     for line in item.lines:
        ...         print(f"  {line}")
"""

# Import all types from the new modular structure for backward compatibility
from .types import *  # noqa: F403, F401
