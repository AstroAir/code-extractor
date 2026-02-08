# MCP API Reference

This document provides detailed API reference for all tools available through the PySearch MCP servers.

## Core Search Tools

### search_text

Perform basic text search across files.

**Parameters:**
- `pattern` (string, required): Text pattern to search for
- `paths` (array of strings, optional): List of paths to search (uses configured paths if not provided)
- `context` (integer, optional, default: 3): Number of context lines around matches
- `case_sensitive` (boolean, optional, default: false): Whether search should be case sensitive

**Returns:**
- `SearchResponse`: Object containing search results, statistics, and query information

### search_regex

Perform regex pattern search across files.

**Parameters:**
- `pattern` (string, required): Regex pattern to search for
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches
- `case_sensitive` (boolean, optional, default: false): Whether search should be case sensitive

**Returns:**
- `SearchResponse`: Object containing search results, statistics, and query information

### search_ast

Perform AST-based search with structural filters.

**Parameters:**
- `pattern` (string, required): Base pattern to search for
- `func_name` (string, optional): Regex pattern to match function names
- `class_name` (string, optional): Regex pattern to match class names
- `decorator` (string, optional): Regex pattern to match decorator names
- `imported` (string, optional): Regex pattern to match imported symbols
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches

**Returns:**
- `SearchResponse`: Object containing search results, statistics, and query information

### search_semantic

Perform semantic concept search.

**Parameters:**
- `concept` (string, required): Semantic concept to search for (e.g., "database", "web", "testing")
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches

**Returns:**
- `SearchResponse`: Object containing search results, statistics, and query information

## Advanced Search Tools (Main Server Only)

### search_fuzzy

Perform fuzzy search with configurable similarity thresholds.

**Parameters:**
- `pattern` (string, required): Text pattern to search for with fuzzy matching
- `paths` (array of strings, optional): List of paths to search
- `config` (FuzzySearchConfig, optional): Fuzzy search configuration
- `context` (integer, optional, default: 3): Number of context lines around matches
- `session_id` (string, optional): Session ID for context management

**Returns:**
- `SearchResponse`: Object containing fuzzy matching results

### search_multi_pattern

Perform multi-pattern search with logical operators.

**Parameters:**
- `query` (MultiPatternQuery, required): Multi-pattern query configuration
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches
- `session_id` (string, optional): Session ID for context management

**Returns:**
- `SearchResponse`: Object containing combined results from multiple patterns

### search_with_ranking

Perform search with advanced result ranking.

**Parameters:**
- `pattern` (string, required): Search pattern
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches
- `use_regex` (boolean, optional, default: false): Whether to use regex search
- `ranking_factors` (dict, optional): Custom weights for ranking factors
- `max_results` (integer, optional, default: 50): Maximum number of results to return
- `session_id` (string, optional): Session ID for context management

**Returns:**
- `array of RankedSearchResult`: List of ranked search results

### search_with_filters

Perform search with advanced filtering capabilities.

**Parameters:**
- `pattern` (string, required): Search pattern
- `search_filter` (SearchFilter, required): Advanced filtering options
- `paths` (array of strings, optional): List of paths to search
- `context` (integer, optional, default: 3): Number of context lines around matches
- `use_regex` (boolean, optional, default: false): Whether to use regex search
- `session_id` (string, optional): Session ID for context management

**Returns:**
- `SearchResponse`: Object containing filtered results

## Configuration Tools

### configure_search

Update search configuration.

**Parameters:**
- `paths` (array of strings, optional): List of paths to search
- `include_patterns` (array of strings, optional): File patterns to include
- `exclude_patterns` (array of strings, optional): File patterns to exclude
- `context` (integer, optional): Number of context lines
- `parallel` (boolean, optional): Whether to use parallel processing
- `workers` (integer, optional): Number of worker threads
- `languages` (array of strings, optional): List of languages to filter by

**Returns:**
- `ConfigResponse`: Object containing updated configuration

### get_search_config

Get current search configuration.

**Returns:**
- `ConfigResponse`: Object containing current configuration

## Utility Tools

### get_supported_languages

Get list of supported programming languages.

**Returns:**
- `array of strings`: List of supported language names

### clear_caches

Clear search engine caches.

**Returns:**
- `dict`: Status message

### get_search_history

Get recent search history.

**Parameters:**
- `limit` (integer, optional, default: 10): Maximum number of history entries to return

**Returns:**
- `array of dicts`: List of recent search operations

### get_file_statistics

Get comprehensive statistics about files in the search paths. (Main server only)

**Parameters:**
- `paths` (array of strings, optional): List of paths to analyze
- `include_analysis` (boolean, optional, default: false): Whether to include detailed file analysis

**Returns:**
- `dict`: Dictionary with file statistics and analysis

## Data Structures

### SearchResponse

```python
{
    "items": list[dict],          # List of search results
    "stats": dict,                # Search statistics
    "query_info": dict,           # Information about the query
    "total_matches": int,         # Total number of matches
    "execution_time_ms": float    # Execution time in milliseconds
}
```

### ConfigResponse

```python
{
    "paths": list[str],              # List of search paths
    "include_patterns": list[str],   # File patterns to include
    "exclude_patterns": list[str],   # File patterns to exclude
    "context_lines": int,            # Number of context lines
    "parallel": bool,                # Whether parallel processing is enabled
    "workers": int,                  # Number of worker threads
    "languages": list[str]           # List of languages to filter by
}
```

### FuzzySearchConfig

```python
{
    "similarity_threshold": float,  # Minimum similarity score (0.0 to 1.0)
    "max_results": int,             # Maximum number of fuzzy matches to return
    "algorithm": str,               # Fuzzy matching algorithm
    "case_sensitive": bool          # Whether matching is case sensitive
}
```

### MultiPatternQuery

```python
{
    "patterns": list[str],      # List of patterns to search for
    "operator": str,            # Logical operator (AND, OR, NOT)
    "use_regex": bool,          # Whether to use regex search
    "use_fuzzy": bool,          # Whether to use fuzzy search
    "fuzzy_config": dict        # Fuzzy search configuration
}
```

### SearchFilter

```python
{
    "min_file_size": int,         # Minimum file size in bytes
    "max_file_size": int,         # Maximum file size in bytes
    "modified_after": datetime,   # Only files modified after this date
    "modified_before": datetime,  # Only files modified before this date
    "authors": list[str],         # Only files by these authors
    "languages": list[str],       # Only files in these languages
    "min_complexity": float,      # Minimum complexity score
    "max_complexity": float,      # Maximum complexity score
    "file_extensions": list[str], # Only files with these extensions
    "exclude_patterns": list[str] # Exclude files matching these patterns
}
```

### RankedSearchResult

```python
{
    "item": dict,              # The search result item
    "relevance_score": float,  # Overall relevance score
    "ranking_factors": dict    # Individual ranking factor scores
}
```