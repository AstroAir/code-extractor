# Advanced MCP Features

PySearch's MCP implementation provides several advanced features that go beyond basic code search. This document covers these capabilities in detail.

## Fuzzy Search

Fuzzy search allows you to find approximate matches rather than exact ones, which is useful when you're not sure about the exact spelling or wording.

### Configuration

Fuzzy search can be configured with several parameters:

```python
fuzzy_config = {
    "similarity_threshold": 0.6,  # Minimum similarity score (0.0 to 1.0)
    "max_results": 100,           # Maximum number of fuzzy matches to return
    "algorithm": "ratio",         # Algorithm: ratio, partial_ratio, token_sort_ratio, token_set_ratio
    "case_sensitive": False       # Whether matching is case sensitive
}

result = search_fuzzy(
    pattern="autenticate_user",   # Intentional typo
    paths=["./src"],
    config=fuzzy_config,
    context=3
)
```

### Use Cases

- Finding code with typos in variable or function names
- Searching when you're unsure of exact spelling
- Matching similar concepts with different implementations

## Multi-Pattern Search

Multi-pattern search allows you to combine multiple search patterns with logical operators.

### Logical Operators

1. **AND**: Find results that match all patterns
2. **OR**: Find results that match any pattern (default)
3. **NOT**: Find results that match the first pattern but not subsequent ones

### Example Usage

```python
# Find functions that handle both authentication and logging
query = {
    "patterns": ["auth", "log"],
    "operator": "AND",
    "use_regex": False
}

result = search_multi_pattern(
    query=query,
    paths=["./src"],
    context=3
)

# Find error handling that is NOT related to network issues
query = {
    "patterns": ["except", "network"],
    "operator": "NOT",
    "use_regex": True
}

result = search_multi_pattern(
    query=query,
    paths=["./src"],
    context=3
)
```

## Result Ranking

Results can be ranked by relevance using multiple factors to ensure the most pertinent results appear first.

### Ranking Factors

1. **Pattern Match Quality**: How well the result matches the search pattern
2. **File Importance**: The importance of the file containing the match
3. **Context Relevance**: How relevant the result is to recent searches
4. **Recency**: How recently the file was modified
5. **File Size**: Preference for moderately-sized files
6. **Language Priority**: Priority based on programming language

### Customizing Rankings

```python
# Customize ranking weights
ranking_factors = {
    "pattern_match_quality": 0.5,
    "file_importance": 0.2,
    "context_relevance": 0.1,
    "recency": 0.1,
    "file_size": 0.05,
    "language_priority": 0.05
}

results = search_with_ranking(
    pattern="user authentication",
    paths=["./src"],
    context=3,
    ranking_factors=ranking_factors,
    max_results=20
)
```

## Advanced Filtering

Filter results based on file properties and content metrics.

### Filter Options

```python
search_filter = {
    "min_file_size": 100,           # Only files larger than 100 bytes
    "max_file_size": 100000,        # Only files smaller than 100KB
    "modified_after": "2023-01-01", # Only recently modified files
    "languages": ["python", "javascript"],  # Only specific languages
    "min_complexity": 5.0,          # Only complex code
    "file_extensions": [".py", ".js"]  # Only specific file types
}

result = search_with_filters(
    pattern="function",
    search_filter=search_filter,
    paths=["./src"],
    context=3
)
```

## File Analysis and Statistics

Get detailed information about your codebase files.

### File Statistics

```python
# Get comprehensive file statistics
stats = get_file_statistics(
    paths=["./src"],
    include_analysis=True
)

# Example response structure:
{
    "total_files": 127,
    "total_size": 2456789,
    "languages": {
        "python": 89,
        "javascript": 23,
        "typescript": 15
    },
    "file_extensions": {
        ".py": 89,
        ".js": 23,
        ".ts": 15
    },
    "size_distribution": {
        "small": 45,
        "medium": 67,
        "large": 12,
        "very_large": 3
    },
    "complexity_distribution": {
        "low": 67,
        "medium": 45,
        "high": 12,
        "very_high": 3
    }
}
```

### Individual File Analysis

```python
# Analyze a specific file
analysis = analyze_file_content(
    file_path="./src/user_auth.py",
    include_complexity=True,
    include_quality_metrics=True
)

# Example response structure:
{
    "file_path": "./src/user_auth.py",
    "file_size": 3456,
    "line_count": 127,
    "complexity_score": 12.3,
    "language": "python",
    "functions_count": 8,
    "classes_count": 2,
    "imports_count": 15,
    "comments_ratio": 0.18,
    "code_quality_score": 87.5,
    "last_modified": "2023-06-15T14:30:00"
}
```

## Session Management

Maintain context across multiple searches for more intelligent results.

### Creating and Using Sessions

```python
# Start a new session
result1 = search_text(
    pattern="database",
    session_id="project_analysis_001"
)

# Continue with the same session
result2 = search_ast(
    pattern="class",
    class_name=".*Model$",
    session_id="project_analysis_001"
)

# The server can use context from previous searches to improve relevance
```

### Session Benefits

- Context-aware result ranking
- Tracking of related searches
- Improved relevance for follow-up queries
- Ability to resume analysis workflows

## Progress Reporting

For long-running searches, progress information is provided to keep users informed.

### Progress Information

While the basic API doesn't expose progress directly, the underlying implementation provides:
- Estimated time to completion
- Files processed count
- Results found so far
- Current operation status

## Composition Support

Chain operations together for complex analysis workflows.

### Example Composition

```python
# This is a conceptual example of how composition might work
# in a future version of the API

# 1. Find all authentication-related files
auth_files = search_semantic("authentication", context=0)

# 2. Analyze those files for complexity
complex_auth_files = []
for file in auth_files["items"]:
    analysis = analyze_file_content(file["file"])
    if analysis["complexity_score"] > 10:
        complex_auth_files.append(file)

# 3. Rank the complex files by importance
ranked_files = search_with_ranking(
    pattern="authentication",
    paths=[file["file"] for file in complex_auth_files]
)
```

## Performance Optimization

### Parallel Processing

Enable parallel processing for faster searches across large codebases:

```python
# Configure for parallel processing
config = configure_search(
    parallel=True,
    workers=8  # Adjust based on your CPU cores
)
```

### Caching

The search engine uses intelligent caching to speed up repeated searches:

```python
# Clear caches when needed
clear_caches()
```

## Security Considerations

### File Access Control

- MCP servers only access files within configured paths
- Exclude patterns prevent access to sensitive directories
- All file access is logged for audit purposes

### Safe Regular Expressions

- Regex patterns are validated before execution
- Timeouts prevent malicious patterns from hanging the server
- Resource limits prevent excessive memory usage

## Extending MCP Functionality

### Adding Custom Tools

You can extend the MCP server with custom tools by modifying the server implementation:

```python
# In your MCP server implementation
@mcp.tool
async def custom_code_analysis(
    paths: List[str],
    analysis_type: str
) -> Dict[str, Any]:
    """Custom code analysis tool."""
    # Implementation here
    pass
```

### Custom Ranking Factors

Add your own ranking factors by extending the ranking system:

```python
# Extend the RankingFactor enum
class CustomRankingFactor(Enum):
    BUSINESS_DOMAIN_RELEVANCE = "business_domain_relevance"
    TEAM_OWNERSHIP = "team_ownership"

# Use in ranking
custom_weights = {
    RankingFactor.PATTERN_MATCH_QUALITY: 0.3,
    RankingFactor.FILE_IMPORTANCE: 0.2,
    CustomRankingFactor.BUSINESS_DOMAIN_RELEVANCE: 0.3,
    CustomRankingFactor.TEAM_OWNERSHIP: 0.2
}
```

## Integration Patterns

### IDE Integration

Use MCP servers to provide search capabilities within IDEs:

1. Start the MCP server as a background process
2. Connect IDE plugin to the server via stdio
3. Expose search functionality through IDE commands

### CI/CD Integration

Integrate code search into CI/CD pipelines:

1. Run MCP server in pipeline environment
2. Execute searches to validate code quality
3. Fail builds based on search results

### Documentation Generation

Use semantic search to automatically generate documentation:

1. Search for functions with specific patterns
2. Extract docstrings and comments
3. Generate structured documentation