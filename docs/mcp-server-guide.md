# PySearch Enhanced MCP Server - Complete Guide

## Overview

The PySearch Enhanced MCP Server is a comprehensive, production-ready Model Context Protocol server that exposes advanced code search capabilities. It provides sophisticated search tools, analytics, session management, and context-aware features for AI-powered code exploration and analysis.

## Features

### Core Search Capabilities
- **Text Search**: Fast literal text search across codebases
- **Regex Search**: Powerful pattern matching with regular expressions
- **AST Search**: Structure-aware search using Abstract Syntax Trees
- **Semantic Search**: Concept-based search using semantic analysis
- **Fuzzy Search**: Approximate string matching with similarity thresholds
- **Multi-Pattern Search**: Combine multiple patterns with logical operators (AND/OR/NOT)

### Advanced Features
- **Result Ranking**: AI-powered ranking based on multiple relevance factors
- **Advanced Filtering**: Filter by file size, date, language, complexity, author
- **File Analysis**: Comprehensive code analysis including complexity and quality metrics
- **Progress Reporting**: Real-time progress updates for long-running operations
- **Session Management**: Context-aware sessions with user personalization
- **Resource Management**: Intelligent caching with LRU eviction and analytics

### Security & Performance
- **Input Validation**: Comprehensive validation with security checks
- **Rate Limiting**: Abuse prevention with configurable limits
- **Error Handling**: Graceful degradation with detailed error reporting
- **Performance Monitoring**: Real-time metrics and health checks

## Installation

### Prerequisites
- Python 3.10+
- FastMCP library
- PySearch dependencies

### Install Dependencies

```bash
pip install fastmcp rapidfuzz fuzzywuzzy python-levenshtein
```

### Optional Dependencies

For enhanced features:
```bash
# For vector search and GraphRAG
pip install sentence-transformers scikit-learn qdrant-client

# For system monitoring
pip install psutil

# For development
pip install pytest pytest-asyncio pytest-benchmark
```

## Quick Start

### 1. Basic Server Setup

```python
from mcp.servers.enhanced_fastmcp_server import EnhancedPySearchMCPServer

# Create and start the server
enhanced_server = EnhancedPySearchMCPServer()
mcp = enhanced_server.create_fastmcp_server()

if mcp:
    print("Starting Enhanced PySearch FastMCP Server...")
    mcp.run()
```

### 2. Command Line Usage

```bash
# Start the server
python -m mcp.servers.enhanced_fastmcp_server

# Or use the direct script
python mcp/servers/enhanced_fastmcp_server.py
```

### 3. MCP Client Configuration

Configure your MCP client (e.g., Claude Desktop) to use the server:

```json
{
  "mcpServers": {
    "pysearch": {
      "command": "python",
      "args": ["/path/to/enhanced_fastmcp_server.py"],
      "cwd": "/path/to/your/codebase"
    }
  }
}
```

## Tool Reference

### Core Search Tools

#### `search_text`
Perform basic text search across files.

**Parameters:**
- `pattern` (string, required): Text pattern to search for
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines (default: 3)
- `case_sensitive` (boolean, optional): Case sensitivity (default: false)
- `session_id` (string, optional): Session ID for context management

**Example:**
```json
{
  "pattern": "async def process_request",
  "paths": ["./src", "./api"],
  "context": 5
}
```

#### `search_regex`
Perform regex pattern search across files.

**Parameters:**
- `pattern` (string, required): Regular expression pattern
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `case_sensitive` (boolean, optional): Case sensitivity
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "pattern": "class \\w+Controller",
  "paths": ["./controllers"],
  "context": 3
}
```

#### `search_ast`
Perform AST-based structural search with filters.

**Parameters:**
- `pattern` (string, required): Base pattern to search for
- `func_name` (string, optional): Regex pattern for function names
- `class_name` (string, optional): Regex pattern for class names
- `decorator` (string, optional): Regex pattern for decorators
- `imported` (string, optional): Regex pattern for imports
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "pattern": "def",
  "func_name": ".*_handler$",
  "decorator": ".*route.*",
  "paths": ["./api"]
}
```

#### `search_semantic`
Perform semantic concept search.

**Parameters:**
- `concept` (string, required): Semantic concept (e.g., "database", "authentication")
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "concept": "authentication",
  "paths": ["./src"],
  "context": 5
}
```

### Advanced Search Tools

#### `search_fuzzy`
Perform fuzzy search with configurable similarity.

**Parameters:**
- `pattern` (string, required): Pattern for fuzzy matching
- `similarity_threshold` (float, optional): Minimum similarity (0.0-1.0, default: 0.6)
- `max_results` (integer, optional): Maximum results (default: 100)
- `algorithm` (string, optional): Fuzzy algorithm ("ratio", "partial_ratio", etc.)
- `case_sensitive` (boolean, optional): Case sensitivity
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "pattern": "autentication",
  "similarity_threshold": 0.8,
  "algorithm": "token_sort_ratio"
}
```

#### `search_multi_pattern`
Search multiple patterns with logical operators.

**Parameters:**
- `patterns` (array, required): List of patterns to search
- `operator` (string, optional): Logical operator ("AND", "OR", "NOT", default: "OR")
- `use_regex` (boolean, optional): Whether patterns are regex
- `use_fuzzy` (boolean, optional): Whether to use fuzzy matching
- `fuzzy_threshold` (float, optional): Fuzzy similarity threshold
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "patterns": ["login", "authentication", "auth"],
  "operator": "OR",
  "use_fuzzy": true,
  "fuzzy_threshold": 0.7
}
```

#### `search_with_ranking`
Search with advanced result ranking.

**Parameters:**
- `pattern` (string, required): Search pattern
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `use_regex` (boolean, optional): Whether to use regex
- `max_results` (integer, optional): Maximum results (default: 50)
- `pattern_weight` (float, optional): Weight for pattern match quality (default: 0.4)
- `importance_weight` (float, optional): Weight for file importance (default: 0.2)
- `relevance_weight` (float, optional): Weight for context relevance (default: 0.2)
- `recency_weight` (float, optional): Weight for file recency (default: 0.1)
- `size_weight` (float, optional): Weight for file size (default: 0.05)
- `language_weight` (float, optional): Weight for language priority (default: 0.05)
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "pattern": "error handling",
  "max_results": 20,
  "importance_weight": 0.3,
  "recency_weight": 0.2
}
```

#### `search_with_filters`
Search with comprehensive filtering.

**Parameters:**
- `pattern` (string, required): Search pattern
- `min_file_size` (integer, optional): Minimum file size in bytes
- `max_file_size` (integer, optional): Maximum file size in bytes
- `modified_after` (string, optional): Modified after date (ISO format)
- `modified_before` (string, optional): Modified before date (ISO format)
- `authors` (array, optional): List of authors to filter by
- `languages` (array, optional): List of programming languages
- `file_extensions` (array, optional): List of file extensions
- `exclude_patterns` (array, optional): Regex patterns to exclude
- `min_complexity` (float, optional): Minimum complexity score
- `max_complexity` (float, optional): Maximum complexity score
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `use_regex` (boolean, optional): Whether to use regex
- `session_id` (string, optional): Session ID

**Example:**
```json
{
  "pattern": "TODO",
  "languages": ["python", "javascript"],
  "modified_after": "2024-01-01",
  "max_complexity": 10.0,
  "file_extensions": [".py", ".js"]
}
```

### Analysis Tools

#### `analyze_file_content`
Analyze file content for comprehensive metrics.

**Parameters:**
- `file_path` (string, required): Path to file to analyze
- `include_complexity` (boolean, optional): Whether to calculate complexity (default: true)
- `include_quality_metrics` (boolean, optional): Whether to calculate quality metrics (default: true)

**Example:**
```json
{
  "file_path": "./src/main.py",
  "include_complexity": true,
  "include_quality_metrics": true
}
```

#### `get_file_statistics`
Get comprehensive file statistics.

**Parameters:**
- `paths` (array, optional): List of paths to analyze
- `include_analysis` (boolean, optional): Whether to include detailed analysis

**Example:**
```json
{
  "paths": ["./src", "./tests"],
  "include_analysis": true
}
```

### Configuration Tools

#### `configure_search`
Update search configuration settings.

**Parameters:**
- `paths` (array, optional): List of paths to search
- `include_patterns` (array, optional): File patterns to include
- `exclude_patterns` (array, optional): File patterns to exclude
- `context` (integer, optional): Number of context lines
- `parallel` (boolean, optional): Whether to use parallel processing
- `workers` (integer, optional): Number of worker threads
- `languages` (array, optional): List of languages to filter by

#### `get_search_config`
Get current search configuration.

#### `get_supported_languages`
Get list of supported programming languages.

#### `clear_caches`
Clear search engine caches.

### Session Management Tools

#### `create_search_session`
Create a new context-aware search session.

**Parameters:**
- `context` (object, optional): Initial context information

**Example:**
```json
{
  "context": {
    "project_type": "web_application",
    "user_preferences": {
      "preferred_languages": ["python", "javascript"]
    }
  }
}
```

### Progress Tools

#### `search_with_progress`
Perform search with real-time progress reporting.

**Parameters:**
- `pattern` (string, required): Search pattern
- `paths` (array, optional): List of paths to search
- `context` (integer, optional): Number of context lines
- `use_regex` (boolean, optional): Whether to use regex

#### `batch_analyze_files`
Analyze multiple files with progress tracking.

**Parameters:**
- `file_paths` (array, required): List of file paths to analyze

#### `get_active_operations`
Get list of active long-running operations.

#### `cancel_operation`
Cancel a running operation.

**Parameters:**
- `operation_id` (string, required): ID of operation to cancel

### Utility Tools

#### `get_search_history`
Get recent search history.

**Parameters:**
- `limit` (integer, optional): Maximum entries to return (default: 10)

## Resource Reference

The server exposes several MCP resources for accessing cached data and analytics:

### `pysearch://config/current`
Current search configuration including paths, patterns, and settings.

### `pysearch://history/searches`
Complete search history with queries, results, and metadata.

### `pysearch://sessions/active`
Currently active search sessions with context and state.

### `pysearch://cache/file-analysis`
Cached file analysis results including complexity and quality metrics.

### `pysearch://stats/overview`
Comprehensive statistics about searches, files, and performance.

### `pysearch://index/languages`
Index of supported languages and file type mappings.

### `pysearch://config/ranking-weights`
Current ranking factor weights for search result scoring.

### `pysearch://performance/metrics`
Performance metrics and timing data for search operations.

### `pysearch://cache/analytics`
Comprehensive cache performance analytics and statistics.

### `pysearch://analytics/usage`
Server usage analytics including tool usage patterns and trends.

### `pysearch://health/status`
Overall server health status and diagnostics.

## Advanced Usage

### Session-Aware Searches

Sessions provide context continuity across multiple searches:

```python
# Create a session with context
session = await create_search_session({
    "context": {
        "project_focus": "authentication",
        "current_task": "debugging login issues"
    }
})

# Use session in subsequent searches
result1 = await search_text("login failed", session_id=session["session_id"])
result2 = await search_semantic("error handling", session_id=session["session_id"])

# Get contextual recommendations
recommendations = await get_contextual_recommendations(session["session_id"])
```

### Progressive Analysis

For large codebases, use progressive analysis with progress tracking:

```python
# Start file analysis with progress
files_to_analyze = ["./src/file1.py", "./src/file2.py", ...]
async for progress in batch_analyze_files(files_to_analyze):
    print(f"Progress: {progress['progress']:.1%} - {progress['current_step']}")
    
    if progress['status'] == 'completed':
        print("Analysis complete!")
        break
```

### Advanced Filtering and Ranking

Combine filtering with ranking for precise results:

```python
# Search for recent, complex authentication code
result = await search_with_filters(
    pattern="authentication",
    modified_after="2024-01-01",
    min_complexity=5.0,
    languages=["python", "javascript"],
    ranking_factors={
        "recency_weight": 0.4,
        "importance_weight": 0.3,
        "complexity_weight": 0.3
    }
)
```

### Multi-Pattern Logical Searches

Use logical operators for complex queries:

```python
# Find error handling that's NOT related to network issues
result = await search_multi_pattern(
    patterns=["try", "except", "network"],
    operator="NOT",  # First pattern without subsequent ones
    use_regex=True
)

# Find functions that handle both authentication AND logging
result = await search_multi_pattern(
    patterns=["auth", "log"],
    operator="AND",
    use_fuzzy=True
)
```

## Performance Tuning

### Configuration Optimization

```python
# Configure for performance
await configure_search(
    parallel=True,
    workers=8,  # Adjust based on CPU cores
    exclude_patterns=[
        "**/node_modules/**",
        "**/.git/**",
        "**/venv/**",
        "**/__pycache__/**",
        "**/build/**",
        "**/dist/**"
    ]
)
```

### Resource Management

Monitor and optimize resource usage:

```python
# Check performance metrics
metrics = await get_resource("pysearch://performance/metrics")

# Monitor cache performance
cache_analytics = await get_resource("pysearch://cache/analytics")

# Check health status
health = await get_resource("pysearch://health/status")
```

### Rate Limiting

The server includes built-in rate limiting to prevent abuse. Default limits:
- 100 requests per minute per identifier
- Configurable through the validation system

## Error Handling

The server provides comprehensive error handling with different error types:

### Validation Errors
- Input validation failures
- Type checking errors
- Format validation issues

### Security Errors
- Path traversal attempts
- Dangerous pattern detection
- System path access attempts

### Performance Errors
- Resource limit exceeded
- Timeout conditions
- Memory constraints

### Example Error Response
```json
{
  "error": {
    "type": "ValidationError",
    "message": "Pattern too long (max 10,000 characters)",
    "field": "pattern",
    "severity": "medium"
  }
}
```

## Security Features

### Input Sanitization
- Automatic sanitization of all inputs
- Path traversal prevention
- Dangerous pattern detection

### Access Control
- Restricted access to system paths
- File system boundary enforcement
- User session isolation

### Rate Limiting
- Request throttling
- Abuse prevention
- Configurable limits

## Monitoring and Analytics

### Health Checks
The server provides comprehensive health monitoring:

```bash
# Check server health
curl "pysearch://health/status"
```

### Usage Analytics
Track server usage patterns:

```bash
# Get usage analytics
curl "pysearch://analytics/usage"
```

### Performance Metrics
Monitor performance:

```bash
# Get performance metrics
curl "pysearch://performance/metrics"
```

## Troubleshooting

### Common Issues

1. **Server Not Starting**
   - Check FastMCP installation
   - Verify Python version (3.10+)
   - Check dependencies

2. **Search Performance Issues**
   - Enable parallel processing
   - Increase worker count
   - Optimize exclude patterns
   - Use more specific search patterns

3. **Memory Usage**
   - Clear caches regularly
   - Reduce context lines for large results
   - Limit max results

4. **Session Issues**
   - Check session ID validity
   - Verify session hasn't expired
   - Clear old sessions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Diagnostics

Use health resources to diagnose issues:

```python
health = await get_resource("pysearch://health/status")
if health["overall_health"]["status"] != "healthy":
    print(f"Health issues detected: {health['diagnostics']}")
```

## API Limits

| Resource | Limit | Description |
|----------|-------|-------------|
| Pattern Length | 10,000 chars | Maximum search pattern length |
| Paths | 100 | Maximum number of search paths |
| Context Lines | 1,000 | Maximum context lines |
| Max Results | 10,000 | Maximum search results |
| File Size | 10 MB | Maximum file size for analysis |
| Session TTL | 24 hours | Session expiration time |
| Cache Size | 100 entries | Maximum cached resources |

## Examples

### Basic Code Search

```python
# Search for function definitions
result = await search_regex(
    pattern=r"def\s+\w+",
    paths=["./src"],
    context=3
)

# Find all TODO comments
todos = await search_text(
    pattern="TODO",
    paths=["./src", "./tests"],
    case_sensitive=False
)
```

### Semantic Analysis

```python
# Find database-related code
db_code = await search_semantic(
    concept="database",
    paths=["./src"]
)

# Find error handling patterns
error_handling = await search_semantic(
    concept="error handling",
    paths=["./src"]
)
```

### File Analysis

```python
# Analyze specific file
analysis = await analyze_file_content(
    file_path="./src/main.py",
    include_complexity=True,
    include_quality_metrics=True
)

# Get codebase statistics
stats = await get_file_statistics(
    paths=["./src"],
    include_analysis=True
)
```

### Advanced Search Workflows

```python
# Progressive search refinement
session = await create_search_session()

# Initial broad search
broad_results = await search_semantic(
    concept="authentication",
    session_id=session["session_id"]
)

# Refine with specific patterns
refined_results = await search_with_ranking(
    pattern="login.*error",
    use_regex=True,
    session_id=session["session_id"],
    max_results=20
)

# Get recommendations based on context
recommendations = await get_contextual_recommendations(
    session["session_id"]
)
```

## Integration Examples

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "pysearch": {
      "command": "python",
      "args": [
        "D:\\Project\\code-extractor\\mcp\\servers\\enhanced_fastmcp_server.py"
      ],
      "cwd": "D:\\Project\\my-codebase",
      "env": {
        "PYTHONPATH": "D:\\Project\\code-extractor"
      }
    }
  }
}
```

### Custom Client Integration

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

async def search_codebase():
    async with stdio_client(
        command="python",
        args=["enhanced_fastmcp_server.py"]
    ) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize client
            await session.initialize()
            
            # Use search tools
            result = await session.call_tool(
                "search_text",
                arguments={"pattern": "async def", "context": 5}
            )
            
            return result
```

## Best Practices

### Search Optimization
1. Use specific patterns to reduce result sets
2. Leverage AST search for structural queries
3. Use semantic search for concept exploration
4. Combine filtering with ranking for precision

### Session Management
1. Create sessions for related search workflows
2. Use context to improve result relevance
3. Clean up old sessions regularly
4. Track user preferences for personalization

### Performance
1. Enable parallel processing for large codebases
2. Use appropriate exclude patterns
3. Monitor cache performance
4. Set reasonable limits for context and results

### Security
1. Validate all inputs before processing
2. Use rate limiting to prevent abuse
3. Monitor for suspicious patterns
4. Restrict access to sensitive paths

## Contributing

To contribute to the PySearch MCP server:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Follow the existing code style
5. Submit a pull request

### Development Setup

```bash
git clone https://github.com/your-org/code-extractor.git
cd code-extractor
pip install -e .[dev]
pytest tests/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the API documentation
- Monitor server health and logs
