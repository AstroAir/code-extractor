# Using PySearch with MCP

This tutorial will guide you through setting up and using PySearch with the Model Context Protocol (MCP) to enable powerful code search capabilities in AI assistants like Claude Desktop.

## Prerequisites

Before you begin, ensure you have:

1. Python 3.8 or higher installed
2. PySearch installed (`pip install -e .`)
3. Required dependencies for MCP functionality:
   ```bash
   pip install fastmcp rapidfuzz
   ```

## Setting Up the MCP Server

### 1. Choose Your Server

PySearch provides three MCP server implementations:

- **Main Server**: Full-featured with advanced capabilities
- **Basic Server**: Core functionality only
- **FastMCP Server**: Optimized implementation

For this tutorial, we'll use the Main Server which provides the most comprehensive feature set.

### 2. Run the Server

Start the MCP server from the project root:

```bash
python mcp/servers/mcp_server.py
```

The server will start and listen for MCP connections.

## Integrating with Claude Desktop

### 1. Configure Claude Desktop

Add the PySearch MCP server to your Claude Desktop configuration:

1. Open Claude Desktop settings
2. Navigate to the "Tools" section
3. Add a new tool with the following configuration:
   - Name: PySearch
   - Command: `python /path/to/pysearch/mcp/servers/mcp_server.py`
   - Working Directory: `/path/to/pysearch`

### 2. Test the Integration

Once configured, Claude will have access to PySearch tools. You can test this by asking Claude to search your codebase:

> "Find all functions that handle user authentication in the src directory"

Claude will automatically use the appropriate PySearch tools to perform this search.

## Using MCP Tools Directly

You can also interact with the MCP server directly for testing purposes.

### Example 1: Basic Text Search

```python
# Example of using the search_text tool
result = search_text(
    pattern="def authenticate_user",
    paths=["./src"],
    context=3
)
```

This will search for the exact text "def authenticate_user" in Python files under the `src` directory, returning 3 lines of context around each match.

### Example 2: Regex Search

```python
# Example of using the search_regex tool
result = search_regex(
    pattern=r"class\s+\w+Test",
    context=2
)
```

This will find all test classes in your codebase using a regular expression pattern.

### Example 3: AST-based Search

```python
# Example of using the search_ast tool
result = search_ast(
    pattern="def",
    func_name=".*_handler$",
    decorator="route|api_endpoint",
    context=3
)
```

This will find all functions that end with "_handler" and are decorated with either "route" or "api_endpoint".

### Example 4: Semantic Search

```python
# Example of using the search_semantic tool
result = search_semantic(
    concept="database",
    context=2
)
```

This will search for code related to database operations using semantic matching.

## Advanced Usage

### Configuration Management

You can configure the search engine to customize its behavior:

```python
# Configure search settings
config = configure_search(
    paths=["./src", "./lib"],
    include_patterns=["**/*.py", "**/*.js"],
    exclude_patterns=["**/tests/**", "**/node_modules/**"],
    context=5,
    parallel=True,
    workers=4
)

# Check current configuration
current_config = get_search_config()
```

### Working with Sessions

The Main MCP Server supports session management for context-aware searches:

```python
# Perform searches within a session
session_result = search_text(
    pattern="User",
    session_id="session123"
)

# Later searches in the same session can leverage context
related_result = search_ast(
    pattern="class",
    class_name="User.*",
    session_id="session123"
)
```

### File Analysis

The Main MCP Server can analyze files for quality metrics:

```python
# Get file statistics
stats = get_file_statistics(
    paths=["./src"],
    include_analysis=True
)
```

## Best Practices

### 1. Use Appropriate Search Types

- Use **text search** for exact matches
- Use **regex search** for pattern matching
- Use **AST search** for structural code queries
- Use **semantic search** for conceptual queries

### 2. Optimize Performance

- Limit the search scope with specific paths when possible
- Use appropriate include/exclude patterns
- Adjust context lines based on your needs
- Use parallel processing for large codebases

### 3. Leverage Advanced Features

- Use session management for context-aware searches
- Apply filters to narrow down results
- Use ranking to prioritize relevant results
- Combine multiple patterns with logical operators

## Troubleshooting

### Common Issues

1. **Server not starting**: Ensure all dependencies are installed
2. **Tools not appearing**: Check Claude Desktop configuration
3. **Slow searches**: Adjust paths and patterns to limit scope
4. **No results**: Verify paths exist and contain the expected files

### Getting Help

For detailed information about each tool, refer to the [MCP API Reference](mcp-api.md).

## Next Steps

- Explore the [Advanced Features](mcp-advanced.md) documentation
- Learn about [Customizing Search Behavior](configuration.md)
- Check out [Real-world Examples](https://github.com/your-org/pysearch/blob/main/examples/mcp_server_example.py)