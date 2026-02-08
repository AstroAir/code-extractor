# Model Context Protocol (MCP) Support

PySearch provides comprehensive integration with the Model Context Protocol (MCP), allowing AI assistants to leverage powerful code search capabilities directly within their workflows.

## What is MCP?

The Model Context Protocol (MCP) is an open standard that enables secure, controlled access to local resources and tools from within AI assistant applications. It allows AI assistants to interact with your codebase in a structured and safe way, providing access to file contents, search functionality, and other development tools.

## PySearch MCP Servers

PySearch includes several MCP server implementations that expose different levels of functionality:

### Available Servers

1. **Main MCP Server** (`mcp_server.py`)
   - Advanced features with fuzzy search, analysis, and composition
   - Comprehensive search capabilities with ranking and filtering
   - Session management and progress reporting
   - File content analysis and quality metrics

2. **Basic MCP Server** (`basic_mcp_server.py`)
   - Core search functionality (text, regex, AST, semantic)
   - Simple configuration management
   - Search history and statistics

3. **FastMCP Server** (`fastmcp_server.py`)
   - FastMCP framework-based implementation
   - Optimized performance
   - Enhanced error handling

## Quick Start

### Running an MCP Server

To run the main MCP server:

```bash
python mcp/servers/mcp_server.py
```

For the basic server:

```bash
python mcp/servers/basic_mcp_server.py
```

### Integration with AI Assistants

The MCP servers are designed to work with MCP-compatible AI assistants like Claude Desktop. Once configured, these tools will be available directly within the assistant interface.

## Core Capabilities

PySearch MCP servers provide the following capabilities to AI assistants:

### Search Functionality

- **Text Search**: Find literal text patterns in your codebase
- **Regex Search**: Use regular expressions for complex pattern matching
- **AST Search**: Structurally search code with filters for functions, classes, decorators, and imports
- **Semantic Search**: Search for conceptual matches rather than exact patterns
- **Fuzzy Search**: Find approximate matches with configurable similarity thresholds
- **Multi-pattern Search**: Combine multiple patterns with logical operators (AND, OR, NOT)

### Advanced Features

- **Result Ranking**: Results are ranked by relevance using multiple factors
- **Filtering**: Filter results by file size, modification date, complexity, and more
- **Context Management**: Maintain search sessions with context awareness
- **File Analysis**: Get detailed statistics and quality metrics for files
- **Progress Reporting**: Track long-running search operations

### Configuration Management

- **Search Paths**: Configure which directories to search
- **Include/Exclude Patterns**: Specify which files to include or exclude
- **Context Lines**: Control how much context to show around matches
- **Parallel Processing**: Configure parallel execution for performance
- **Language Filtering**: Filter by programming language

### Utility Functions

- **Supported Languages**: Get a list of supported programming languages
- **Cache Management**: Clear search caches to free memory
- **Search History**: Access recent search operations

## Usage Examples

### Basic Text Search

Find all occurrences of "main" function definitions:

```
search_text("def main", paths=["./src"], context=2)
```

### Regex Search

Find all test classes:

```
search_regex("class \\w+Test", context=3)
```

### AST-based Search

Find all handler functions:

```
search_ast("def", func_name=".*_handler$", context=2)
```

### Semantic Search

Find database-related code:

```
search_semantic("database", context=3)
```

### Configuration Management

Update search configuration:

```
configure_search(
    paths=["./src", "./tests"],
    context=5,
    workers=4
)
```

## Integration with Claude Desktop

To use PySearch with Claude Desktop:

1. Install the required dependencies:
   ```bash
   pip install fastmcp rapidfuzz
   ```

2. Configure Claude Desktop to connect to the PySearch MCP server

3. Start using the tools directly in your conversations with Claude

The tools will appear as available functions within Claude, allowing you to search your codebase naturally as part of your conversation.

## API Reference

For detailed API documentation, see [MCP API Reference](api.md).