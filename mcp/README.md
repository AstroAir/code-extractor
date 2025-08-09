# PySearch MCP Servers

This directory contains Model Context Protocol (MCP) server implementations for PySearch, providing LLM-accessible interfaces to PySearch functionality.

## Directory Structure

```
mcp/
├── servers/                    # MCP server implementations
│   ├── mcp_server.py          # Main MCP server with advanced features
│   ├── basic_mcp_server.py    # Basic/legacy MCP server
│   ├── fastmcp_server.py      # FastMCP-based server
│   └── pysearch_mcp_server.py # Alternative implementation
├── shared/                     # Shared MCP utilities
│   ├── composition.py         # Search composition utilities
│   ├── progress.py            # Progress reporting
│   ├── prompts.py             # Prompt templates
│   └── resources.py           # Resource management
├── mcp_config.json            # MCP configuration
└── README.md                  # This file
```

## Available Servers

### Main MCP Server (`mcp_server.py`)

The primary MCP server with comprehensive advanced features:

- Fuzzy search with configurable similarity thresholds
- Multi-pattern search with logical operators
- File content analysis and complexity metrics
- Advanced search result ranking
- Comprehensive filtering capabilities
- Progress reporting for long operations
- Context-aware search sessions
- MCP resource management
- Prompt templates for common scenarios
- Composition support for chaining operations

### Basic MCP Server (`basic_mcp_server.py`)

Legacy/simple implementation with core functionality:

- Text and regex pattern search
- AST-based search with filters
- Configuration management
- Search history and statistics

### FastMCP Server (`fastmcp_server.py`)

FastMCP framework-based implementation:

- All main server features
- Optimized performance
- Enhanced error handling
- Better resource management

## Usage

### Running a Server

```bash
# Main server (recommended)
python mcp/servers/mcp_server.py

# Basic server (legacy)
python mcp/servers/basic_mcp_server.py

# FastMCP server
python mcp/servers/fastmcp_server.py
```

### Configuration

Edit `mcp_config.json` to customize server behavior:

```json
{
  "default_paths": ["."],
  "include_patterns": ["**/*.py"],
  "exclude_patterns": ["**/__pycache__/**"],
  "max_results": 100,
  "context_lines": 3
}
```

## Integration

These servers are designed to work with MCP-compatible LLM clients, providing seamless access to PySearch's powerful code search capabilities within AI-assisted development workflows.

## Documentation

- `MCP_SERVER_README.md` - Detailed server documentation
- `MCP_SERVER_SUMMARY.md` - Quick reference
- `ENHANCED_FEATURES_README.md` - Enhanced features documentation
- `ENHANCED_IMPLEMENTATION_SUMMARY.md` - Implementation details
