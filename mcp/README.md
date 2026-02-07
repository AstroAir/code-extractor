# PySearch MCP Server

Production-ready MCP (Model Context Protocol) server for PySearch, built with FastMCP.
Exposes all PySearch code search capabilities as tools for LLM consumption.

## Directory Structure

```
mcp/
├── servers/
│   └── pysearch_mcp_server.py    # Consolidated FastMCP server (all features)
├── shared/
│   ├── validation.py             # Input validation & security
│   ├── session_manager.py        # Context-aware session management
│   ├── progress.py               # Progress tracking for long operations
│   └── resource_manager.py       # LRU cache & resource analytics
├── mcp_config.json               # MCP client configuration
└── README.md                     # This file
```

## Quick Start

```bash
# Install dependencies
pip install fastmcp

# Run with STDIO transport (default, for MCP clients)
python -m mcp.servers.pysearch_mcp_server

# Or use FastMCP CLI
fastmcp run mcp/servers/pysearch_mcp_server.py

# HTTP transport (for web services)
python -m mcp.servers.pysearch_mcp_server --transport http --host 127.0.0.1 --port 9000
```

## Available Tools (13)

| Category | Tool | Description |
|----------|------|-------------|
| Core Search | `search_text` | Text pattern search across files |
| Core Search | `search_regex` | Regex pattern search with validation |
| Core Search | `search_ast` | AST-based structural search with filters |
| Core Search | `search_semantic` | Semantic concept search with expansion |
| Advanced | `search_fuzzy` | Fuzzy approximate string matching |
| Advanced | `search_multi_pattern` | Multi-pattern search with AND/OR |
| Analysis | `analyze_file` | File metrics (lines, functions, classes) |
| Config | `configure_search` | Update search configuration |
| Config | `get_search_config` | Get current configuration |
| Config | `get_supported_languages` | List supported languages |
| Config | `clear_caches` | Clear all caches |
| Utility | `get_search_history` | Recent search history |
| Utility | `get_server_health` | Server health diagnostics |

## MCP Resources

| URI | Description |
|-----|-------------|
| `pysearch://config/current` | Current search configuration |
| `pysearch://history/searches` | Search history |
| `pysearch://stats/overview` | Server statistics |
| `pysearch://languages/supported` | Supported languages |

## MCP Client Configuration

Add to your MCP client config (e.g., Claude Desktop, Cursor):

```json
{
  "mcpServers": {
    "pysearch": {
      "command": "python",
      "args": ["-m", "mcp.servers.pysearch_mcp_server"],
      "env": {"PYTHONPATH": "/path/to/code-extractor"}
    }
  }
}
```

## Integration

The server integrates with all PySearch core features:

- **PySearch engine** — Full text/regex/AST/semantic search
- **Input validation** — Security sanitization, rate limiting, path traversal protection
- **Session management** — Context-aware sessions with intent detection
- **Resource caching** — LRU cache with analytics and health monitoring
- **Progress tracking** — Real-time progress for long-running operations
