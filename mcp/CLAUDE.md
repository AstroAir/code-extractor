# MCP Module

[根目录](../CLAUDE.md) > **mcp**

---

## Change Log (Changelog)

### 2026-02-08 - FastMCP Integration Update
- Consolidated all MCP server implementations into single FastMCP server
- Integrated session management, progress tracking, resource management
- Added 19 tools and 5 resources
- Enhanced with validation, rate limiting, and security

### 2026-02-07 - Server Consolidation
- Merged all server implementations into `pysearch_mcp_server.py`
- Removed duplicate/broken server files

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive MCP module documentation

---

## Module Responsibility

The **MCP** (Model Context Protocol) module provides a production-ready FastMCP server for LLM integration, exposing all PySearch capabilities as tools and resources.

### Key Responsibilities
1. **MCP Server**: FastMCP-based server with STDIO/HTTP/SSE transports
2. **Tool Exposure**: 19 tools covering all PySearch features
3. **Resource Management**: 5 resources for configuration and state
4. **Shared Utilities**: Validation, session management, progress tracking

---

## Entry and Startup

### Main Entry Point
- **`servers/pysearch_mcp_server.py`** - Consolidated FastMCP server
  - All 19 tools in one file
  - FastMCP framework integration

### Shared Utilities
- **`shared/validation.py`** - Input validation and security
- **`shared/session_manager.py`** - Session management
- **`shared/progress.py`** - Progress tracking
- **`shared/resource_manager.py`** - Resource caching

---

## Public API

### Running the Server

```bash
# STDIO transport (default, for MCP clients)
python -m mcp.servers.pysearch_mcp_server

# FastMCP CLI
fastmcp run mcp/servers/pysearch_mcp_server.py

# HTTP transport (for web services)
python -m mcp.servers.pysearch_mcp_server --transport http --host 127.0.0.1 --port 9000
```

### MCP Client Configuration

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

---

## MCP Tools (19)

### Core Search (4)
- `search_text` - Text search across files
- `search_regex` - Regex pattern search with validation
- `search_ast` - AST-based structural search with filters
- `search_semantic` - Semantic concept search with expansion

### Advanced Search (4)
- `search_fuzzy` - Fuzzy approximate string matching
- `search_multi_pattern` - Multi-pattern search with AND/OR
- `suggest_corrections` - Spelling corrections from codebase
- `search_word_fuzzy` - Word-level fuzzy search with algorithms

### Analysis (1)
- `analyze_file` - File metrics (lines, functions, classes)

### Configuration (4)
- `configure_search` - Update search settings
- `get_search_config` - Get current configuration
- `get_supported_languages` - List supported languages
- `clear_caches` - Clear caches, optimize resources

### Utilities (2)
- `get_search_history` - Recent search history
- `get_server_health` - Comprehensive health stats

### Session Management (2)
- `create_session` - Create context-aware search session
- `get_session_info` - Session details, intent, recommendations

### Progress Tracking (2)
- `get_operation_progress` - Query progress of operations
- `cancel_operation` - Cancel a running operation

All search tools accept optional `session_id` for context tracking.

---

## MCP Resources (5)

- `pysearch://config/current` - Current configuration
- `pysearch://history/searches` - Search history
- `pysearch://stats/overview` - Statistics with session & progress data
- `pysearch://sessions/analytics` - Session analytics
- `pysearch://languages/supported` - Supported languages

---

## Key Dependencies and Configuration

### Internal Dependencies
- `pysearch` - Core search engine
- `pysearch.core.types` - ASTFilters, Language, SearchResult
- `pysearch.semantic` - Semantic query expansion (optional)

### External Dependencies
- `fastmcp>=2.11.0` - MCP framework

---

## Shared Utility Details

### Validation (`shared/validation.py`)

```python
from mcp.shared.validation import validate_tool_input, check_validation_results

results = validate_tool_input(
    pattern="def main",
    paths=["."],
    context=3
)
check_validation_results(results)  # raises ValidationError on failure
```

### Session Management (`shared/session_manager.py`)

```python
from mcp.shared.session_manager import get_session_manager

mgr = get_session_manager()
session = mgr.create_session(user_id="user1")
mgr.record_search(session.session_id, {"pattern": "def main"}, 12.5, 3)
```

### Progress Tracking (`shared/progress.py`)

```python
from mcp.shared.progress import ProgressTracker

tracker = ProgressTracker()
tracker.start_operation("search_001", total_steps=10, description="Searching")
tracker.update_progress("search_001", 5, "Half done")
tracker.complete_operation("search_001", success=True)
```

### Resource Management (`shared/resource_manager.py`)

```python
from mcp.shared.resource_manager import ResourceManager

rm = ResourceManager(max_cache_size=100, default_ttl=300.0)
rm.set_cache("key", {"data": "value"})
rm.get_cache("key")
rm.get_health_status()
```

---

## Testing

### Test Directory
- `tests/integration/` - Integration tests
  - `test_mcp_features.py` - MCP feature tests
  - `test_mcp_server.py` - MCP server tests
- `tests/test_enhanced_mcp_server.py` - Enhanced server tests

### Running Tests
```bash
pytest tests/integration/test_mcp_features.py -v
pytest tests/integration/test_mcp_server.py -v
python test_mcp_core.py
```

---

## Common Issues and Solutions

### Issue 1: MCP server not starting
**Symptoms**: Server fails to start
**Solution**: Check fastmcp installation:
```bash
pip install fastmcp>=2.11.0
```

### Issue 2: Tools not available in client
**Symptoms**: Tools don't appear in MCP client
**Solution**: Verify server configuration and check server logs for errors

### Issue 3: Session tracking not working
**Symptoms**: Search history not associated with sessions
**Solution**: Ensure `session_id` is passed to search tools:
```python
result = await session.call_tool("search_text", {
    "pattern": "def main",
    "session_id": session_id
})
```

---

## Related Files

### MCP Module Files
- `mcp/__init__.py`
- `mcp/servers/pysearch_mcp_server.py` - Consolidated FastMCP server
- `mcp/shared/validation.py` - Input validation
- `mcp/shared/session_manager.py` - Session management
- `mcp/shared/progress.py` - Progress tracking
- `mcp/shared/resource_manager.py` - Resource management
- `mcp/mcp_config.json` - MCP client configuration
- `mcp/README.md` - Module documentation

---

## Module Structure

```
mcp/
├── __init__.py
├── servers/
│   ├── __init__.py
│   └── pysearch_mcp_server.py    # Consolidated FastMCP server
├── shared/
│   ├── __init__.py
│   ├── validation.py             # Input validation
│   ├── session_manager.py        # Session management
│   ├── progress.py               # Progress tracking
│   └── resource_manager.py       # Resource caching
├── mcp_config.json               # MCP configuration
└── README.md                     # Module docs
```
