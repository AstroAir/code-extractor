# MCP Servers Module

[根目录](../CLAUDE.md) > **mcp**

---

## Change Log (Changelog)

### 2026-02-07 - Consolidated MCP Server

- Merged all server implementations into single `pysearch_mcp_server.py`
- Removed duplicate/broken files: `basic_mcp_server.py`, `enhanced_fastmcp_server.py.disabled`, `enhanced_fastmcp_server_simple.py`
- Full FastMCP integration with all PySearch features
- Input validation, session management, progress tracking, resource management

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive MCP module documentation

---

## Module Responsibility

The **MCP** (Model Context Protocol) module provides a FastMCP server for LLM integration:

1. **MCP Server**: Production FastMCP server exposing all PySearch capabilities
2. **Shared Utilities**: Validation, session management, progress tracking, resource caching
3. **LLM Integration**: Protocol-based AI assistant integration via STDIO/HTTP/SSE

---

## Key Files

### Server
| File | Purpose | Description |
|------|---------|-------------|
| `servers/pysearch_mcp_server.py` | MCP Server | Consolidated FastMCP server with all features |

### Shared Utilities
| File | Purpose | Description |
|------|---------|-------------|
| `shared/validation.py` | Validation | Input validation, security sanitization, rate limiting |
| `shared/session_manager.py` | Session Management | Context-aware session management with learning |
| `shared/progress.py` | Progress Reporting | Progress tracking for long-running operations |
| `shared/resource_manager.py` | Resource Management | LRU cache with analytics and health monitoring |

### Configuration
| File | Purpose | Description |
|------|---------|-------------|
| `mcp_config.json` | MCP Configuration | Server tool/resource definitions for MCP clients |
| `README.md` | Module Docs | MCP module documentation |

---

## MCP Server

**File**: `servers/pysearch_mcp_server.py`

### Tools (13 total)

**Core Search** (4):

- `search_text` — Text search across files
- `search_regex` — Regex pattern search with validation
- `search_ast` — AST-based structural search with filters
- `search_semantic` — Semantic concept search with pattern expansion

**Advanced Search** (2):
- `search_fuzzy` — Fuzzy approximate string matching
- `search_multi_pattern` — Multi-pattern search with AND/OR operators

**Analysis** (1):
- `analyze_file` — File metrics (lines, functions, classes, imports, complexity)

**Configuration** (4):
- `configure_search` — Update search settings
- `get_search_config` — Get current configuration
- `get_supported_languages` — List supported languages
- `clear_caches` — Clear all caches

**Utilities** (2):
- `get_search_history` — Recent search history
- `get_server_health` — Health diagnostics

### Resources (4)
- `pysearch://config/current` — Current configuration
- `pysearch://history/searches` — Search history
- `pysearch://stats/overview` — Server statistics
- `pysearch://languages/supported` — Supported languages

### Usage

```bash
# STDIO transport (default, for MCP clients)
python -m mcp.servers.pysearch_mcp_server

# FastMCP CLI
fastmcp run mcp/servers/pysearch_mcp_server.py

# HTTP transport (for web services)
python -m mcp.servers.pysearch_mcp_server --transport http --host 127.0.0.1 --port 9000
```

---

## Shared Utility Details

### Validation (`shared/validation.py`)

```python
from mcp.shared.validation import validate_tool_input, check_validation_results
results = validate_tool_input(pattern="def main", paths=["."], context=3)
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

## Dependencies

### Internal
- `pysearch` — Core search engine (PySearch, SearchConfig, Query, etc.)
- `pysearch.core.types` — ASTFilters, Language, SearchResult
- `pysearch.semantic` — Semantic query expansion (optional)

### External
- `fastmcp>=2.11.0` — MCP framework

---

## Testing

```bash
# Run MCP-related tests
pytest tests/integration/test_mcp_features.py -v
pytest tests/test_enhanced_mcp_server.py -v
python test_mcp_core.py
```
