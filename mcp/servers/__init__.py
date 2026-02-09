"""
PySearch MCP Servers

This module contains the MCP server implementation for PySearch.

Available servers:
- pysearch_mcp_server: Production FastMCP server with all PySearch features

Sub-modules:
- engine: Core PySearchEngine class and data structures
- tools: MCP tool registration modules (core_search, advanced_search, etc.)
- resources: MCP resource endpoint registration
"""

from .engine import ConfigResponse, PySearchEngine, SearchResponse
from .pysearch_mcp_server import create_mcp_server

__all__ = [
    "PySearchEngine",
    "SearchResponse",
    "ConfigResponse",
    "create_mcp_server",
]
