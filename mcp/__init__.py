"""
PySearch MCP (Model Context Protocol) Package

This package contains MCP server implementations and utilities for exposing
PySearch functionality to LLM clients.

Available modules:
- servers: MCP server implementations
- shared: Shared utilities and components
"""

__version__ = "0.1.0"
__author__ = "Kilo Code"

# Re-export commonly used components
# Note: Import these modules only when needed to avoid circular dependencies
# from .shared import mcp_composition, mcp_progress, mcp_prompts, mcp_resources

__all__: list[str] = [
    # "mcp_composition",
    # "mcp_progress",
    # "mcp_prompts",
    # "mcp_resources",
]
