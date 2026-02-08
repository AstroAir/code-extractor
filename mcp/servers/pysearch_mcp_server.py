#!/usr/bin/env python3
"""
PySearch MCP Server — Modular FastMCP Implementation

A production-ready MCP (Model Context Protocol) server that exposes all PySearch
functionality as tools for LLM consumption using the FastMCP framework.

Features:
    Core Search:
        - Text search with case sensitivity control
        - Regex pattern search with validation
        - AST-based structural search with filters
        - Semantic concept search with pattern expansion

    Advanced Search:
        - Fuzzy search with configurable similarity thresholds
        - Multi-pattern search with logical operators (AND/OR)
        - Search with advanced result ranking
        - Search with comprehensive filtering (size, date, language, etc.)

    Analysis & Utilities:
        - File content analysis with complexity metrics
        - Search configuration management
        - Supported language listing
        - Cache management
        - Search history tracking

    MCP Features:
        - Progress reporting for long-running operations
        - Session management for context-aware searches
        - Resource management with LRU caching
        - Input validation and security sanitization
        - MCP resource endpoints for configuration and statistics

Usage:
    # Run with STDIO transport (default, for MCP clients)
    python -m mcp.servers.pysearch_mcp_server

    # Or use FastMCP CLI
    fastmcp run mcp/servers/pysearch_mcp_server.py

    # For HTTP transport (web services)
    python -m mcp.servers.pysearch_mcp_server --transport http --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

import logging
from typing import Any

# FastMCP imports
try:
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False

# Engine and data structures (re-exported for backward compatibility)
from .engine import ConfigResponse, PySearchEngine, SearchResponse

# Shared MCP utilities
from ..shared.validation import (
    PerformanceValidationError,
    SecurityValidationError,
    ValidationError,
    check_validation_results,
    get_sanitized_values,
    validate_tool_input,
)

# Tool and resource registration
from .tools import register_all_tools
from .resources import register_resources

# Setup logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastMCP server factory
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP | None:
    """Create and configure the FastMCP server with all PySearch tools."""
    if not FASTMCP_AVAILABLE:
        logger.error("FastMCP not available. Install with: pip install fastmcp")
        return None

    engine = PySearchEngine()

    mcp = FastMCP(
        name="PySearch",
        instructions="""
        PySearch MCP Server — Comprehensive code search capabilities for LLM agents.

        Core Search Tools:
        - search_text: Basic text search across files
        - search_regex: Regex pattern search with validation
        - search_ast: AST-based structural search with filters
        - search_semantic: Semantic concept search with pattern expansion

        Advanced Search Tools:
        - search_fuzzy: Fuzzy text search with similarity matching
        - search_multi_pattern: Multiple pattern search with AND/OR operators
        - suggest_corrections: Spelling corrections based on codebase identifiers
        - search_word_fuzzy: Word-level fuzzy search with similarity algorithms

        Analysis Tools:
        - analyze_file: File content analysis with metrics

        Configuration Tools:
        - configure_search: Update search configuration
        - get_search_config: Get current configuration
        - get_supported_languages: List supported programming languages
        - clear_caches: Clear search engine caches and stale data

        Utility Tools:
        - get_search_history: Get recent search history
        - get_server_health: Get server health and diagnostics

        Session Management Tools:
        - create_session: Create a context-aware search session
        - get_session_info: Get session details, intent, and recommendations

        Progress Tracking Tools:
        - get_operation_progress: Query progress of running operations
        - cancel_operation: Cancel a running operation

        All search tools accept an optional session_id for context tracking.

        Resources:
        - pysearch://config/current: Current search configuration
        - pysearch://history/searches: Search history
        - pysearch://stats/overview: Server statistics with session and progress data
        - pysearch://sessions/analytics: Session management analytics
        - pysearch://languages/supported: Supported languages
        """
    )

    # -- Validation helper --------------------------------------------------

    def _validate(*, is_regex: bool = False, **kwargs: Any) -> dict[str, Any]:
        """Validate and sanitize tool inputs, raising ToolError on failure.

        Args:
            is_regex: If True, use regex-specific pattern validation.
            **kwargs: Tool parameters to validate.
        """
        # Rate limiting — check before spending resources on validation
        rate_result = engine.validator.check_rate_limit("mcp_client")
        if not rate_result.is_valid:
            raise ToolError(
                f"Rate limit exceeded: {rate_result.errors[0].message}"
                if rate_result.errors else "Rate limit exceeded"
            )

        # Use regex-specific validation when applicable
        if is_regex and "pattern" in kwargs:
            regex_result = engine.validator.validate_regex_pattern(kwargs["pattern"])
            engine.validator.record_validation("regex_pattern", regex_result)
            if not regex_result.is_valid:
                errors_msg = "; ".join(e.message for e in regex_result.errors)
                raise ToolError(f"Regex validation failed: {errors_msg}")
            # Replace pattern with sanitized value for downstream
            kwargs["pattern"] = regex_result.sanitized_value

        results = validate_tool_input(**kwargs)
        try:
            check_validation_results(results)
        except ValidationError as e:
            raise ToolError(f"Input validation failed: {e.message}") from e
        except SecurityValidationError as e:
            raise ToolError(f"Security validation failed: {e.message}") from e
        except PerformanceValidationError as e:
            raise ToolError(f"Performance validation failed: {e.message}") from e
        return get_sanitized_values(results)

    # -- Register all tools and resources -----------------------------------

    register_all_tools(mcp, engine, _validate)
    register_resources(mcp, engine)

    return mcp


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Module-level server instance for `fastmcp run` compatibility
mcp = create_mcp_server()


def main() -> None:
    """Main entry point for the PySearch MCP server."""
    if not FASTMCP_AVAILABLE:
        print("Error: FastMCP is not available.")
        print("Please install it with: pip install fastmcp")
        return

    global mcp
    if mcp is None:
        mcp = create_mcp_server()

    if mcp:
        print("Starting PySearch MCP Server...")
        # Default: STDIO transport for MCP clients
        mcp.run()
        # For HTTP transport: mcp.run(transport="http", host="127.0.0.1", port=9000)
        # For SSE transport: mcp.run(transport="sse", host="127.0.0.1", port=9000)
    else:
        print("Failed to create MCP server")


if __name__ == "__main__":
    main()
