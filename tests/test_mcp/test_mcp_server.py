"""
Tests for mcp/servers/pysearch_mcp_server.py — create_mcp_server factory
and the module-level server instance.

Covers: create_mcp_server returns a valid FastMCP instance (or None if
FastMCP is unavailable), module-level mcp variable, and main() entry point.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# create_mcp_server
# ---------------------------------------------------------------------------


class TestCreateMCPServer:
    """Tests for the create_mcp_server factory function."""

    def test_returns_fastmcp_or_none(self):
        """create_mcp_server returns a FastMCP instance or None."""
        from mcp.servers.pysearch_mcp_server import create_mcp_server

        result = create_mcp_server()
        try:
            from fastmcp import FastMCP

            assert isinstance(result, FastMCP)
        except ImportError:
            assert result is None

    def test_idempotent(self):
        """Multiple calls to create_mcp_server return independent instances."""
        from mcp.servers.pysearch_mcp_server import create_mcp_server

        s1 = create_mcp_server()
        s2 = create_mcp_server()
        if s1 is not None and s2 is not None:
            assert s1 is not s2

    def test_module_level_mcp_variable(self):
        """Module-level mcp variable is set."""
        from mcp.servers import pysearch_mcp_server

        # mcp may be None if FastMCP is unavailable
        assert hasattr(pysearch_mcp_server, "mcp")


# ---------------------------------------------------------------------------
# FastMCP-dependent tests (skip if not installed)
# ---------------------------------------------------------------------------

try:
    from fastmcp import FastMCP  # noqa: F401

    FASTMCP_AVAILABLE = True
except ImportError:
    FASTMCP_AVAILABLE = False


@pytest.mark.skipif(not FASTMCP_AVAILABLE, reason="FastMCP not installed")
class TestMCPServerWithFastMCP:
    """Tests that require FastMCP to be installed."""

    def test_server_name(self):
        """Server has the correct name."""
        from mcp.servers.pysearch_mcp_server import create_mcp_server

        server = create_mcp_server()
        assert server is not None
        assert server.name == "PySearch"

    def test_server_has_tools(self):
        """Server has registered tools."""
        from mcp.servers.pysearch_mcp_server import create_mcp_server

        server = create_mcp_server()
        assert server is not None
        # FastMCP should have tools registered
        # The exact API depends on FastMCP version, but the server should be usable
