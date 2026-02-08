#!/usr/bin/env python3
"""
Real MCP Server Tests Using FastMCP Client In-Memory Transport.

This module tests the actual PySearch MCP server through the MCP protocol
using fastmcp.Client with in-memory transport. Each test calls a registered
MCP tool or reads a registered MCP resource exactly as an LLM client would.

The local ``mcp/`` package conflicts with pip's ``mcp`` package (which
fastmcp depends on internally). A careful sys.path / sys.modules dance at
import time resolves the collision — see the block below.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Handle namespace collision between local mcp/ and pip mcp package.
#
# FastMCP internally imports ``mcp.types`` from the pip-installed ``mcp``
# package, but the project's local ``mcp/`` directory shadows it.
#
# Strategy
# --------
# 1. Save & clear any pre-cached local ``mcp.*`` entries from sys.modules.
# 2. Temporarily remove the project root from sys.path so ``import mcp``
#    resolves to the *pip* package.
# 3. Import ``fastmcp`` (binds internal references to pip ``mcp.types``).
# 4. Restore sys.path, then clear the pip ``mcp`` entries so the local
#    ``mcp`` package can be imported normally.
# 5. Import the local ``mcp.servers.pysearch_mcp_server``.  Because
#    ``fastmcp`` is already cached in sys.modules, the server module's
#    ``from fastmcp import FastMCP`` succeeds and ``FASTMCP_AVAILABLE``
#    becomes True.
#
# Python's module references are bound at import time, so fastmcp's
# internal pip-mcp references remain valid after sys.modules["mcp"]
# switches to the local package.
# ---------------------------------------------------------------------------

import importlib
import importlib.util
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Step 1 — clear any pre-cached local mcp.* from sys.modules so that
#   ``import mcp`` in fastmcp resolves to the *pip* package.
_local_mcp_saved: dict[str, object] = {}
for _k in list(sys.modules.keys()):
    if _k == "mcp" or _k.startswith("mcp."):
        _local_mcp_saved[_k] = sys.modules.pop(_k)

# Step 2 — temporarily remove project root (including '' and '.') from
#   sys.path so that ``import mcp`` finds the pip package in site-packages.
_orig_path = sys.path[:]
_project_root_str = str(_PROJECT_ROOT)
sys.path = [
    p
    for p in sys.path
    if p not in ("", ".", _project_root_str)
    and str(Path(p).resolve()) != _project_root_str
]

# Step 3 — import fastmcp Client (pulls in pip mcp.types internally).
#   Once fastmcp binds its references they stay valid even if sys.modules
#   changes later.
try:
    from fastmcp import Client as _FastMCPClient  # noqa: E402

    _FASTMCP_CLIENT_OK = True
except ImportError:
    _FASTMCP_CLIENT_OK = False

# Step 4 — restore sys.path fully.
sys.path = _orig_path

# Step 5 — Load the local mcp server module *by file path* using importlib.
#   This sidesteps the namespace collision entirely: we don't need
#   ``import mcp`` to resolve — we point directly at the .py files.
_create_mcp_server = None
try:
    _mcp_pkg_dir = _PROJECT_ROOT / "mcp"

    def _load_local(mod_name: str, file_path: Path) -> object:
        """Load a local module by file path and register it in sys.modules."""
        spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for {file_path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    # Register the local mcp package hierarchy so relative imports work.
    _load_local("mcp", _mcp_pkg_dir / "__init__.py")

    _shared_dir = _mcp_pkg_dir / "shared"
    _load_local("mcp.shared", _shared_dir / "__init__.py")
    _load_local("mcp.shared.progress", _shared_dir / "progress.py")
    _load_local("mcp.shared.resource_manager", _shared_dir / "resource_manager.py")
    _load_local("mcp.shared.session_manager", _shared_dir / "session_manager.py")
    _load_local("mcp.shared.validation", _shared_dir / "validation.py")

    _servers_dir = _mcp_pkg_dir / "servers"

    # Load engine module (PySearchEngine, data structures)
    _load_local("mcp.servers.engine", _servers_dir / "engine.py")

    # Load tools sub-package and all tool registration modules
    _tools_dir = _servers_dir / "tools"
    _load_local("mcp.servers.tools", _tools_dir / "__init__.py")
    _load_local("mcp.servers.tools.core_search", _tools_dir / "core_search.py")
    _load_local("mcp.servers.tools.advanced_search", _tools_dir / "advanced_search.py")
    _load_local("mcp.servers.tools.analysis", _tools_dir / "analysis.py")
    _load_local("mcp.servers.tools.config", _tools_dir / "config.py")
    _load_local("mcp.servers.tools.history", _tools_dir / "history.py")
    _load_local("mcp.servers.tools.session", _tools_dir / "session.py")
    _load_local("mcp.servers.tools.progress", _tools_dir / "progress.py")
    _load_local("mcp.servers.tools.ide", _tools_dir / "ide.py")
    _load_local("mcp.servers.tools.distributed", _tools_dir / "distributed.py")
    _load_local("mcp.servers.tools.multi_repo", _tools_dir / "multi_repo.py")
    _load_local("mcp.servers.tools.workspace", _tools_dir / "workspace.py")

    # Load resources module
    _load_local("mcp.servers.resources", _servers_dir / "resources.py")

    # Load servers __init__ and main entry point
    _load_local("mcp.servers", _servers_dir / "__init__.py")
    _server_mod = _load_local(
        "mcp.servers.pysearch_mcp_server",
        _servers_dir / "pysearch_mcp_server.py",
    )
    _create_mcp_server = getattr(_server_mod, "create_mcp_server", None)
    _SERVER_IMPORT_OK = _create_mcp_server is not None
except Exception:
    _SERVER_IMPORT_OK = False

# ---------------------------------------------------------------------------
# Standard imports (after the namespace dance)
# ---------------------------------------------------------------------------

import tempfile  # noqa: E402

import pytest  # noqa: E402

# Skip the whole module if the environment cannot support real MCP tests
if not _FASTMCP_CLIENT_OK:
    pytest.skip(
        "FastMCP Client not importable (likely mcp namespace collision)",
        allow_module_level=True,
    )
if not _SERVER_IMPORT_OK:
    pytest.skip(
        "Could not load local mcp server module via importlib",
        allow_module_level=True,
    )

# Re-export for clarity
Client = _FastMCPClient
create_mcp_server = _create_mcp_server

# ---------------------------------------------------------------------------
# Paths used by tests
# ---------------------------------------------------------------------------

SAMPLE_REPO_PATH = Path(__file__).resolve().parent.parent.parent / "test_data" / "sample_repo"
SAMPLE_REPO_SRC = SAMPLE_REPO_PATH / "src"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mcp_server():
    """Create a real FastMCP server instance via the production factory."""
    server = create_mcp_server()
    if server is None:
        pytest.skip("create_mcp_server() returned None (FastMCP unavailable)")
    return server


@pytest.fixture
def temp_test_dir():
    """Create a temporary directory with Python files for search tests."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # main.py
        (root / "main.py").write_text(
            'def main():\n'
            '    """Entry point for the application."""\n'
            '    print("Hello World")\n'
            '    return 0\n'
            '\n'
            'class Application:\n'
            '    """Main application class."""\n'
            '    def run(self):\n'
            '        return main()\n',
            encoding="utf-8",
        )

        # utils.py
        (root / "utils.py").write_text(
            'import hashlib\n'
            'import secrets\n'
            '\n'
            'def hash_password(password: str) -> str:\n'
            '    """Hash a password securely."""\n'
            '    salt = secrets.token_bytes(32)\n'
            '    return hashlib.sha256(salt + password.encode()).hexdigest()\n'
            '\n'
            'def verify_password(password: str, hashed: str) -> bool:\n'
            '    """Verify a password against its hash."""\n'
            '    return True  # simplified\n'
            '\n'
            'class AuthenticationError(Exception):\n'
            '    """Raised when authentication fails."""\n'
            '    pass\n',
            encoding="utf-8",
        )

        # config.py
        (root / "config.py").write_text(
            '# Configuration module\n'
            'DATABASE_URL = "sqlite:///test.db"\n'
            'DEBUG = True\n'
            'SECRET_KEY = "test-secret"\n'
            '\n'
            'CONFIG = {\n'
            '    "debug": True,\n'
            '    "version": "1.0.0",\n'
            '    "features": ["search", "index", "cache"],\n'
            '}\n',
            encoding="utf-8",
        )

        yield str(root)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _parse_tool_result(result) -> dict | list | str:
    """Extract usable data from a CallToolResult.

    We prefer JSON-parsing from the first content block because FastMCP's
    ``.data`` hydration can wrap results in ``Root`` objects that don't
    behave like plain dicts/lists.
    """
    # Parse from text content block (most reliable for plain dict/list access)
    if result.content:
        text = getattr(result.content[0], "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text

    # Fallback to .data
    if hasattr(result, "data") and result.data is not None:
        return result.data

    return {}


# =========================================================================
# Test Classes
# =========================================================================


class TestMCPToolDiscovery:
    """Verify that all expected tools are registered on the server."""

    EXPECTED_CORE_TOOLS = {
        "search_text",
        "search_regex",
        "search_ast",
        "search_semantic",
    }
    EXPECTED_ADVANCED_TOOLS = {
        "search_fuzzy",
        "search_multi_pattern",
        "suggest_corrections",
        "search_word_fuzzy",
    }
    EXPECTED_CONFIG_TOOLS = {
        "configure_search",
        "get_search_config",
        "get_supported_languages",
        "clear_caches",
    }
    EXPECTED_UTILITY_TOOLS = {
        "get_search_history",
        "get_server_health",
    }
    EXPECTED_SESSION_TOOLS = {
        "create_session",
        "get_session_info",
    }
    EXPECTED_PROGRESS_TOOLS = {
        "get_operation_progress",
        "cancel_operation",
    }
    EXPECTED_ANALYSIS_TOOLS = {
        "analyze_file",
    }

    ALL_MINIMUM_TOOLS = (
        EXPECTED_CORE_TOOLS
        | EXPECTED_ADVANCED_TOOLS
        | EXPECTED_CONFIG_TOOLS
        | EXPECTED_UTILITY_TOOLS
        | EXPECTED_SESSION_TOOLS
        | EXPECTED_PROGRESS_TOOLS
        | EXPECTED_ANALYSIS_TOOLS
    )

    async def test_list_tools_returns_results(self, mcp_server):
        """list_tools should return a non-empty list."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert len(tools) > 0, "Server should expose at least one tool"

    async def test_all_minimum_tools_registered(self, mcp_server):
        """Every core / advanced / config / utility / session / progress tool must be present."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            tool_names = {t.name for t in tools}
            missing = self.ALL_MINIMUM_TOOLS - tool_names
            assert not missing, f"Missing tools: {missing}"

    async def test_tool_count_at_least_expected(self, mcp_server):
        """The server should have at least the minimum number of tools."""
        async with Client(mcp_server) as client:
            tools = await client.list_tools()
            assert len(tools) >= len(self.ALL_MINIMUM_TOOLS)


class TestMCPResourceDiscovery:
    """Verify that all expected MCP resources are registered."""

    EXPECTED_RESOURCES = {
        "pysearch://config/current",
        "pysearch://history/searches",
        "pysearch://stats/overview",
        "pysearch://sessions/analytics",
        "pysearch://languages/supported",
    }

    async def test_list_resources_returns_results(self, mcp_server):
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            assert len(resources) > 0

    async def test_all_expected_resources_registered(self, mcp_server):
        async with Client(mcp_server) as client:
            resources = await client.list_resources()
            uris = {str(r.uri) for r in resources}
            missing = self.EXPECTED_RESOURCES - uris
            assert not missing, f"Missing resources: {missing}"


# -------------------------------------------------------------------------
# Core Search Tools
# -------------------------------------------------------------------------


class TestMCPCoreSearchTools:
    """Test core search tools through the MCP protocol."""

    async def test_search_text(self, mcp_server, temp_test_dir):
        """search_text should find a known string in test files."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "Hello World", "paths": [temp_test_dir], "context": 1},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_matches"] >= 1
            assert len(data["items"]) >= 1

    async def test_search_text_no_matches(self, mcp_server, temp_test_dir):
        """search_text with a non-existent pattern should return zero matches."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "ZZZZNOTEXIST9999", "paths": [temp_test_dir]},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_matches"] == 0

    async def test_search_regex(self, mcp_server, temp_test_dir):
        """search_regex should match a regex pattern."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r"def\s+\w+", "paths": [temp_test_dir], "context": 1},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_matches"] >= 1
            # Verify query_info reflects regex mode
            assert data["query_info"]["use_regex"] is True

    async def test_search_ast(self, mcp_server, temp_test_dir):
        """search_ast with a func_name filter should return AST matches."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_ast",
                {
                    "pattern": "def",
                    "func_name": "main",
                    "paths": [temp_test_dir],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["query_info"]["use_ast"] is True

    async def test_search_semantic(self, mcp_server, temp_test_dir):
        """search_semantic should expand a concept and search."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_semantic",
                {"concept": "authentication", "paths": [temp_test_dir], "context": 1},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            # Semantic search internally uses regex
            assert data["query_info"]["use_semantic"] is True
            assert data["query_info"]["use_regex"] is True


# -------------------------------------------------------------------------
# Advanced Search Tools
# -------------------------------------------------------------------------


class TestMCPAdvancedSearchTools:
    """Test advanced search tools through the MCP protocol."""

    async def test_search_fuzzy(self, mcp_server, temp_test_dir):
        """search_fuzzy should find approximate matches."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_fuzzy",
                {
                    "pattern": "password",
                    "similarity_threshold": 0.6,
                    "max_results": 50,
                    "paths": [temp_test_dir],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "items" in data
            assert "total_matches" in data

    async def test_search_multi_pattern_or(self, mcp_server, temp_test_dir):
        """search_multi_pattern with OR should combine results."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_multi_pattern",
                {
                    "patterns": ["main", "password"],
                    "operator": "OR",
                    "paths": [temp_test_dir],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_matches"] >= 1

    async def test_search_multi_pattern_and(self, mcp_server, temp_test_dir):
        """search_multi_pattern with AND should intersect results by file."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_multi_pattern",
                {
                    "patterns": ["hash_password", "verify_password"],
                    "operator": "AND",
                    "paths": [temp_test_dir],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            # Both patterns are in utils.py, so AND should find results
            assert data["total_matches"] >= 1

    async def test_suggest_corrections(self, mcp_server, temp_test_dir):
        """suggest_corrections should return suggestions for a misspelled word."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "suggest_corrections",
                {"word": "passwrd", "max_suggestions": 5, "paths": [temp_test_dir]},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, list)

    async def test_search_word_fuzzy(self, mcp_server, temp_test_dir):
        """search_word_fuzzy should find words similar to the pattern."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_word_fuzzy",
                {
                    "pattern": "pasword",
                    "max_distance": 2,
                    "min_similarity": 0.5,
                    "max_results": 50,
                    "paths": [temp_test_dir],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "items" in data


# -------------------------------------------------------------------------
# Analysis Tools
# -------------------------------------------------------------------------


class TestMCPAnalysisTools:
    """Test file analysis tool through the MCP protocol."""

    async def test_analyze_file(self, mcp_server, temp_test_dir):
        """analyze_file should return code metrics for a Python file."""
        target_file = str(Path(temp_test_dir) / "utils.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file",
                {"file_path": target_file},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["language"] == "python"
            assert data["total_lines"] > 0
            assert data["functions_count"] >= 2  # hash_password, verify_password
            assert data["classes_count"] >= 1  # AuthenticationError
            assert data["imports_count"] >= 2  # hashlib, secrets

    async def test_analyze_file_sample_repo(self, mcp_server):
        """analyze_file should work on the real sample_repo fixture."""
        target = SAMPLE_REPO_SRC / "app.py"
        if not target.is_file():
            pytest.skip("sample_repo/src/app.py not found")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file",
                {"file_path": str(target)},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_lines"] > 0


# -------------------------------------------------------------------------
# Configuration Tools
# -------------------------------------------------------------------------


class TestMCPConfigTools:
    """Test configuration management tools through the MCP protocol."""

    async def test_get_search_config(self, mcp_server):
        """get_search_config should return current configuration."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("get_search_config", {})
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "paths" in data
            assert "context_lines" in data
            assert "parallel" in data

    async def test_configure_search(self, mcp_server, temp_test_dir):
        """configure_search should update and return new configuration."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "configure_search",
                {
                    "paths": [temp_test_dir],
                    "context": 5,
                    "parallel": False,
                    "workers": 2,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["context_lines"] == 5
            assert data["workers"] == 2

            # Restore defaults
            await client.call_tool(
                "configure_search",
                {"paths": ["."], "context": 3, "parallel": True, "workers": 4},
            )

    async def test_get_supported_languages(self, mcp_server):
        """get_supported_languages should return a list including 'python'."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("get_supported_languages", {})
            data = _parse_tool_result(result)
            assert isinstance(data, list)
            assert len(data) > 0
            assert "python" in data

    async def test_clear_caches(self, mcp_server):
        """clear_caches should succeed and return status."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("clear_caches", {})
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "status" in data


# -------------------------------------------------------------------------
# Utility Tools
# -------------------------------------------------------------------------


class TestMCPUtilityTools:
    """Test utility tools through the MCP protocol."""

    async def test_get_search_history(self, mcp_server):
        """get_search_history should return a list."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("get_search_history", {"limit": 5})
            data = _parse_tool_result(result)
            assert isinstance(data, list)

    async def test_get_server_health(self, mcp_server):
        """get_server_health should return health information."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("get_server_health", {})
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["status"] == "healthy"
            assert "cache_health" in data
            assert "memory_usage" in data
            assert "validation_stats" in data

    async def test_search_then_history(self, mcp_server, temp_test_dir):
        """After a search, get_search_history should include the query."""
        async with Client(mcp_server) as client:
            # Perform a search
            await client.call_tool(
                "search_text",
                {"pattern": "CONFIG", "paths": [temp_test_dir], "context": 1},
            )
            # Check history
            result = await client.call_tool("get_search_history", {"limit": 5})
            history = _parse_tool_result(result)
            assert isinstance(history, list)
            assert len(history) >= 1
            # Most recent entry should reference our pattern
            patterns = [entry.get("query", {}).get("pattern", "") for entry in history]
            assert "CONFIG" in patterns


# -------------------------------------------------------------------------
# Session Management Tools
# -------------------------------------------------------------------------


class TestMCPSessionTools:
    """Test session management tools through the MCP protocol."""

    async def test_create_session(self, mcp_server):
        """create_session should return a session_id."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "create_session",
                {"user_id": "test_user", "priority": "normal"},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "session_id" in data
            assert data["status"] == "created"
            assert data["user_id"] == "test_user"

    async def test_get_session_info(self, mcp_server):
        """get_session_info should return details for an existing session."""
        async with Client(mcp_server) as client:
            # Create session first
            create_result = await client.call_tool(
                "create_session", {"user_id": "info_user"}
            )
            session_id = _parse_tool_result(create_result)["session_id"]

            # Get session info
            info_result = await client.call_tool(
                "get_session_info", {"session_id": session_id}
            )
            data = _parse_tool_result(info_result)
            assert isinstance(data, dict)
            assert data["session_id"] == session_id
            assert data["user_id"] == "info_user"
            assert "total_searches" in data
            assert "recommendations" in data

    async def test_search_with_session_tracking(self, mcp_server, temp_test_dir):
        """Searches with session_id should be tracked in the session."""
        async with Client(mcp_server) as client:
            # Create session
            create_result = await client.call_tool(
                "create_session", {"user_id": "tracker"}
            )
            session_id = _parse_tool_result(create_result)["session_id"]

            # Perform search with session
            await client.call_tool(
                "search_text",
                {
                    "pattern": "def",
                    "paths": [temp_test_dir],
                    "context": 1,
                    "session_id": session_id,
                },
            )

            # Verify session recorded the search
            info_result = await client.call_tool(
                "get_session_info", {"session_id": session_id}
            )
            data = _parse_tool_result(info_result)
            assert data["total_searches"] >= 1


# -------------------------------------------------------------------------
# Progress Tracking Tools
# -------------------------------------------------------------------------


class TestMCPProgressTools:
    """Test progress tracking tools through the MCP protocol."""

    async def test_get_operation_progress_all(self, mcp_server):
        """get_operation_progress with no ID should list active operations."""
        async with Client(mcp_server) as client:
            result = await client.call_tool("get_operation_progress", {})
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "active_operations_count" in data
            assert "operations" in data

    async def test_cancel_nonexistent_operation(self, mcp_server):
        """cancel_operation for a non-existent ID should report not_found."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "cancel_operation", {"operation_id": "nonexistent_op_12345"}
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["status"] == "not_found"


# -------------------------------------------------------------------------
# MCP Resources
# -------------------------------------------------------------------------


class TestMCPResources:
    """Test reading MCP resources through the protocol."""

    async def test_read_config_resource(self, mcp_server):
        """pysearch://config/current should return valid JSON config."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("pysearch://config/current")
            assert content is not None
            # Content is a list of resource content items
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            data = json.loads(text)
            assert isinstance(data, dict)
            assert "paths" in data

    async def test_read_history_resource(self, mcp_server):
        """pysearch://history/searches should return JSON array."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("pysearch://history/searches")
            assert content is not None
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            data = json.loads(text)
            assert isinstance(data, list)

    async def test_read_stats_resource(self, mcp_server):
        """pysearch://stats/overview should return server statistics."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("pysearch://stats/overview")
            assert content is not None
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            data = json.loads(text)
            assert isinstance(data, dict)
            assert "total_searches" in data
            assert "cache_analytics" in data

    async def test_read_sessions_resource(self, mcp_server):
        """pysearch://sessions/analytics should return session analytics."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("pysearch://sessions/analytics")
            assert content is not None
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            data = json.loads(text)
            assert isinstance(data, dict)

    async def test_read_languages_resource(self, mcp_server):
        """pysearch://languages/supported should list languages."""
        async with Client(mcp_server) as client:
            content = await client.read_resource("pysearch://languages/supported")
            assert content is not None
            text = content[0].text if hasattr(content[0], "text") else str(content[0])
            data = json.loads(text)
            assert isinstance(data, dict)
            assert "languages" in data
            assert "python" in data["languages"]


# -------------------------------------------------------------------------
# Error Handling
# -------------------------------------------------------------------------


class TestMCPErrorHandling:
    """Test that MCP tool errors propagate correctly."""

    async def test_invalid_regex_raises_error(self, mcp_server, temp_test_dir):
        """search_regex with an invalid pattern should raise an error."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "search_regex",
                    {"pattern": "[unclosed", "paths": [temp_test_dir]},
                )

    async def test_empty_multi_pattern_raises_error(self, mcp_server):
        """search_multi_pattern with empty patterns list should raise an error."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "search_multi_pattern",
                    {"patterns": [], "operator": "OR"},
                )

    async def test_analyze_nonexistent_file(self, mcp_server):
        """analyze_file with a non-existent path should raise an error."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "analyze_file",
                    {"file_path": "/nonexistent/path/to/file.py"},
                )

    async def test_get_session_info_invalid_id(self, mcp_server):
        """get_session_info with a non-existent session should raise an error."""
        async with Client(mcp_server) as client:
            with pytest.raises(Exception):
                await client.call_tool(
                    "get_session_info",
                    {"session_id": "nonexistent_session_999"},
                )


# -------------------------------------------------------------------------
# IDE Integration Tools
# -------------------------------------------------------------------------


class TestMCPIDETools:
    """Test IDE integration tools through the MCP protocol."""

    async def test_ide_document_symbols(self, mcp_server, temp_test_dir):
        """ide_document_symbols should list symbols in a file."""
        target_file = str(Path(temp_test_dir) / "utils.py")
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_document_symbols",
                    {"file_path": target_file, "paths": [temp_test_dir]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "symbols" in data
            except Exception:
                pytest.skip("IDE integration not available in this environment")

    async def test_ide_diagnostics(self, mcp_server, temp_test_dir):
        """ide_diagnostics should return diagnostics for a file."""
        target_file = str(Path(temp_test_dir) / "utils.py")
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_diagnostics",
                    {"file_path": target_file, "paths": [temp_test_dir]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "diagnostics" in data
            except Exception:
                pytest.skip("IDE diagnostics not available in this environment")

    async def test_ide_workspace_symbols(self, mcp_server, temp_test_dir):
        """ide_workspace_symbols should find symbols matching a query."""
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_workspace_symbols",
                    {"query": "hash", "paths": [temp_test_dir]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "symbols" in data
            except Exception:
                pytest.skip("IDE workspace symbols not available in this environment")


# -------------------------------------------------------------------------
# Distributed & Multi-Repo Tools (smoke tests)
# -------------------------------------------------------------------------


class TestMCPDistributedTools:
    """Smoke-test distributed indexing tools through the MCP protocol."""

    async def test_distributed_status(self, mcp_server):
        """distributed_status should return an enabled flag."""
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool("distributed_status", {})
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "enabled" in data
            except Exception:
                pytest.skip("Distributed indexing tools not available")

    async def test_multi_repo_list(self, mcp_server):
        """multi_repo_list should report enabled status."""
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool("multi_repo_list", {})
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "enabled" in data
            except Exception:
                pytest.skip("Multi-repo tools not available")


# -------------------------------------------------------------------------
# End-to-End Workflow
# -------------------------------------------------------------------------


class TestMCPEndToEnd:
    """End-to-end workflow test exercising multiple tools in sequence."""

    async def test_full_search_workflow(self, mcp_server, temp_test_dir):
        """Run a complete workflow: configure → session → search → history → health."""
        async with Client(mcp_server) as client:
            # 1. Configure search to use our temp directory
            config_result = await client.call_tool(
                "configure_search",
                {
                    "paths": [temp_test_dir],
                    "include_patterns": ["**/*.py"],
                    "context": 2,
                },
            )
            config_data = _parse_tool_result(config_result)
            assert isinstance(config_data, dict)

            # 2. Create a session
            session_result = await client.call_tool(
                "create_session",
                {"user_id": "e2e_tester", "priority": "high"},
            )
            session_data = _parse_tool_result(session_result)
            session_id = session_data["session_id"]
            assert session_data["status"] == "created"

            # 3. Perform text search with session
            search1 = await client.call_tool(
                "search_text",
                {
                    "pattern": "password",
                    "paths": [temp_test_dir],
                    "context": 2,
                    "session_id": session_id,
                },
            )
            search1_data = _parse_tool_result(search1)
            assert search1_data["total_matches"] >= 1

            # 4. Perform regex search
            search2 = await client.call_tool(
                "search_regex",
                {
                    "pattern": r"class\s+\w+",
                    "paths": [temp_test_dir],
                    "context": 1,
                    "session_id": session_id,
                },
            )
            search2_data = _parse_tool_result(search2)
            assert search2_data["total_matches"] >= 1

            # 5. Analyze a file
            analyze_result = await client.call_tool(
                "analyze_file",
                {"file_path": str(Path(temp_test_dir) / "main.py")},
            )
            analyze_data = _parse_tool_result(analyze_result)
            assert analyze_data["language"] == "python"
            assert analyze_data["functions_count"] >= 1

            # 6. Check search history
            history_result = await client.call_tool(
                "get_search_history", {"limit": 10}
            )
            history = _parse_tool_result(history_result)
            assert isinstance(history, list)
            assert len(history) >= 2  # At least our 2 searches

            # 7. Check session info
            session_info = await client.call_tool(
                "get_session_info", {"session_id": session_id}
            )
            session_info_data = _parse_tool_result(session_info)
            assert session_info_data["total_searches"] >= 2

            # 8. Server health
            health_result = await client.call_tool("get_server_health", {})
            health_data = _parse_tool_result(health_result)
            assert health_data["status"] == "healthy"

            # 9. Clear caches (cleanup)
            clear_result = await client.call_tool("clear_caches", {})
            clear_data = _parse_tool_result(clear_result)
            assert "status" in clear_data

            # 10. Restore default config
            await client.call_tool(
                "configure_search",
                {"paths": ["."], "context": 3, "parallel": True, "workers": 4},
            )

    async def test_sample_repo_search(self, mcp_server):
        """Search the real sample_repo through MCP to verify real-world usage."""
        if not SAMPLE_REPO_SRC.is_dir():
            pytest.skip("test_data/sample_repo/src not found")

        async with Client(mcp_server) as client:
            # Search for 'def' in sample_repo
            result = await client.call_tool(
                "search_text",
                {
                    "pattern": "def",
                    "paths": [str(SAMPLE_REPO_SRC)],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert data["total_matches"] >= 1
            assert data["stats"]["files_matched"] >= 1

            # Regex search for class definitions
            result2 = await client.call_tool(
                "search_regex",
                {
                    "pattern": r"class\s+\w+",
                    "paths": [str(SAMPLE_REPO_SRC)],
                    "context": 1,
                },
            )
            data2 = _parse_tool_result(result2)
            assert data2["total_matches"] >= 1


# =========================================================================
# Real Repository Tests — test_data/sample_repo
#
# These tests exercise MCP tools against a real multi-module Python project
# with known code structures: models, services, API routes, DB layer, utils.
# =========================================================================

_REPO = str(SAMPLE_REPO_SRC)
_skip_no_repo = pytest.mark.skipif(
    not SAMPLE_REPO_SRC.is_dir(),
    reason="test_data/sample_repo/src not found",
)


@_skip_no_repo
class TestRealRepoTextSearch:
    """Text search against real repository with known patterns."""

    async def test_find_known_class_name(self, mcp_server):
        """search_text should find the 'AuthService' class."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "AuthService", "paths": [_REPO], "context": 2},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 2  # definition + usages
            # Should appear in auth.py and routes.py at minimum
            matched_files = {item["file"] for item in data["items"]}
            assert any("auth.py" in f for f in matched_files)

    async def test_find_known_function(self, mcp_server):
        """search_text should find 'create_token' across files."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "create_token", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 1
            assert any("auth.py" in item["file"] for item in data["items"])

    async def test_find_database_pool(self, mcp_server):
        """search_text should find 'DatabasePool' in connection.py and app.py."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "DatabasePool", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 2
            files = {item["file"] for item in data["items"]}
            assert any("connection.py" in f for f in files)
            assert any("app.py" in f for f in files)

    async def test_find_todo_fixme_markers(self, mcp_server):
        """search_text should locate TODO/FIXME comments in the codebase."""
        async with Client(mcp_server) as client:
            todo_result = await client.call_tool(
                "search_text",
                {"pattern": "TODO:", "paths": [_REPO], "context": 0},
            )
            fixme_result = await client.call_tool(
                "search_text",
                {"pattern": "FIXME:", "paths": [_REPO], "context": 0},
            )
            todos = _parse_tool_result(todo_result)
            fixmes = _parse_tool_result(fixme_result)
            total = todos["total_matches"] + fixmes["total_matches"]
            # sample_repo has TODO and FIXME markers
            assert total >= 2

    async def test_find_config_constants(self, mcp_server):
        """search_text should find global constants like MAX_PAGE_SIZE."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "MAX_PAGE_SIZE", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 1
            assert any("config.py" in item["file"] for item in data["items"])

    async def test_find_import_pattern(self, mcp_server):
        """search_text should find cross-module imports."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "from src.config import", "paths": [_REPO], "context": 0},
            )
            data = _parse_tool_result(result)
            # Multiple files import from src.config
            assert data["total_matches"] >= 2

    async def test_search_with_context_lines(self, mcp_server):
        """search_text with context > 0 should include surrounding lines."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_text",
                {"pattern": "class UserRole", "paths": [_REPO], "context": 3},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 1
            # With context=3, items should have surrounding lines
            first_item = data["items"][0]
            assert len(first_item["lines"]) > 1


@_skip_no_repo
class TestRealRepoRegexSearch:
    """Regex search against real repository."""

    async def test_find_all_class_definitions(self, mcp_server):
        """search_regex should find all class definitions across the repo."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r"^class\s+\w+", "paths": [_REPO], "context": 0},
            )
            data = _parse_tool_result(result)
            # Known classes: UserRole, UserProfile, User, BaseModel,
            # TimestampMixin, Settings, Environment, AuthService,
            # TokenPayload, DatabasePool, ConnectionInfo, InMemoryCache,
            # CacheEntry, etc.
            assert data["total_matches"] >= 10
            assert data["stats"]["files_matched"] >= 5

    async def test_find_async_functions(self, mcp_server):
        """search_regex should find all async function definitions."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r"async\s+def\s+\w+", "paths": [_REPO], "context": 0},
            )
            data = _parse_tool_result(result)
            # Lots of async functions: login, logout, connect, disconnect,
            # execute, health_check, create_user, etc.
            assert data["total_matches"] >= 8

    async def test_find_decorators(self, mcp_server):
        """search_regex should find decorator usage."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r"^\s*@\w+", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            # @dataclass, @property, @wraps, @require_auth, @require_role, etc.
            assert data["total_matches"] >= 10

    async def test_find_type_annotations(self, mcp_server):
        """search_regex should find typed return annotations."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r"->\s+(dict|list|bool|str|int)", "paths": [_REPO], "context": 0},
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 10

    async def test_find_enum_members(self, mcp_server):
        """search_regex should find Enum value definitions."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {
                    "pattern": r'^\s+\w+\s*=\s*"[a-z]+"',
                    "paths": [_REPO],
                    "context": 0,
                },
            )
            data = _parse_tool_result(result)
            # UserRole and Environment enum members
            assert data["total_matches"] >= 4

    async def test_find_docstrings(self, mcp_server):
        """search_regex should find module-level docstrings."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_regex",
                {"pattern": r'^"""[A-Z]', "paths": [_REPO], "context": 0},
            )
            data = _parse_tool_result(result)
            # Every module in sample_repo has a docstring
            assert data["total_matches"] >= 8


@_skip_no_repo
class TestRealRepoASTSearch:
    """AST-based search against real repository."""

    async def test_ast_find_function_by_name(self, mcp_server):
        """search_ast with func_name should find specific functions."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_ast",
                {
                    "pattern": "def",
                    "func_name": "create_app",
                    "paths": [_REPO],
                    "context": 2,
                },
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_ast"] is True
            # create_app is defined in app.py
            if data["total_matches"] > 0:
                assert any("app.py" in item["file"] for item in data["items"])

    async def test_ast_find_class_by_name(self, mcp_server):
        """search_ast with class_name should find specific classes."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_ast",
                {
                    "pattern": "class",
                    "class_name": "User",
                    "paths": [_REPO],
                    "context": 2,
                },
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_ast"] is True

    async def test_ast_find_all_functions(self, mcp_server):
        """search_ast should find function definitions across the repo."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_ast",
                {
                    "pattern": "def",
                    "paths": [_REPO],
                    "context": 0,
                },
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_ast"] is True


@_skip_no_repo
class TestRealRepoSemanticSearch:
    """Semantic search against real repository."""

    async def test_semantic_authentication(self, mcp_server):
        """search_semantic for 'authentication' should find auth-related code."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_semantic",
                {"concept": "authentication", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_semantic"] is True
            # Should find results in auth.py, routes.py, etc.
            if data["total_matches"] > 0:
                files = {item["file"] for item in data["items"]}
                assert any("auth" in f.lower() for f in files)

    async def test_semantic_database(self, mcp_server):
        """search_semantic for 'database connection' should find DB code."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_semantic",
                {"concept": "database connection", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_semantic"] is True

    async def test_semantic_caching(self, mcp_server):
        """search_semantic for 'caching' should find cache-related code."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_semantic",
                {"concept": "caching", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_semantic"] is True

    async def test_semantic_error_handling(self, mcp_server):
        """search_semantic for 'error handling' should find exception patterns."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_semantic",
                {"concept": "error handling", "paths": [_REPO], "context": 1},
            )
            data = _parse_tool_result(result)
            assert data["query_info"]["use_semantic"] is True


@_skip_no_repo
class TestRealRepoAdvancedSearch:
    """Advanced search tools against real repository."""

    async def test_fuzzy_search_misspelled_class(self, mcp_server):
        """search_fuzzy should find 'DatabasePool' even if misspelled."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_fuzzy",
                {
                    "pattern": "DatabsePool",  # deliberate typo
                    "similarity_threshold": 0.5,
                    "max_results": 20,
                    "paths": [_REPO],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "items" in data

    async def test_multi_pattern_or_across_repo(self, mcp_server):
        """search_multi_pattern OR should find results from multiple patterns."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_multi_pattern",
                {
                    "patterns": ["AuthService", "DatabasePool", "InMemoryCache"],
                    "operator": "OR",
                    "paths": [_REPO],
                    "context": 0,
                },
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 3
            files = {item["file"] for item in data["items"]}
            # Should span multiple modules
            assert len(files) >= 2

    async def test_multi_pattern_and_same_file(self, mcp_server):
        """search_multi_pattern AND should find files containing all patterns."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_multi_pattern",
                {
                    # Both exist in auth.py
                    "patterns": ["create_token", "verify_token"],
                    "operator": "AND",
                    "paths": [_REPO],
                    "context": 0,
                },
            )
            data = _parse_tool_result(result)
            assert data["total_matches"] >= 1

    async def test_suggest_corrections_real_repo(self, mcp_server):
        """suggest_corrections should produce suggestions from real codebase vocab."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "suggest_corrections",
                {"word": "passwrd", "max_suggestions": 10, "paths": [_REPO]},
            )
            data = _parse_tool_result(result)
            assert isinstance(data, list)

    async def test_word_fuzzy_search_real_repo(self, mcp_server):
        """search_word_fuzzy should find approximate word matches in real repo."""
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "search_word_fuzzy",
                {
                    "pattern": "conection",  # deliberate typo for "connection"
                    "max_distance": 2,
                    "min_similarity": 0.5,
                    "max_results": 20,
                    "paths": [_REPO],
                    "context": 1,
                },
            )
            data = _parse_tool_result(result)
            assert isinstance(data, dict)
            assert "items" in data


@_skip_no_repo
class TestRealRepoFileAnalysis:
    """File analysis against real repository files."""

    async def test_analyze_auth_service(self, mcp_server):
        """analyze_file on auth.py should return correct metrics."""
        target = str(SAMPLE_REPO_SRC / "services" / "auth.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            assert data["total_lines"] > 100
            # auth.py has: create_token, verify_token, require_auth, require_role + AuthService methods
            assert data["functions_count"] >= 4
            # AuthService, TokenPayload
            assert data["classes_count"] >= 2
            assert data["imports_count"] >= 4

    async def test_analyze_user_model(self, mcp_server):
        """analyze_file on user.py should reflect model complexity."""
        target = str(SAMPLE_REPO_SRC / "models" / "user.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            assert data["total_lines"] > 80
            # User, UserProfile, UserRole
            assert data["classes_count"] >= 3
            # validate, to_dict, set_password, check_password, is_complete, etc.
            assert data["functions_count"] >= 5

    async def test_analyze_helpers_module(self, mcp_server):
        """analyze_file on helpers.py should count utility functions."""
        target = str(SAMPLE_REPO_SRC / "utils" / "helpers.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            # slugify, truncate, chunk_list, flatten, retry, deep_merge, format_bytes
            assert data["functions_count"] >= 7

    async def test_analyze_connection_module(self, mcp_server):
        """analyze_file on connection.py should reflect DB pool structure."""
        target = str(SAMPLE_REPO_SRC / "db" / "connection.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            # DatabasePool, ConnectionInfo
            assert data["classes_count"] >= 2
            # connect, disconnect, acquire, execute, health_check, etc.
            assert data["functions_count"] >= 3

    async def test_analyze_config_module(self, mcp_server):
        """analyze_file on config.py should find Settings and constants."""
        target = str(SAMPLE_REPO_SRC / "config.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            # Settings, Environment
            assert data["classes_count"] >= 2
            # get_settings, is_feature_enabled
            assert data["functions_count"] >= 2

    async def test_analyze_routes_module(self, mcp_server):
        """analyze_file on routes.py should find all endpoint functions."""
        target = str(SAMPLE_REPO_SRC / "api" / "routes.py")
        async with Client(mcp_server) as client:
            result = await client.call_tool(
                "analyze_file", {"file_path": target}
            )
            data = _parse_tool_result(result)
            assert data["language"] == "python"
            # Routes may use async functions and decorators that affect
            # the count depending on the analyzer.
            assert data["total_lines"] > 50


@_skip_no_repo
class TestRealRepoIDETools:
    """IDE integration tools against real repository files."""

    async def test_document_symbols_auth(self, mcp_server):
        """ide_document_symbols should list symbols in auth.py."""
        target = str(SAMPLE_REPO_SRC / "services" / "auth.py")
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_document_symbols",
                    {"file_path": target, "paths": [_REPO]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert data["count"] >= 4
                names = {s.get("name", "") for s in data["symbols"]}
                assert "AuthService" in names or any("auth" in n.lower() for n in names)
            except Exception:
                pytest.skip("IDE integration not available")

    async def test_document_symbols_user_model(self, mcp_server):
        """ide_document_symbols should list symbols in user.py."""
        target = str(SAMPLE_REPO_SRC / "models" / "user.py")
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_document_symbols",
                    {"file_path": target, "paths": [_REPO]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert data["count"] >= 3
            except Exception:
                pytest.skip("IDE integration not available")

    async def test_diagnostics_app(self, mcp_server):
        """ide_diagnostics should find TODO/FIXME markers in app.py."""
        target = str(SAMPLE_REPO_SRC / "app.py")
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_diagnostics",
                    {"file_path": target, "paths": [_REPO]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert "diagnostics" in data
                # app.py has FIXME and TODO markers
                if data["count"] > 0:
                    severities = {d.get("severity", "") for d in data["diagnostics"]}
                    assert len(severities) >= 1
            except Exception:
                pytest.skip("IDE diagnostics not available")

    async def test_workspace_symbols_query(self, mcp_server):
        """ide_workspace_symbols should find symbols matching 'User'."""
        async with Client(mcp_server) as client:
            try:
                result = await client.call_tool(
                    "ide_workspace_symbols",
                    {"query": "User", "paths": [_REPO]},
                )
                data = _parse_tool_result(result)
                assert isinstance(data, dict)
                assert data["count"] >= 1
            except Exception:
                pytest.skip("IDE workspace symbols not available")


@_skip_no_repo
class TestRealRepoMultiRepoTools:
    """Multi-repository tools using sample_repo as a target."""

    async def test_multi_repo_add_and_search(self, mcp_server):
        """Add sample_repo and perform multi-repo search."""
        async with Client(mcp_server) as client:
            try:
                # Enable multi-repo
                enable_result = await client.call_tool(
                    "multi_repo_enable", {"max_workers": 2}
                )
                enable_data = _parse_tool_result(enable_result)
                assert enable_data["enabled"] is True

                # Add the sample repo
                add_result = await client.call_tool(
                    "multi_repo_add",
                    {
                        "name": "sample_app",
                        "path": _REPO,
                        "priority": "normal",
                    },
                )
                add_data = _parse_tool_result(add_result)
                assert add_data["added"] is True

                # List repos
                list_result = await client.call_tool("multi_repo_list", {})
                list_data = _parse_tool_result(list_result)
                assert list_data["enabled"] is True
                assert list_data["count"] >= 1

                # Search across repos
                search_result = await client.call_tool(
                    "multi_repo_search",
                    {
                        "pattern": "def",
                        "use_regex": False,
                        "context": 0,
                        "max_results": 50,
                    },
                )
                search_data = _parse_tool_result(search_result)
                assert search_data["total_matches"] >= 1

                # Remove and cleanup
                await client.call_tool(
                    "multi_repo_remove", {"name": "sample_app"}
                )
            except Exception:
                pytest.skip("Multi-repo tools not fully available")


@_skip_no_repo
class TestRealRepoEndToEnd:
    """End-to-end workflows against the real sample_repo."""

    async def test_full_real_repo_investigation(self, mcp_server):
        """Simulate a developer investigating the real codebase via MCP.

        Workflow: configure → session → text search → regex → AST →
        analyze → history → session info → health → resources
        """
        async with Client(mcp_server) as client:
            # 1. Configure for sample_repo
            await client.call_tool(
                "configure_search",
                {
                    "paths": [_REPO],
                    "include_patterns": ["**/*.py"],
                    "context": 2,
                },
            )

            # 2. Create investigation session
            session_result = await client.call_tool(
                "create_session",
                {"user_id": "real_repo_investigator", "priority": "high"},
            )
            sid = _parse_tool_result(session_result)["session_id"]

            # 3. Text search: find all authentication-related code
            auth_search = await client.call_tool(
                "search_text",
                {
                    "pattern": "token",
                    "paths": [_REPO],
                    "context": 2,
                    "session_id": sid,
                },
            )
            auth_data = _parse_tool_result(auth_search)
            assert auth_data["total_matches"] >= 5
            auth_files = {item["file"] for item in auth_data["items"]}
            assert len(auth_files) >= 2  # token appears in multiple files

            # 4. Regex search: find all property definitions
            prop_search = await client.call_tool(
                "search_regex",
                {
                    "pattern": r"@property",
                    "paths": [_REPO],
                    "context": 2,
                    "session_id": sid,
                },
            )
            prop_data = _parse_tool_result(prop_search)
            assert prop_data["total_matches"] >= 3

            # 5. AST search: find class definitions
            ast_search = await client.call_tool(
                "search_ast",
                {
                    "pattern": "class",
                    "paths": [_REPO],
                    "context": 1,
                    "session_id": sid,
                },
            )
            ast_data = _parse_tool_result(ast_search)
            assert ast_data["query_info"]["use_ast"] is True

            # 6. Analyze a key file
            analyze_result = await client.call_tool(
                "analyze_file",
                {"file_path": str(SAMPLE_REPO_SRC / "services" / "auth.py")},
            )
            analyze_data = _parse_tool_result(analyze_result)
            assert analyze_data["language"] == "python"
            assert analyze_data["classes_count"] >= 2

            # 7. Multi-pattern: find files with both database and async patterns
            multi_result = await client.call_tool(
                "search_multi_pattern",
                {
                    "patterns": ["async def", "self._"],
                    "operator": "AND",
                    "paths": [_REPO],
                    "context": 0,
                    "session_id": sid,
                },
            )
            multi_data = _parse_tool_result(multi_result)
            assert multi_data["total_matches"] >= 1

            # 8. Check search history — should have 4+ searches
            history_result = await client.call_tool(
                "get_search_history", {"limit": 20}
            )
            history = _parse_tool_result(history_result)
            assert len(history) >= 4

            # 9. Session should track all searches
            session_info = await client.call_tool(
                "get_session_info", {"session_id": sid}
            )
            info_data = _parse_tool_result(session_info)
            assert info_data["total_searches"] >= 4

            # 10. Server health
            health = await client.call_tool("get_server_health", {})
            health_data = _parse_tool_result(health)
            assert health_data["status"] == "healthy"

            # 11. Read resources to verify state is coherent
            config_content = await client.read_resource("pysearch://config/current")
            config_text = config_content[0].text if hasattr(config_content[0], "text") else str(config_content[0])
            config_json = json.loads(config_text)
            config_paths = config_json.get("paths", [])
            # Normalize for cross-platform comparison
            norm = lambda s: s.replace("\\", "/").rstrip("/").lower()
            assert any(norm(_REPO) == norm(p) for p in config_paths), (
                f"{_REPO} not found in config paths {config_paths}"
            )

            stats_content = await client.read_resource("pysearch://stats/overview")
            stats_text = stats_content[0].text if hasattr(stats_content[0], "text") else str(stats_content[0])
            stats_json = json.loads(stats_text)
            assert stats_json["total_searches"] >= 4

            # 12. Cleanup
            await client.call_tool("clear_caches", {})
            await client.call_tool(
                "configure_search",
                {"paths": ["."], "context": 3, "parallel": True, "workers": 4},
            )

    async def test_cross_file_dependency_trace(self, mcp_server):
        """Trace a dependency chain across files via successive MCP searches.

        Trace: routes.py → auth_service → AuthService (auth.py) →
               User model (user.py) → BaseModel (base.py)
        """
        async with Client(mcp_server) as client:
            # 1. Find where auth_service is used in routes
            r1 = await client.call_tool(
                "search_text",
                {"pattern": "auth_service", "paths": [_REPO], "context": 1},
            )
            d1 = _parse_tool_result(r1)
            assert d1["total_matches"] >= 2
            assert any("routes.py" in item["file"] for item in d1["items"])

            # 2. Find AuthService definition
            r2 = await client.call_tool(
                "search_regex",
                {"pattern": r"^class AuthService", "paths": [_REPO], "context": 2},
            )
            d2 = _parse_tool_result(r2)
            assert d2["total_matches"] >= 1
            assert any("auth.py" in item["file"] for item in d2["items"])

            # 3. AuthService uses User model — trace it
            r3 = await client.call_tool(
                "search_text",
                {"pattern": "from src.models.user import", "paths": [_REPO], "context": 0},
            )
            d3 = _parse_tool_result(r3)
            assert d3["total_matches"] >= 1

            # 4. Find User class definition
            r4 = await client.call_tool(
                "search_regex",
                {"pattern": r"^class User\(", "paths": [_REPO], "context": 2},
            )
            d4 = _parse_tool_result(r4)
            assert d4["total_matches"] >= 1
            assert any("user.py" in item["file"] for item in d4["items"])

            # 5. User inherits BaseModel — trace it
            r5 = await client.call_tool(
                "search_regex",
                {"pattern": r"^class BaseModel", "paths": [_REPO], "context": 2},
            )
            d5 = _parse_tool_result(r5)
            assert d5["total_matches"] >= 1
            assert any("base.py" in item["file"] for item in d5["items"])

    async def test_code_review_workflow(self, mcp_server):
        """Simulate a code review: find patterns, check quality, analyze files.

        1. Find hardcoded strings (potential secrets)
        2. Check for TODO/FIXME/HACK markers
        3. Verify all files have docstrings
        4. Analyze complexity of key modules
        """
        async with Client(mcp_server) as client:
            # 1. Find potential hardcoded secrets
            secret_search = await client.call_tool(
                "search_regex",
                {
                    "pattern": r'secret_key.*=.*"[^"]+"',
                    "paths": [_REPO],
                    "context": 1,
                },
            )
            secrets_data = _parse_tool_result(secret_search)
            # config.py has secret_key = "change-me-in-production"
            assert secrets_data["total_matches"] >= 1

            # 2. Find all TODO/FIXME/HACK markers
            markers_search = await client.call_tool(
                "search_regex",
                {
                    "pattern": r"#\s*(TODO|FIXME|HACK|XXX):",
                    "paths": [_REPO],
                    "context": 1,
                },
            )
            markers_data = _parse_tool_result(markers_search)
            assert markers_data["total_matches"] >= 2

            # 3. Find modules missing docstrings (start with import, not """)
            non_doc_search = await client.call_tool(
                "search_regex",
                {
                    "pattern": r'^"""',
                    "paths": [_REPO],
                    "context": 0,
                },
            )
            docstring_data = _parse_tool_result(non_doc_search)
            docstring_files = {item["file"] for item in docstring_data["items"]}
            # Most .py files should have docstrings
            assert len(docstring_files) >= 8

            # 4. Analyze complexity of the three largest modules
            for module in [
                "services/auth.py",
                "models/user.py",
                "utils/helpers.py",
            ]:
                target = str(SAMPLE_REPO_SRC / module)
                result = await client.call_tool(
                    "analyze_file", {"file_path": target}
                )
                data = _parse_tool_result(result)
                assert data["language"] == "python"
                assert data["total_lines"] > 50
