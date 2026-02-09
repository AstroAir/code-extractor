"""
Tests for module import compatibility and tool/resource registration.

Covers: backward-compatible imports of PySearchEngine, SearchResponse,
ConfigResponse from multiple paths, create_mcp_server importability,
all tool registration functions, and register_resources.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ---------------------------------------------------------------------------
# Import sources (module-level so they are available to all tests)
# ---------------------------------------------------------------------------

from mcp.servers import (
    ConfigResponse as ConfigResponseFromInit,
)
from mcp.servers import (
    PySearchEngine as PySearchEngineFromInit,
)
from mcp.servers import (
    SearchResponse as SearchResponseFromInit,
)
from mcp.servers import (
    create_mcp_server,
)
from mcp.servers.engine import (
    ConfigResponse as ConfigResponseFromEngine,
)
from mcp.servers.engine import (
    PySearchEngine as PySearchEngineFromEngine,
)
from mcp.servers.engine import (
    SearchResponse as SearchResponseFromEngine,
)
from mcp.servers.pysearch_mcp_server import PySearchEngine as PySearchEngineFromServer

# ---------------------------------------------------------------------------
# PySearchEngine import compatibility
# ---------------------------------------------------------------------------


class TestPySearchEngineImports:
    """Verify PySearchEngine is importable from all expected paths."""

    def test_import_from_pysearch_mcp_server(self):
        """PySearchEngine importable from pysearch_mcp_server (backward compat)."""
        assert PySearchEngineFromServer is not None
        assert PySearchEngineFromServer is PySearchEngineFromEngine

    def test_import_from_engine(self):
        """PySearchEngine importable from engine module."""
        assert PySearchEngineFromEngine is not None

    def test_import_from_init(self):
        """PySearchEngine importable from mcp.servers package."""
        assert PySearchEngineFromInit is PySearchEngineFromEngine


# ---------------------------------------------------------------------------
# SearchResponse import compatibility
# ---------------------------------------------------------------------------


class TestSearchResponseImports:
    """Verify SearchResponse is importable from all expected paths."""

    def test_import_from_engine(self):
        """SearchResponse importable from engine module."""
        assert SearchResponseFromEngine is not None

    def test_import_from_init(self):
        """SearchResponse importable from mcp.servers package."""
        assert SearchResponseFromInit is SearchResponseFromEngine


# ---------------------------------------------------------------------------
# ConfigResponse import compatibility
# ---------------------------------------------------------------------------


class TestConfigResponseImports:
    """Verify ConfigResponse is importable from all expected paths."""

    def test_import_from_engine(self):
        """ConfigResponse importable from engine module."""
        assert ConfigResponseFromEngine is not None

    def test_import_from_init(self):
        """ConfigResponse importable from mcp.servers package."""
        assert ConfigResponseFromInit is ConfigResponseFromEngine


# ---------------------------------------------------------------------------
# create_mcp_server
# ---------------------------------------------------------------------------


class TestCreateMCPServerImport:
    """Verify create_mcp_server is importable and callable."""

    def test_importable(self):
        """create_mcp_server is importable from mcp.servers."""
        assert create_mcp_server is not None

    def test_callable(self):
        """create_mcp_server is callable."""
        assert callable(create_mcp_server)


# ---------------------------------------------------------------------------
# Tool registration function imports
# ---------------------------------------------------------------------------


class TestToolRegistrationImports:
    """All tool registration functions should be importable."""

    def test_register_all_tools(self):
        from mcp.servers.tools import register_all_tools

        assert callable(register_all_tools)

    def test_register_core_search_tools(self):
        from mcp.servers.tools.core_search import register_core_search_tools

        assert callable(register_core_search_tools)

    def test_register_advanced_search_tools(self):
        from mcp.servers.tools.advanced_search import register_advanced_search_tools

        assert callable(register_advanced_search_tools)

    def test_register_analysis_tools(self):
        from mcp.servers.tools.analysis import register_analysis_tools

        assert callable(register_analysis_tools)

    def test_register_config_tools(self):
        from mcp.servers.tools.config import register_config_tools

        assert callable(register_config_tools)

    def test_register_history_tools(self):
        from mcp.servers.tools.history import register_history_tools

        assert callable(register_history_tools)

    def test_register_session_tools(self):
        from mcp.servers.tools.session import register_session_tools

        assert callable(register_session_tools)

    def test_register_progress_tools(self):
        from mcp.servers.tools.progress import register_progress_tools

        assert callable(register_progress_tools)

    def test_register_ide_tools(self):
        from mcp.servers.tools.ide import register_ide_tools

        assert callable(register_ide_tools)

    def test_register_distributed_tools(self):
        from mcp.servers.tools.distributed import register_distributed_tools

        assert callable(register_distributed_tools)

    def test_register_multi_repo_tools(self):
        from mcp.servers.tools.multi_repo import register_multi_repo_tools

        assert callable(register_multi_repo_tools)

    def test_register_workspace_tools(self):
        from mcp.servers.tools.workspace import register_workspace_tools

        assert callable(register_workspace_tools)


# ---------------------------------------------------------------------------
# Resource registration import
# ---------------------------------------------------------------------------


class TestResourceRegistrationImport:
    """register_resources should be importable from resources module."""

    def test_importable(self):
        from mcp.servers.resources import register_resources

        assert callable(register_resources)


# ---------------------------------------------------------------------------
# Shared module imports
# ---------------------------------------------------------------------------


class TestSharedModuleImports:
    """Shared modules should be importable."""

    def test_resource_manager(self):
        from mcp.shared.resource_manager import ResourceManager

        assert ResourceManager is not None

    def test_session_manager(self):
        from mcp.shared.session_manager import EnhancedSessionManager, UserProfile

        assert EnhancedSessionManager is not None
        assert UserProfile is not None

    def test_validation(self):
        from mcp.shared.validation import InputValidator

        assert InputValidator is not None

    def test_progress(self):
        from mcp.shared.progress import ProgressTracker

        assert ProgressTracker is not None
