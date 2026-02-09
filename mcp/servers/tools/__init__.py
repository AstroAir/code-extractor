"""
PySearch MCP Tool Registration Modules

Each sub-module exports a register_*_tools() function that registers
MCP tools on a FastMCP instance using a shared PySearchEngine and
validation helper.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine

from .advanced_search import register_advanced_search_tools
from .analysis import register_analysis_tools
from .config import register_config_tools
from .core_search import register_core_search_tools
from .distributed import register_distributed_tools
from .history import register_history_tools
from .ide import register_ide_tools
from .multi_repo import register_multi_repo_tools
from .progress import register_progress_tools
from .session import register_session_tools
from .workspace import register_workspace_tools


def register_all_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register all MCP tools on the given server instance."""
    register_core_search_tools(mcp, engine, _validate)
    register_advanced_search_tools(mcp, engine, _validate)
    register_analysis_tools(mcp, engine, _validate)
    register_config_tools(mcp, engine, _validate)
    register_history_tools(mcp, engine, _validate)
    register_session_tools(mcp, engine, _validate)
    register_progress_tools(mcp, engine, _validate)
    register_ide_tools(mcp, engine, _validate)
    register_distributed_tools(mcp, engine, _validate)
    register_multi_repo_tools(mcp, engine, _validate)
    register_workspace_tools(mcp, engine, _validate)


__all__ = [
    "register_all_tools",
    "register_core_search_tools",
    "register_advanced_search_tools",
    "register_analysis_tools",
    "register_config_tools",
    "register_history_tools",
    "register_session_tools",
    "register_progress_tools",
    "register_ide_tools",
    "register_distributed_tools",
    "register_multi_repo_tools",
    "register_workspace_tools",
]
