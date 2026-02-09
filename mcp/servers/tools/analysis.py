"""Analysis Tools — file content analysis with metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_analysis_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register analysis tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def analyze_file(
        file_path: str,
    ) -> dict[str, Any]:
        """
        Analyze a file for code metrics and statistics.

        Args:
            file_path: Path to the file to analyze

        Returns:
            File analysis with line counts, function/class counts, complexity indicators
        """
        _validate(file_path=file_path)
        try:
            return engine.analyze_file(file_path)
        except Exception as e:
            raise ToolError(f"File analysis failed: {e}") from e
