"""IDE Integration Tools — jump-to-definition, references, completion, hover, symbols, diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_ide_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register IDE integration tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def ide_jump_to_definition(
        file_path: str,
        line: int,
        symbol: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Find the definition location of a symbol (function, class, variable).

        Args:
            file_path: File where the symbol is referenced
            line: Line number where the symbol appears
            symbol: The identifier to look up
            paths: Optional search paths (default: current config paths)

        Returns:
            Definition location with file, line, symbol_name, and symbol_type
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            result = eng.jump_to_definition(file_path, line, symbol)
            if result is None:
                return {"found": False, "message": f"No definition found for '{symbol}'"}
            return {"found": True, **result}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Jump to definition failed: {e}") from e

    @mcp.tool
    def ide_find_references(
        file_path: str,
        line: int,
        symbol: str,
        include_definition: bool = True,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Find all references to a symbol across the codebase.

        Args:
            file_path: Originating file
            line: Originating line number
            symbol: The identifier to search for
            include_definition: Whether to include the definition itself
            paths: Optional search paths

        Returns:
            List of reference locations with file, line, context, and is_definition flag
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            refs = eng.find_references(file_path, line, symbol, include_definition)
            return {"symbol": symbol, "references": refs, "count": len(refs)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Find references failed: {e}") from e

    @mcp.tool
    def ide_completion(
        file_path: str,
        line: int,
        column: int,
        prefix: str = "",
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Provide auto-completion suggestions for the given cursor position.

        Args:
            file_path: Current file
            line: Cursor line number
            column: Cursor column number
            prefix: Partially typed identifier
            paths: Optional search paths

        Returns:
            List of completion items with label, kind, and detail
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            items = eng.provide_completion(file_path, line, column, prefix)
            return {"prefix": prefix, "completions": items, "count": len(items)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Completion failed: {e}") from e

    @mcp.tool
    def ide_hover(
        file_path: str,
        line: int,
        column: int,
        symbol: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Provide hover information for a symbol (type, docstring, signature).

        Args:
            file_path: Current file
            line: Cursor line number
            column: Cursor column number
            symbol: The hovered identifier
            paths: Optional search paths

        Returns:
            Hover information with symbol_name, symbol_type, contents, and documentation
        """
        _validate(pattern=symbol, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            info = eng.provide_hover(file_path, line, column, symbol)
            if info is None:
                return {"found": False, "message": f"No hover info for '{symbol}'"}
            return {"found": True, **info}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Hover failed: {e}") from e

    @mcp.tool
    def ide_document_symbols(
        file_path: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        List all symbols (functions, classes, variables) in a file.

        Args:
            file_path: Path to the file to analyze
            paths: Optional search paths

        Returns:
            List of symbols with name, kind, and line number
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            symbols = eng.get_document_symbols(file_path)
            return {"file": file_path, "symbols": symbols, "count": len(symbols)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Document symbols failed: {e}") from e

    @mcp.tool
    def ide_workspace_symbols(
        query: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Search for symbols across the entire workspace.

        Args:
            query: Filter string for symbol names (minimum 2 characters)
            paths: Optional search paths

        Returns:
            List of matching symbols with name, kind, line, and file detail
        """
        _validate(pattern=query, paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            symbols = eng.get_workspace_symbols(query)
            return {"query": query, "symbols": symbols, "count": len(symbols)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Workspace symbols failed: {e}") from e

    @mcp.tool
    def ide_diagnostics(
        file_path: str,
        paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run lightweight diagnostics on a file (TODO/FIXME/HACK markers, circular imports).

        Args:
            file_path: The file to diagnose
            paths: Optional search paths

        Returns:
            List of diagnostics with line, severity, message, and code
        """
        _validate(paths=paths)
        try:
            eng = engine._get_engine(paths)
            if not eng.enable_ide_integration():
                raise ToolError("Failed to enable IDE integration")
            diags = eng.get_diagnostics(file_path)
            return {"file": file_path, "diagnostics": diags, "count": len(diags)}
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Diagnostics failed: {e}") from e
