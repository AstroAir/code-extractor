"""
IDE integration manager.

This module provides IDE integration capabilities for pysearch,
enabling features like jump-to-definition, find-references,
auto-completion, hover information, document symbols, and diagnostics.

Classes:
    IDEIntegrationManager: Manages IDE integration functionality

Key Features:
    - Jump to definition for symbols
    - Find all references across the codebase
    - Auto-completion suggestions based on indexed code
    - Hover information with signatures and docstrings
    - Document and workspace symbol listing
    - Lightweight diagnostics (TODO/FIXME markers, self-imports)
    - Structured query interface for IDE consumption

Example:
    Using IDE integration:
        >>> from pysearch.core.managers.ide_integration import IDEIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = IDEIntegrationManager(config)
        >>> manager.enable_ide_integration(engine)
        >>> loc = manager.jump_to_definition("src/main.py", 10, "my_func")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..config import SearchConfig

if TYPE_CHECKING:
    from ..api import PySearch


class IDEIntegrationManager:
    """Manages IDE integration functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self._ide_integration: Any = None
        self._ide_hooks: Any = None
        self._enabled = False

    def enable_ide_integration(self, engine: PySearch) -> bool:
        """
        Enable IDE integration backed by a PySearch engine.

        Args:
            engine: The PySearch engine instance to use for IDE features

        Returns:
            True if IDE integration was enabled successfully, False otherwise
        """
        if self._enabled:
            return True

        try:
            from ...integrations.ide_hooks import IDEHooks, IDEIntegration

            self._ide_integration = IDEIntegration(engine)
            self._ide_hooks = IDEHooks()

            # Auto-register built-in IDE features as hooks
            self._ide_hooks.register_jump_to_definition(
                lambda file_path, line, symbol: self._ide_integration.jump_to_definition(
                    file_path, line, symbol
                )
            )
            self._ide_hooks.register_find_references(
                lambda file_path, line, symbol, include_definition=True: (
                    self._ide_integration.find_references(
                        file_path, line, symbol, include_definition
                    )
                )
            )
            self._ide_hooks.register_search_handler(
                lambda query: engine.run(query)
            )
            self._ide_hooks.register_completion_handler(
                lambda file_path, line, column, prefix="": (
                    self._ide_integration.provide_completion(file_path, line, column, prefix)
                )
            )
            self._ide_hooks.register_hover_handler(
                lambda file_path, line, column, symbol: (
                    self._ide_integration.provide_hover(file_path, line, column, symbol)
                )
            )
            self._ide_hooks.register_diagnostics_handler(
                lambda file_path: self._ide_integration.get_diagnostics(file_path)
            )
            self._ide_hooks.register_document_symbols_handler(
                lambda file_path: self._ide_integration.get_document_symbols(file_path)
            )
            self._ide_hooks.register_workspace_symbols_handler(
                lambda query="": self._ide_integration.get_workspace_symbols(query)
            )

            self._enabled = True
            return True

        except Exception:
            return False

    def disable_ide_integration(self) -> None:
        """Disable IDE integration."""
        if not self._enabled:
            return

        self._ide_integration = None
        self._ide_hooks = None
        self._enabled = False

    def is_ide_enabled(self) -> bool:
        """Check if IDE integration is enabled."""
        return self._enabled

    def jump_to_definition(
        self, file_path: str, line: int, symbol: str
    ) -> dict[str, Any] | None:
        """
        Find the definition of a symbol.

        Args:
            file_path: File requesting the jump
            line: Line number where the symbol appears
            symbol: The identifier to look up

        Returns:
            Dictionary with definition location, or None if not found
        """
        if not self._ide_integration:
            return None

        try:
            result = self._ide_integration.jump_to_definition(file_path, line, symbol)
            return result.to_dict() if result else None
        except Exception:
            return None

    def find_references(
        self,
        file_path: str,
        line: int,
        symbol: str,
        include_definition: bool = True,
    ) -> list[dict[str, Any]]:
        """
        Find all references to a symbol across the codebase.

        Args:
            file_path: Originating file
            line: Originating line
            symbol: The identifier to search for
            include_definition: Whether to include the definition itself

        Returns:
            List of reference location dictionaries
        """
        if not self._ide_integration:
            return []

        try:
            refs = self._ide_integration.find_references(
                file_path, line, symbol, include_definition
            )
            return [ref.to_dict() for ref in refs]
        except Exception:
            return []

    def provide_completion(
        self, file_path: str, line: int, column: int, prefix: str = ""
    ) -> list[dict[str, Any]]:
        """
        Provide auto-completion suggestions.

        Args:
            file_path: Current file
            line: Cursor line
            column: Cursor column
            prefix: Partially typed identifier

        Returns:
            List of completion item dictionaries
        """
        if not self._ide_integration:
            return []

        try:
            items = self._ide_integration.provide_completion(file_path, line, column, prefix)
            return [item.to_dict() for item in items]
        except Exception:
            return []

    def provide_hover(
        self, file_path: str, line: int, column: int, symbol: str
    ) -> dict[str, Any] | None:
        """
        Provide hover information for a symbol.

        Args:
            file_path: Current file
            line: Cursor line
            column: Cursor column
            symbol: The hovered identifier

        Returns:
            Dictionary with hover information, or None
        """
        if not self._ide_integration:
            return None

        try:
            info = self._ide_integration.provide_hover(file_path, line, column, symbol)
            return info.to_dict() if info else None
        except Exception:
            return None

    def get_document_symbols(self, file_path: str) -> list[dict[str, Any]]:
        """
        List all symbols in a file.

        Args:
            file_path: Path to the file

        Returns:
            List of document symbol dictionaries
        """
        if not self._ide_integration:
            return []

        try:
            symbols = self._ide_integration.get_document_symbols(file_path)
            return [sym.to_dict() for sym in symbols]
        except Exception:
            return []

    def get_workspace_symbols(self, query: str = "") -> list[dict[str, Any]]:
        """
        Search for symbols across the entire workspace.

        Args:
            query: Filter string for symbol names

        Returns:
            List of document symbol dictionaries
        """
        if not self._ide_integration:
            return []

        try:
            symbols = self._ide_integration.get_workspace_symbols(query)
            return [sym.to_dict() for sym in symbols]
        except Exception:
            return []

    def get_diagnostics(self, file_path: str) -> list[dict[str, Any]]:
        """
        Run lightweight diagnostics on a file.

        Args:
            file_path: The file to diagnose

        Returns:
            List of diagnostic dictionaries
        """
        if not self._ide_integration:
            return []

        try:
            diagnostics = self._ide_integration.get_diagnostics(file_path)
            return [diag.to_dict() for diag in diagnostics]
        except Exception:
            return []

    def ide_query(self, query: Any) -> dict[str, Any]:
        """
        Structured query interface for IDE consumption.

        Args:
            query: A Query object for searching

        Returns:
            JSON-serialisable dict with search results and stats
        """
        if not self._ide_integration:
            return {"items": [], "stats": {}}

        try:
            from ...integrations.ide_hooks import ide_query

            return ide_query(self._ide_integration.engine, query)
        except Exception:
            return {"items": [], "stats": {}}

    def register_hook(self, hook_type: str, handler: Any) -> str | None:
        """
        Register a custom IDE hook handler.

        Args:
            hook_type: Hook type name (e.g. "search", "completion", "hover")
            handler: Callable handler for the hook

        Returns:
            Hook ID string, or None if registration failed
        """
        if not self._ide_hooks:
            return None

        try:
            from ...integrations.ide_hooks import HookType

            ht = HookType(hook_type)
            return self._ide_hooks._register(ht, handler)
        except Exception:
            return None

    def trigger_hook(self, hook_id: str, **kwargs: Any) -> Any:
        """
        Trigger a registered hook by ID.

        Args:
            hook_id: The hook identifier
            **kwargs: Arguments forwarded to the handler

        Returns:
            Whatever the handler returns, or None
        """
        if not self._ide_hooks:
            return None

        try:
            return self._ide_hooks.trigger_hook(hook_id, **kwargs)
        except Exception:
            return None

    def list_hooks(self) -> list[dict[str, str]]:
        """
        List all registered IDE hooks.

        Returns:
            List of hook metadata dictionaries
        """
        if not self._ide_hooks:
            return []

        try:
            return self._ide_hooks.list_hooks()
        except Exception:
            return []
