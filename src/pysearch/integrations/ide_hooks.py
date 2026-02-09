"""
IDE integration module for pysearch.

Provides interfaces for IDE/editor plugins to leverage pysearch capabilities:
- Hook registration for custom IDE actions (jump-to-definition, find-references, etc.)
- Structured query/response for IDE consumption
- Symbol completion based on indexed code
- Diagnostic information from dependency analysis
- Hover information for symbols
- Document symbol listing

Designed for integration via JSON-RPC, LSP adapters, or direct API calls.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from ..core.api import PySearch

from ..core.types import Query

logger = get_logger()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class HookType(str, Enum):
    """Supported IDE hook types."""

    JUMP_TO_DEFINITION = "jump_to_definition"
    FIND_REFERENCES = "find_references"
    SEARCH = "search"
    COMPLETION = "completion"
    HOVER = "hover"
    DIAGNOSTICS = "diagnostics"
    DOCUMENT_SYMBOLS = "document_symbols"
    WORKSPACE_SYMBOLS = "workspace_symbols"


@dataclass
class DefinitionLocation:
    """Location of a symbol definition."""

    file: str
    line: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None
    symbol_name: str = ""
    symbol_type: str = ""  # "function", "class", "variable", "import"

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None and v != ""}


@dataclass
class ReferenceLocation:
    """Location of a symbol reference."""

    file: str
    line: int
    column: int = 0
    context: str = ""
    is_definition: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CompletionItem:
    """An auto-completion suggestion."""

    label: str
    kind: str = "text"  # "function", "class", "variable", "module", "keyword"
    detail: str = ""
    documentation: str = ""
    insert_text: str = ""
    sort_priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if not d["insert_text"]:
            d["insert_text"] = d["label"]
        return d


@dataclass
class HoverInfo:
    """Hover information for a symbol."""

    contents: str
    language: str = "python"
    symbol_name: str = ""
    symbol_type: str = ""
    file: str = ""
    line: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


@dataclass
class DocumentSymbol:
    """A symbol within a document (function, class, variable, etc.)."""

    name: str
    kind: str  # "function", "class", "variable", "import"
    line: int
    end_line: int | None = None
    detail: str = ""
    children: list[DocumentSymbol] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
        }
        if self.end_line is not None:
            d["end_line"] = self.end_line
        if self.detail:
            d["detail"] = self.detail
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d


@dataclass
class Diagnostic:
    """A diagnostic message (error, warning, info)."""

    file: str
    line: int
    column: int = 0
    severity: str = "warning"  # "error", "warning", "info", "hint"
    message: str = ""
    source: str = "pysearch"
    code: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v}


# ---------------------------------------------------------------------------
# IDEHooks – generic hook registry
# ---------------------------------------------------------------------------


class IDEHooks:
    """
    Registry for IDE-triggered hooks.

    Allows IDE plugins to register handlers for events such as
    jump-to-definition, find-references, and custom search actions.
    Handlers are identified by a unique hook ID.

    Example::

        hooks = IDEHooks()
        hook_id = hooks.register_search_handler(
            lambda query: engine.search(query.pattern)
        )
        result = hooks.trigger_hook(hook_id, query=Query(pattern="main"))
    """

    def __init__(self) -> None:
        self._hooks: dict[str, tuple[HookType, Callable[..., Any]]] = {}
        self._lock = threading.Lock()

    # -- convenience registration helpers ----------------------------------

    def register_jump_to_definition(self, handler: Callable[..., Any]) -> str:
        """Register a jump-to-definition handler. Returns the hook ID."""
        return self._register(HookType.JUMP_TO_DEFINITION, handler)

    def register_find_references(self, handler: Callable[..., Any]) -> str:
        """Register a find-references handler. Returns the hook ID."""
        return self._register(HookType.FIND_REFERENCES, handler)

    def register_search_handler(self, handler: Callable[..., Any]) -> str:
        """Register a search handler. Returns the hook ID."""
        return self._register(HookType.SEARCH, handler)

    def register_completion_handler(self, handler: Callable[..., Any]) -> str:
        """Register a completion handler. Returns the hook ID."""
        return self._register(HookType.COMPLETION, handler)

    def register_hover_handler(self, handler: Callable[..., Any]) -> str:
        """Register a hover handler. Returns the hook ID."""
        return self._register(HookType.HOVER, handler)

    def register_diagnostics_handler(self, handler: Callable[..., Any]) -> str:
        """Register a diagnostics handler. Returns the hook ID."""
        return self._register(HookType.DIAGNOSTICS, handler)

    def register_document_symbols_handler(self, handler: Callable[..., Any]) -> str:
        """Register a document symbols handler. Returns the hook ID."""
        return self._register(HookType.DOCUMENT_SYMBOLS, handler)

    def register_workspace_symbols_handler(self, handler: Callable[..., Any]) -> str:
        """Register a workspace symbols handler. Returns the hook ID."""
        return self._register(HookType.WORKSPACE_SYMBOLS, handler)

    # -- core methods ------------------------------------------------------

    def _register(self, hook_type: HookType, handler: Callable[..., Any]) -> str:
        hook_id = f"{hook_type.value}_{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._hooks[hook_id] = (hook_type, handler)
        return hook_id

    def unregister_hook(self, hook_id: str) -> bool:
        """Remove a previously registered hook. Returns True if found."""
        with self._lock:
            return self._hooks.pop(hook_id, None) is not None

    def trigger_hook(self, hook_id: str, **kwargs: Any) -> Any:
        """
        Trigger a registered hook by ID.

        Args:
            hook_id: The hook identifier returned by a register_* method.
            **kwargs: Arguments forwarded to the handler.

        Returns:
            Whatever the handler returns, or None if the hook is missing.
        """
        with self._lock:
            entry = self._hooks.get(hook_id)
        if entry is None:
            logger.warning(f"Hook {hook_id} not found")
            return None
        _, handler = entry
        try:
            return handler(**kwargs)
        except Exception as exc:
            logger.error(f"Hook {hook_id} raised: {exc}")
            return None

    def list_hooks(self) -> list[dict[str, str]]:
        """Return metadata about all registered hooks."""
        with self._lock:
            return [
                {"hook_id": hid, "type": htype.value} for hid, (htype, _) in self._hooks.items()
            ]


# ---------------------------------------------------------------------------
# IDEIntegration – high-level IDE features backed by PySearch
# ---------------------------------------------------------------------------


class IDEIntegration:
    """
    High-level IDE integration backed by a PySearch engine instance.

    Provides structured APIs for common IDE features:
    - Jump to definition
    - Find all references
    - Auto-completion
    - Hover information
    - Document / workspace symbols
    - Diagnostics (circular dependencies, unused imports, etc.)

    Example::

        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("src/main.py", 10, "my_func")
    """

    def __init__(self, engine: PySearch) -> None:
        self.engine = engine
        self._symbol_cache: dict[str, list[dict[str, Any]]] = {}
        self._cache_ttl = 60.0  # seconds
        self._cache_timestamps: dict[str, float] = {}

    # -- jump to definition ------------------------------------------------

    def jump_to_definition(
        self,
        file_path: str,
        line: int,
        symbol: str,
    ) -> DefinitionLocation | None:
        """
        Find the definition of *symbol* as seen from *file_path:line*.

        Searches for ``def <symbol>``, ``class <symbol>``, and assignment
        patterns across the indexed codebase.

        Args:
            file_path: File requesting the jump.
            line: Line number where the symbol appears.
            symbol: The identifier to look up.

        Returns:
            DefinitionLocation if found, None otherwise.
        """
        if not symbol:
            return None

        # Search for definition patterns
        patterns = [
            (rf"def\s+{re.escape(symbol)}\s*\(", "function"),
            (rf"class\s+{re.escape(symbol)}\s*[:\(]", "class"),
            (rf"^{re.escape(symbol)}\s*=", "variable"),
        ]

        for pattern, sym_type in patterns:
            try:
                result = self.engine.search(pattern, regex=True, context=0)
                if result.items:
                    best = result.items[0]
                    return DefinitionLocation(
                        file=str(best.file),
                        line=best.start_line,
                        symbol_name=symbol,
                        symbol_type=sym_type,
                    )
            except Exception as exc:
                logger.debug(f"Definition search error for '{symbol}': {exc}")

        return None

    # -- find references ---------------------------------------------------

    def find_references(
        self,
        file_path: str,
        line: int,
        symbol: str,
        include_definition: bool = True,
    ) -> list[ReferenceLocation]:
        """
        Find all references to *symbol* across the codebase.

        Args:
            file_path: Originating file.
            line: Originating line.
            symbol: The identifier to search for.
            include_definition: Whether to include the definition itself.

        Returns:
            List of ReferenceLocation objects.
        """
        if not symbol:
            return []

        refs: list[ReferenceLocation] = []
        try:
            pattern = rf"\b{re.escape(symbol)}\b"
            result = self.engine.search(pattern, regex=True, context=0)

            for item in result.items:
                is_def = any(
                    kw in "\n".join(item.lines) for kw in (f"def {symbol}", f"class {symbol}")
                )
                if not include_definition and is_def:
                    continue

                ctx = item.lines[0].strip() if item.lines else ""
                refs.append(
                    ReferenceLocation(
                        file=str(item.file),
                        line=item.start_line,
                        context=ctx,
                        is_definition=is_def,
                    )
                )
        except Exception as exc:
            logger.error(f"Find references error for '{symbol}': {exc}")

        return refs

    # -- completion --------------------------------------------------------

    def provide_completion(
        self,
        file_path: str,
        line: int,
        column: int,
        prefix: str = "",
    ) -> list[CompletionItem]:
        """
        Provide completion suggestions for the given cursor position.

        Uses the indexed codebase to suggest functions, classes, and
        variable names matching *prefix*.

        Args:
            file_path: Current file.
            line: Cursor line.
            column: Cursor column.
            prefix: Partially typed identifier.

        Returns:
            List of CompletionItem sorted by relevance.
        """
        if not prefix or len(prefix) < 2:
            return []

        completions: list[CompletionItem] = []
        seen: set[str] = set()

        # Search for definitions matching the prefix
        def_patterns = [
            (rf"def\s+({re.escape(prefix)}\w*)\s*\(", "function"),
            (rf"class\s+({re.escape(prefix)}\w*)\s*[:\(]", "class"),
        ]

        for pattern, kind in def_patterns:
            try:
                result = self.engine.search(pattern, regex=True, context=0)
                for item in result.items:
                    for raw_line in item.lines:
                        m = re.search(pattern, raw_line)
                        if m:
                            name = m.group(1)
                            if name not in seen:
                                seen.add(name)
                                completions.append(
                                    CompletionItem(
                                        label=name,
                                        kind=kind,
                                        detail=f"{kind} in {item.file.name}",
                                        sort_priority=0 if kind == "function" else 1,
                                    )
                                )
            except Exception:
                pass

        completions.sort(key=lambda c: (c.sort_priority, c.label))
        return completions

    # -- hover -------------------------------------------------------------

    def provide_hover(
        self,
        file_path: str,
        line: int,
        column: int,
        symbol: str,
    ) -> HoverInfo | None:
        """
        Provide hover information for *symbol*.

        Returns the signature / docstring of the definition if found.

        Args:
            file_path: Current file.
            line: Cursor line.
            column: Cursor column.
            symbol: The hovered identifier.

        Returns:
            HoverInfo or None.
        """
        if not symbol:
            return None

        # Try function signature
        pattern = rf"def\s+{re.escape(symbol)}\s*\([^)]*\)"
        try:
            result = self.engine.search(pattern, regex=True, context=3)
            if result.items:
                item = result.items[0]
                signature = "\n".join(item.lines).strip()
                return HoverInfo(
                    contents=signature,
                    symbol_name=symbol,
                    symbol_type="function",
                    file=str(item.file),
                    line=item.start_line,
                )
        except Exception:
            pass

        # Try class
        pattern = rf"class\s+{re.escape(symbol)}\s*[:\(]"
        try:
            result = self.engine.search(pattern, regex=True, context=3)
            if result.items:
                item = result.items[0]
                signature = "\n".join(item.lines).strip()
                return HoverInfo(
                    contents=signature,
                    symbol_name=symbol,
                    symbol_type="class",
                    file=str(item.file),
                    line=item.start_line,
                )
        except Exception:
            pass

        return None

    # -- cache helpers -----------------------------------------------------

    def _get_cached(self, cache_key: str) -> list[dict[str, Any]] | None:
        """Return cached value if still within TTL, else None."""
        ts = self._cache_timestamps.get(cache_key)
        if ts is not None and (time.time() - ts) < self._cache_ttl:
            return self._symbol_cache.get(cache_key)
        return None

    def _set_cached(self, cache_key: str, value: list[dict[str, Any]]) -> None:
        """Store value in cache with current timestamp."""
        self._symbol_cache[cache_key] = value
        self._cache_timestamps[cache_key] = time.time()

    def invalidate_cache(self, file_path: str | None = None) -> None:
        """Invalidate cached symbols for a file, or all caches if *file_path* is None."""
        if file_path is None:
            self._symbol_cache.clear()
            self._cache_timestamps.clear()
        else:
            for key in list(self._symbol_cache):
                if key.startswith(f"doc:{file_path}") or key.startswith("ws:"):
                    self._symbol_cache.pop(key, None)
                    self._cache_timestamps.pop(key, None)

    # -- document symbols --------------------------------------------------

    def get_document_symbols(self, file_path: str) -> list[DocumentSymbol]:
        """
        List all symbols (functions, classes, variables) in *file_path*.

        Args:
            file_path: Absolute or project-relative path to the file.

        Returns:
            List of DocumentSymbol objects.
        """
        # Check cache
        cache_key = f"doc:{file_path}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [DocumentSymbol(**item) for item in cached]

        symbols: list[DocumentSymbol] = []

        try:
            path = Path(file_path)
            if not path.exists():
                return symbols

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")

            for idx, raw_line in enumerate(lines, start=1):
                stripped = raw_line.strip()
                if stripped.startswith("def "):
                    m = re.match(r"def\s+(\w+)", stripped)
                    if m:
                        symbols.append(DocumentSymbol(name=m.group(1), kind="function", line=idx))
                elif stripped.startswith("class "):
                    m = re.match(r"class\s+(\w+)", stripped)
                    if m:
                        symbols.append(DocumentSymbol(name=m.group(1), kind="class", line=idx))
                elif re.match(r"^[A-Z_][A-Z0-9_]*\s*=", stripped):
                    name = stripped.split("=", 1)[0].strip()
                    symbols.append(DocumentSymbol(name=name, kind="variable", line=idx))
        except Exception as exc:
            logger.error(f"Error listing symbols for {file_path}: {exc}")

        # Store in cache
        self._set_cached(cache_key, [s.to_dict() for s in symbols])

        return symbols

    # -- workspace symbols -------------------------------------------------

    def get_workspace_symbols(self, query: str = "") -> list[DocumentSymbol]:
        """
        Search for symbols across the entire workspace.

        When multi-repo is enabled, also searches across all registered
        repositories for a truly workspace-wide symbol lookup.

        Args:
            query: Optional filter string for symbol names.

        Returns:
            List of DocumentSymbol objects from across the workspace.
        """
        if not query or len(query) < 2:
            return []

        # Check cache
        cache_key = f"ws:{query}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return [DocumentSymbol(**item) for item in cached]

        symbols: list[DocumentSymbol] = []
        pattern = rf"(def|class)\s+(\w*{re.escape(query)}\w*)"

        # Search in the primary engine's paths
        try:
            result = self.engine.search(pattern, regex=True, context=0)
            for item in result.items:
                for raw_line in item.lines:
                    m = re.search(pattern, raw_line)
                    if m:
                        kind = "function" if m.group(1) == "def" else "class"
                        symbols.append(
                            DocumentSymbol(
                                name=m.group(2),
                                kind=kind,
                                line=item.start_line,
                                detail=str(item.file),
                            )
                        )
        except Exception as exc:
            logger.error(f"Workspace symbol search error: {exc}")

        # Also search across multi-repo repositories if enabled
        try:
            if self.engine.is_multi_repo_enabled():
                multi_result = self.engine.search_all_repositories(
                    pattern=pattern, use_regex=True, context=0
                )
                if multi_result and multi_result.repository_results:
                    seen = {(s.name, s.line, s.detail) for s in symbols}
                    for _repo_name, repo_result in multi_result.repository_results.items():
                        for item in repo_result.items:
                            for raw_line in item.lines:
                                m = re.search(pattern, raw_line)
                                if m:
                                    kind = "function" if m.group(1) == "def" else "class"
                                    key = (m.group(2), item.start_line, str(item.file))
                                    if key not in seen:
                                        seen.add(key)
                                        symbols.append(
                                            DocumentSymbol(
                                                name=m.group(2),
                                                kind=kind,
                                                line=item.start_line,
                                                detail=str(item.file),
                                            )
                                        )
        except Exception as exc:
            logger.error(f"Multi-repo workspace symbol search error: {exc}")

        # Store in cache
        self._set_cached(cache_key, [s.to_dict() for s in symbols])

        return symbols

    # -- diagnostics -------------------------------------------------------

    def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """
        Run lightweight diagnostics on *file_path*.

        Currently checks for:
        - Circular dependency warnings (via DependencyAnalyzer)
        - TODO / FIXME / HACK markers

        Args:
            file_path: The file to diagnose.

        Returns:
            List of Diagnostic objects.
        """
        diagnostics: list[Diagnostic] = []

        try:
            path = Path(file_path)
            if not path.exists():
                return diagnostics

            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.split("\n")

            # Marker detection
            marker_re = re.compile(r"#\s*(TODO|FIXME|HACK|XXX)\b(.*)", re.IGNORECASE)
            for idx, raw_line in enumerate(lines, start=1):
                m = marker_re.search(raw_line)
                if m:
                    severity = (
                        "warning" if m.group(1).upper() in ("FIXME", "HACK", "XXX") else "info"
                    )
                    diagnostics.append(
                        Diagnostic(
                            file=file_path,
                            line=idx,
                            severity=severity,
                            message=f"{m.group(1).upper()}: {m.group(2).strip()}",
                            code=m.group(1).upper(),
                        )
                    )

            # Circular dependency detection
            try:
                from ..analysis.dependency_analysis import DependencyAnalyzer

                analyzer = DependencyAnalyzer()
                imports = analyzer.analyze_file(path)
                # Flag self-imports
                module_name = path.stem
                for imp in imports:
                    if imp.module == module_name:
                        diagnostics.append(
                            Diagnostic(
                                file=file_path,
                                line=getattr(imp, "line", 0),
                                severity="warning",
                                message=f"Module imports itself: '{imp.module}'",
                                code="self_import",
                            )
                        )
            except (ImportError, Exception):
                pass  # DependencyAnalyzer not available or file not parseable

        except Exception as exc:
            logger.error(f"Diagnostics error for {file_path}: {exc}")

        return diagnostics


# ---------------------------------------------------------------------------
# Convenience function (backward-compatible public API)
# ---------------------------------------------------------------------------


def ide_query(engine: PySearch, query: Query) -> dict[str, Any]:
    """
    Structured query interface for IDE consumption.

    Returns a JSON-serialisable dict with search results and stats.
    """
    res = engine.run(query)
    return {
        "items": [
            {
                "file": str(it.file),
                "start_line": it.start_line,
                "end_line": it.end_line,
                "lines": it.lines,
                "spans": [(li, (a, b)) for li, (a, b) in it.match_spans],
            }
            for it in res.items
        ],
        "stats": asdict(res.stats),
    }
