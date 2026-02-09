"""Tests for pysearch.integrations.ide_hooks module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from pysearch.integrations.ide_hooks import (
    CompletionItem,
    DefinitionLocation,
    Diagnostic,
    DocumentSymbol,
    HookType,
    HoverInfo,
    IDEHooks,
    IDEIntegration,
    ReferenceLocation,
    ide_query,
)

# ---------------------------------------------------------------------------
# HookType
# ---------------------------------------------------------------------------


class TestHookType:
    """Tests for HookType enum."""

    def test_values(self):
        assert HookType.JUMP_TO_DEFINITION == "jump_to_definition"
        assert HookType.FIND_REFERENCES == "find_references"
        assert HookType.SEARCH == "search"
        assert HookType.COMPLETION == "completion"
        assert HookType.HOVER == "hover"
        assert HookType.DIAGNOSTICS == "diagnostics"
        assert HookType.DOCUMENT_SYMBOLS == "document_symbols"
        assert HookType.WORKSPACE_SYMBOLS == "workspace_symbols"

    def test_is_string_enum(self):
        assert isinstance(HookType.SEARCH, str)

    def test_all_members(self):
        assert len(HookType) == 8


# ---------------------------------------------------------------------------
# DefinitionLocation
# ---------------------------------------------------------------------------


class TestDefinitionLocation:
    """Tests for DefinitionLocation dataclass."""

    def test_creation(self):
        loc = DefinitionLocation(file="test.py", line=10, column=5)
        assert loc.file == "test.py"
        assert loc.line == 10
        assert loc.column == 5

    def test_defaults(self):
        loc = DefinitionLocation(file="a.py", line=1)
        assert loc.column == 0
        assert loc.end_line is None
        assert loc.end_column is None
        assert loc.symbol_name == ""
        assert loc.symbol_type == ""

    def test_to_dict(self):
        loc = DefinitionLocation(
            file="test.py",
            line=10,
            symbol_name="main",
            symbol_type="function",
        )
        d = loc.to_dict()
        assert d["file"] == "test.py"
        assert d["line"] == 10
        assert d["symbol_name"] == "main"

    def test_to_dict_excludes_none_and_empty(self):
        loc = DefinitionLocation(file="a.py", line=1)
        d = loc.to_dict()
        assert "end_line" not in d
        assert "symbol_name" not in d
        assert "file" in d
        assert "line" in d


# ---------------------------------------------------------------------------
# ReferenceLocation
# ---------------------------------------------------------------------------


class TestReferenceLocation:
    """Tests for ReferenceLocation dataclass."""

    def test_creation(self):
        ref = ReferenceLocation(file="test.py", line=5)
        assert ref.file == "test.py"
        assert ref.line == 5

    def test_defaults(self):
        ref = ReferenceLocation(file="a.py", line=1)
        assert ref.column == 0
        assert ref.context == ""
        assert ref.is_definition is False

    def test_to_dict(self):
        ref = ReferenceLocation(file="a.py", line=1, context="x = 1")
        d = ref.to_dict()
        assert isinstance(d, dict)
        assert d["file"] == "a.py"
        assert d["context"] == "x = 1"

    def test_to_dict_includes_all_fields(self):
        ref = ReferenceLocation(
            file="b.py",
            line=10,
            column=5,
            context="y = 2",
            is_definition=True,
        )
        d = ref.to_dict()
        assert d["is_definition"] is True
        assert d["column"] == 5


# ---------------------------------------------------------------------------
# CompletionItem
# ---------------------------------------------------------------------------


class TestCompletionItem:
    """Tests for CompletionItem dataclass."""

    def test_creation(self):
        item = CompletionItem(label="main", kind="function")
        assert item.label == "main"
        assert item.kind == "function"

    def test_defaults(self):
        item = CompletionItem(label="x")
        assert item.kind == "text"
        assert item.detail == ""
        assert item.insert_text == ""
        assert item.documentation == ""
        assert item.sort_priority == 0

    def test_to_dict_uses_label_as_insert_text(self):
        item = CompletionItem(label="hello")
        d = item.to_dict()
        assert d["insert_text"] == "hello"

    def test_to_dict_preserves_explicit_insert_text(self):
        item = CompletionItem(label="hello", insert_text="hello()")
        d = item.to_dict()
        assert d["insert_text"] == "hello()"


# ---------------------------------------------------------------------------
# HoverInfo
# ---------------------------------------------------------------------------


class TestHoverInfo:
    """Tests for HoverInfo dataclass."""

    def test_creation(self):
        info = HoverInfo(contents="docs here", symbol_name="main")
        assert info.contents == "docs here"
        assert info.symbol_name == "main"

    def test_defaults(self):
        info = HoverInfo(contents="x")
        assert info.language == "python"
        assert info.symbol_type == ""

    def test_to_dict_excludes_falsy(self):
        info = HoverInfo(contents="sig", symbol_name="fn", file="a.py", line=5)
        d = info.to_dict()
        assert d["contents"] == "sig"
        assert d["symbol_name"] == "fn"
        assert d["file"] == "a.py"
        assert d["line"] == 5
        assert "symbol_type" not in d  # empty string is falsy

    def test_to_dict_minimal(self):
        info = HoverInfo(contents="x")
        d = info.to_dict()
        assert "contents" in d
        assert "language" in d  # "python" is truthy


# ---------------------------------------------------------------------------
# DocumentSymbol
# ---------------------------------------------------------------------------


class TestDocumentSymbol:
    """Tests for DocumentSymbol dataclass."""

    def test_creation(self):
        sym = DocumentSymbol(
            name="main",
            kind="function",
            line=1,
            end_line=10,
        )
        assert sym.name == "main"
        assert sym.kind == "function"
        assert sym.line == 1

    def test_defaults(self):
        sym = DocumentSymbol(name="x", kind="variable", line=1)
        assert sym.end_line is None
        assert sym.detail == ""
        assert sym.children == []

    def test_to_dict_minimal(self):
        sym = DocumentSymbol(name="foo", kind="function", line=5)
        d = sym.to_dict()
        assert d == {"name": "foo", "kind": "function", "line": 5}
        assert "end_line" not in d
        assert "detail" not in d
        assert "children" not in d

    def test_to_dict_with_children(self):
        child = DocumentSymbol(name="inner", kind="function", line=3)
        parent = DocumentSymbol(
            name="Cls",
            kind="class",
            line=1,
            end_line=10,
            detail="A class",
            children=[child],
        )
        d = parent.to_dict()
        assert d["end_line"] == 10
        assert d["detail"] == "A class"
        assert len(d["children"]) == 1
        assert d["children"][0]["name"] == "inner"


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_creation(self):
        diag = Diagnostic(
            file="test.py",
            line=5,
            message="unused variable",
            severity="warning",
        )
        assert diag.message == "unused variable"
        assert diag.severity == "warning"

    def test_defaults(self):
        diag = Diagnostic(file="a.py", line=1)
        assert diag.column == 0
        assert diag.severity == "warning"
        assert diag.message == ""
        assert diag.source == "pysearch"
        assert diag.code == ""

    def test_to_dict_excludes_falsy(self):
        diag = Diagnostic(file="a.py", line=1)
        d = diag.to_dict()
        assert d["file"] == "a.py"
        assert d["line"] == 1
        assert d["severity"] == "warning"
        assert d["source"] == "pysearch"
        assert "message" not in d  # empty
        assert "code" not in d  # empty


# ---------------------------------------------------------------------------
# IDEHooks
# ---------------------------------------------------------------------------


class TestIDEHooks:
    """Tests for IDEHooks class (hook registry)."""

    def test_init(self):
        hooks = IDEHooks()
        assert hooks.list_hooks() == []

    def test_register_search_handler(self):
        hooks = IDEHooks()
        handler = MagicMock()
        hook_id = hooks.register_search_handler(handler)
        assert isinstance(hook_id, str)
        assert hook_id.startswith("search_")
        hook_list = hooks.list_hooks()
        assert len(hook_list) == 1
        assert hook_list[0]["type"] == "search"

    def test_register_jump_to_definition(self):
        hooks = IDEHooks()
        hook_id = hooks.register_jump_to_definition(lambda: None)
        assert hook_id.startswith("jump_to_definition_")

    def test_register_find_references(self):
        hooks = IDEHooks()
        hook_id = hooks.register_find_references(lambda: None)
        assert hook_id.startswith("find_references_")

    def test_register_completion_handler(self):
        hooks = IDEHooks()
        hook_id = hooks.register_completion_handler(lambda: None)
        assert hook_id.startswith("completion_")

    def test_register_hover_handler(self):
        hooks = IDEHooks()
        hook_id = hooks.register_hover_handler(lambda: None)
        assert hook_id.startswith("hover_")

    def test_register_diagnostics_handler(self):
        hooks = IDEHooks()
        hook_id = hooks.register_diagnostics_handler(lambda: None)
        assert hook_id.startswith("diagnostics_")

    def test_unregister_hook(self):
        hooks = IDEHooks()
        handler = MagicMock()
        hook_id = hooks.register_search_handler(handler)
        assert hooks.unregister_hook(hook_id) is True
        assert hooks.list_hooks() == []

    def test_unregister_nonexistent(self):
        hooks = IDEHooks()
        assert hooks.unregister_hook("nonexistent_id") is False

    def test_trigger_hook(self):
        hooks = IDEHooks()
        handler = MagicMock(return_value=[{"result": "ok"}])
        hook_id = hooks.register_search_handler(handler)
        result = hooks.trigger_hook(hook_id, query="test")
        assert result == [{"result": "ok"}]
        handler.assert_called_once_with(query="test")

    def test_trigger_hook_missing(self):
        hooks = IDEHooks()
        result = hooks.trigger_hook("nonexistent_id")
        assert result is None

    def test_trigger_hook_handler_exception(self):
        hooks = IDEHooks()
        handler = MagicMock(side_effect=RuntimeError("boom"))
        hook_id = hooks.register_search_handler(handler)
        result = hooks.trigger_hook(hook_id)
        assert result is None

    def test_list_hooks_multiple(self):
        hooks = IDEHooks()
        hooks.register_search_handler(lambda: None)
        hooks.register_hover_handler(lambda: None)
        hooks.register_diagnostics_handler(lambda: None)
        hook_list = hooks.list_hooks()
        assert len(hook_list) == 3
        types = {h["type"] for h in hook_list}
        assert types == {"search", "hover", "diagnostics"}


# ---------------------------------------------------------------------------
# IDEIntegration
# ---------------------------------------------------------------------------


def _mock_engine_with_search_results(items=None):
    """Create a mock PySearch engine returning specified search items."""
    engine = MagicMock()
    mock_result = MagicMock()
    mock_result.items = items or []
    engine.search.return_value = mock_result
    return engine


class TestIDEIntegration:
    """Tests for IDEIntegration class."""

    def test_init(self):
        mock_engine = MagicMock()
        integration = IDEIntegration(mock_engine)
        assert integration.engine is mock_engine
        assert integration._cache_ttl == 60.0

    # -- jump_to_definition ------------------------------------------------

    def test_jump_to_definition_empty_symbol(self):
        integration = IDEIntegration(MagicMock())
        assert integration.jump_to_definition("f.py", 1, "") is None

    def test_jump_to_definition_found(self):
        item = MagicMock()
        item.file = Path("src/main.py")
        item.start_line = 10
        engine = _mock_engine_with_search_results([item])
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("caller.py", 5, "my_func")
        assert loc is not None
        assert str(Path(loc.file)) == str(Path("src/main.py"))
        assert loc.line == 10
        assert loc.symbol_name == "my_func"

    def test_jump_to_definition_not_found(self):
        engine = _mock_engine_with_search_results([])
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("f.py", 1, "missing_symbol")
        assert loc is None

    def test_jump_to_definition_engine_exception(self):
        engine = MagicMock()
        engine.search.side_effect = RuntimeError("search failed")
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("f.py", 1, "sym")
        assert loc is None

    # -- find_references ---------------------------------------------------

    def test_find_references_empty_symbol(self):
        integration = IDEIntegration(MagicMock())
        assert integration.find_references("f.py", 1, "") == []

    def test_find_references_found(self):
        item = MagicMock()
        item.file = Path("a.py")
        item.start_line = 3
        item.lines = ["x = my_func()"]
        engine = _mock_engine_with_search_results([item])
        integration = IDEIntegration(engine)
        refs = integration.find_references("b.py", 1, "my_func")
        assert len(refs) >= 1
        assert refs[0].file == "a.py"
        assert refs[0].line == 3

    def test_find_references_exclude_definition(self):
        def_item = MagicMock()
        def_item.file = Path("a.py")
        def_item.start_line = 1
        def_item.lines = ["def my_func():"]

        ref_item = MagicMock()
        ref_item.file = Path("b.py")
        ref_item.start_line = 5
        ref_item.lines = ["my_func()"]

        engine = _mock_engine_with_search_results([def_item, ref_item])
        integration = IDEIntegration(engine)
        refs = integration.find_references("c.py", 1, "my_func", include_definition=False)
        assert all(not r.is_definition for r in refs)

    def test_find_references_engine_exception(self):
        engine = MagicMock()
        engine.search.side_effect = RuntimeError("fail")
        integration = IDEIntegration(engine)
        refs = integration.find_references("f.py", 1, "sym")
        assert refs == []

    # -- provide_completion ------------------------------------------------

    def test_provide_completion_short_prefix(self):
        integration = IDEIntegration(MagicMock())
        assert integration.provide_completion("f.py", 1, 0, prefix="") == []
        assert integration.provide_completion("f.py", 1, 0, prefix="x") == []

    def test_provide_completion_found(self):
        item = MagicMock()
        item.file = MagicMock()
        item.file.name = "utils.py"
        item.lines = ["def process_data():"]
        engine = _mock_engine_with_search_results([item])
        integration = IDEIntegration(engine)
        completions = integration.provide_completion("f.py", 1, 0, prefix="proc")
        # May or may not find depending on regex match; test no crash
        assert isinstance(completions, list)

    # -- provide_hover -----------------------------------------------------

    def test_provide_hover_empty_symbol(self):
        integration = IDEIntegration(MagicMock())
        assert integration.provide_hover("f.py", 1, 0, "") is None

    def test_provide_hover_function_found(self):
        item = MagicMock()
        item.file = Path("utils.py")
        item.start_line = 5
        item.lines = ["def my_func(x, y):", "    return x + y"]
        engine = _mock_engine_with_search_results([item])
        integration = IDEIntegration(engine)
        hover = integration.provide_hover("f.py", 1, 0, "my_func")
        assert hover is not None
        assert hover.symbol_name == "my_func"
        assert hover.symbol_type == "function"

    def test_provide_hover_not_found(self):
        engine = _mock_engine_with_search_results([])
        integration = IDEIntegration(engine)
        hover = integration.provide_hover("f.py", 1, 0, "missing")
        assert hover is None

    # -- get_document_symbols ----------------------------------------------

    def test_get_document_symbols(self, tmp_path):
        p = tmp_path / "sample.py"
        p.write_text(
            "def foo():\n" "    pass\n" "\n" "class Bar:\n" "    pass\n" "\n" "MAX_SIZE = 100\n",
            encoding="utf-8",
        )
        integration = IDEIntegration(MagicMock())
        symbols = integration.get_document_symbols(str(p))
        names = [s.name for s in symbols]
        assert "foo" in names
        assert "Bar" in names
        assert "MAX_SIZE" in names

    def test_get_document_symbols_nonexistent(self):
        integration = IDEIntegration(MagicMock())
        symbols = integration.get_document_symbols("/nonexistent/file.py")
        assert symbols == []

    # -- get_workspace_symbols ---------------------------------------------

    def test_get_workspace_symbols_short_query(self):
        integration = IDEIntegration(MagicMock())
        assert integration.get_workspace_symbols("") == []
        assert integration.get_workspace_symbols("x") == []

    def test_get_workspace_symbols_found(self):
        item = MagicMock()
        item.file = Path("mod.py")
        item.start_line = 3
        item.lines = ["def process_data():"]
        engine = _mock_engine_with_search_results([item])
        integration = IDEIntegration(engine)
        symbols = integration.get_workspace_symbols("process")
        assert isinstance(symbols, list)

    # -- get_diagnostics ---------------------------------------------------

    def test_get_diagnostics_markers(self, tmp_path):
        p = tmp_path / "todo.py"
        p.write_text(
            "x = 1  # TODO fix this\n" "y = 2  # FIXME urgent\n" "z = 3  # HACK workaround\n",
            encoding="utf-8",
        )
        integration = IDEIntegration(MagicMock())
        diags = integration.get_diagnostics(str(p))
        assert len(diags) >= 3
        codes = [d.code for d in diags]
        assert "TODO" in codes
        assert "FIXME" in codes
        assert "HACK" in codes
        # FIXME and HACK should be warnings, TODO should be info
        todo_diag = next(d for d in diags if d.code == "TODO")
        fixme_diag = next(d for d in diags if d.code == "FIXME")
        assert todo_diag.severity == "info"
        assert fixme_diag.severity == "warning"

    def test_get_diagnostics_nonexistent(self):
        integration = IDEIntegration(MagicMock())
        diags = integration.get_diagnostics("/nonexistent/file.py")
        assert diags == []

    def test_get_diagnostics_clean_file(self, tmp_path):
        p = tmp_path / "clean.py"
        p.write_text("x = 1\ny = 2\n", encoding="utf-8")
        integration = IDEIntegration(MagicMock())
        diags = integration.get_diagnostics(str(p))
        assert diags == []


# ---------------------------------------------------------------------------
# ide_query
# ---------------------------------------------------------------------------


class TestIdeQuery:
    """Tests for the ide_query convenience function."""

    def test_basic_structure(self):
        from pysearch.core.types import SearchStats

        mock_item = MagicMock()
        mock_item.file = Path("test.py")
        mock_item.start_line = 1
        mock_item.end_line = 3
        mock_item.lines = ["def foo():", "    pass"]
        mock_item.match_spans = [(1, (4, 7))]

        mock_result = MagicMock()
        mock_result.items = [mock_item]
        mock_result.stats = SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=0.5)

        engine = MagicMock()
        engine.run.return_value = mock_result

        from pysearch.core.types import Query

        query = Query(pattern="foo")
        result = ide_query(engine, query)

        assert "items" in result
        assert "stats" in result
        assert len(result["items"]) == 1
        assert result["items"][0]["file"] == "test.py"
        assert result["items"][0]["start_line"] == 1

    def test_empty_results(self):
        from pysearch.core.types import SearchStats

        mock_result = MagicMock()
        mock_result.items = []
        mock_result.stats = SearchStats(files_scanned=0, files_matched=0, items=0, elapsed_ms=0.0)

        engine = MagicMock()
        engine.run.return_value = mock_result

        from pysearch.core.types import Query

        query = Query(pattern="nonexistent")
        result = ide_query(engine, query)
        assert result["items"] == []
