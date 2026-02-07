"""Tests for pysearch.integrations.ide_hooks module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

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
)


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
            file="test.py", line=10, symbol_name="main",
            symbol_type="function",
        )
        d = loc.to_dict()
        assert d["file"] == "test.py"
        assert d["line"] == 10
        assert d["symbol_name"] == "main"


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

    def test_to_dict_uses_label_as_insert_text(self):
        item = CompletionItem(label="hello")
        d = item.to_dict()
        assert d["insert_text"] == "hello"


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


class TestDocumentSymbol:
    """Tests for DocumentSymbol dataclass."""

    def test_creation(self):
        sym = DocumentSymbol(
            name="main", kind="function",
            line=1, end_line=10,
        )
        assert sym.name == "main"
        assert sym.kind == "function"
        assert sym.line == 1


class TestDiagnostic:
    """Tests for Diagnostic dataclass."""

    def test_creation(self):
        diag = Diagnostic(
            file="test.py", line=5,
            message="unused variable", severity="warning",
        )
        assert diag.message == "unused variable"
        assert diag.severity == "warning"


class TestIDEHooks:
    """Tests for IDEHooks class (hook registry)."""

    def test_init(self):
        hooks = IDEHooks()
        assert hooks is not None

    def test_register_search_handler(self):
        hooks = IDEHooks()
        handler = MagicMock()
        hook_id = hooks.register_search_handler(handler)
        assert isinstance(hook_id, str)
        hook_list = hooks.list_hooks()
        types = [h["type"] for h in hook_list]
        assert "search" in types

    def test_unregister_hook(self):
        hooks = IDEHooks()
        handler = MagicMock()
        hook_id = hooks.register_search_handler(handler)
        result = hooks.unregister_hook(hook_id)
        assert result is True
        hook_list = hooks.list_hooks()
        assert len(hook_list) == 0

    def test_trigger_hook(self):
        hooks = IDEHooks()
        handler = MagicMock(return_value=[{"result": "ok"}])
        hook_id = hooks.register_search_handler(handler)
        result = hooks.trigger_hook(hook_id, query="test")
        assert result == [{"result": "ok"}]
        handler.assert_called_once_with(query="test")

    def test_list_hooks_empty(self):
        hooks = IDEHooks()
        result = hooks.list_hooks()
        assert isinstance(result, list)
        assert len(result) == 0


class TestIDEIntegration:
    """Tests for IDEIntegration class."""

    def test_init(self):
        mock_engine = MagicMock()
        integration = IDEIntegration(mock_engine)
        assert integration is not None
        assert integration.engine is mock_engine
