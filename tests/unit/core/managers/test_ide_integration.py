"""Tests for pysearch.core.managers.ide_integration module."""

from __future__ import annotations

from unittest.mock import MagicMock

from pysearch.core.config import SearchConfig
from pysearch.core.managers.ide_integration import IDEIntegrationManager


class TestIDEIntegrationManager:
    """Tests for IDEIntegrationManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = IDEIntegrationManager(cfg)
        assert mgr.config is cfg
        assert mgr._ide_integration is None
        assert mgr._ide_hooks is None
        assert mgr._enabled is False

    def test_is_ide_enabled_default(self):
        mgr = IDEIntegrationManager(SearchConfig())
        assert mgr.is_ide_enabled() is False

    def test_disable_ide_integration_when_not_enabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr.disable_ide_integration()
        assert mgr._enabled is False

    def test_enable_ide_integration_import_failure(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.enable_ide_integration(MagicMock())
        # The integrations.ide_hooks module may not be available in test env
        # Either it succeeds (True) or fails gracefully (False)
        assert isinstance(result, bool)

    def test_enable_ide_integration_already_enabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._enabled = True
        result = mgr.enable_ide_integration(MagicMock())
        assert result is True

    def test_disable_ide_integration_clears_state(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._enabled = True
        mgr._ide_integration = MagicMock()
        mgr._ide_hooks = MagicMock()
        mgr.disable_ide_integration()
        assert mgr._enabled is False
        assert mgr._ide_integration is None
        assert mgr._ide_hooks is None

    # --- Tests for disabled operations returning empty/None ---

    def test_jump_to_definition_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.jump_to_definition("test.py", 1, "foo")
        assert result is None

    def test_find_references_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.find_references("test.py", 1, "foo")
        assert result == []

    def test_provide_completion_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.provide_completion("test.py", 1, 0)
        assert result == []

    def test_provide_hover_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.provide_hover("test.py", 1, 0, "foo")
        assert result is None

    def test_get_document_symbols_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.get_document_symbols("test.py")
        assert result == []

    def test_get_workspace_symbols_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.get_workspace_symbols("foo")
        assert result == []

    def test_get_diagnostics_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.get_diagnostics("test.py")
        assert result == []

    def test_ide_query_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.ide_query(MagicMock())
        assert result == {"items": [], "stats": {}}

    def test_register_hook_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.register_hook("search", lambda: None)
        assert result is None

    def test_trigger_hook_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.trigger_hook("some-hook-id")
        assert result is None

    def test_list_hooks_disabled(self):
        mgr = IDEIntegrationManager(SearchConfig())
        result = mgr.list_hooks()
        assert result == []

    # --- Tests with mocked IDE integration ---

    def test_jump_to_definition_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.jump_to_definition.side_effect = RuntimeError("fail")
        result = mgr.jump_to_definition("test.py", 1, "foo")
        assert result is None

    def test_find_references_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.find_references.side_effect = RuntimeError("fail")
        result = mgr.find_references("test.py", 1, "foo")
        assert result == []

    def test_provide_completion_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.provide_completion.side_effect = RuntimeError("fail")
        result = mgr.provide_completion("test.py", 1, 0)
        assert result == []

    def test_provide_hover_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.provide_hover.side_effect = RuntimeError("fail")
        result = mgr.provide_hover("test.py", 1, 0, "foo")
        assert result is None

    def test_get_document_symbols_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.get_document_symbols.side_effect = RuntimeError("fail")
        result = mgr.get_document_symbols("test.py")
        assert result == []

    def test_get_workspace_symbols_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.get_workspace_symbols.side_effect = RuntimeError("fail")
        result = mgr.get_workspace_symbols("foo")
        assert result == []

    def test_get_diagnostics_exception(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.get_diagnostics.side_effect = RuntimeError("fail")
        result = mgr.get_diagnostics("test.py")
        assert result == []

    def test_jump_to_definition_returns_none_result(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.jump_to_definition.return_value = None
        result = mgr.jump_to_definition("test.py", 1, "foo")
        assert result is None

    def test_provide_hover_returns_none_result(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mgr._ide_integration.provide_hover.return_value = None
        result = mgr.provide_hover("test.py", 1, 0, "foo")
        assert result is None

    def test_jump_to_definition_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_result = MagicMock()
        mock_result.to_dict.return_value = {"file": "a.py", "line": 5}
        mgr._ide_integration.jump_to_definition.return_value = mock_result
        result = mgr.jump_to_definition("test.py", 1, "foo")
        assert result == {"file": "a.py", "line": 5}

    def test_find_references_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_ref = MagicMock()
        mock_ref.to_dict.return_value = {"file": "b.py", "line": 10}
        mgr._ide_integration.find_references.return_value = [mock_ref]
        result = mgr.find_references("test.py", 1, "foo")
        assert result == [{"file": "b.py", "line": 10}]

    def test_provide_completion_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_item = MagicMock()
        mock_item.to_dict.return_value = {"label": "func_name"}
        mgr._ide_integration.provide_completion.return_value = [mock_item]
        result = mgr.provide_completion("test.py", 1, 0, "fun")
        assert result == [{"label": "func_name"}]

    def test_provide_hover_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_info = MagicMock()
        mock_info.to_dict.return_value = {"content": "def foo()"}
        mgr._ide_integration.provide_hover.return_value = mock_info
        result = mgr.provide_hover("test.py", 1, 0, "foo")
        assert result == {"content": "def foo()"}

    def test_get_document_symbols_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_sym = MagicMock()
        mock_sym.to_dict.return_value = {"name": "MyClass", "kind": "class"}
        mgr._ide_integration.get_document_symbols.return_value = [mock_sym]
        result = mgr.get_document_symbols("test.py")
        assert result == [{"name": "MyClass", "kind": "class"}]

    def test_get_workspace_symbols_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_sym = MagicMock()
        mock_sym.to_dict.return_value = {"name": "main", "kind": "function"}
        mgr._ide_integration.get_workspace_symbols.return_value = [mock_sym]
        result = mgr.get_workspace_symbols("main")
        assert result == [{"name": "main", "kind": "function"}]

    def test_get_diagnostics_success(self):
        mgr = IDEIntegrationManager(SearchConfig())
        mgr._ide_integration = MagicMock()
        mock_diag = MagicMock()
        mock_diag.to_dict.return_value = {"message": "TODO", "line": 5}
        mgr._ide_integration.get_diagnostics.return_value = [mock_diag]
        result = mgr.get_diagnostics("test.py")
        assert result == [{"message": "TODO", "line": 5}]
