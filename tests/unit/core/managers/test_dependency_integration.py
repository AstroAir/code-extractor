"""Tests for pysearch.core.managers.dependency_integration module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from pysearch.core.config import SearchConfig
from pysearch.core.managers.dependency_integration import DependencyIntegrationManager


def _make_mock_graph(nodes=None, edges=None):
    """Create a mock dependency graph object."""
    graph = MagicMock()
    graph.nodes = nodes or []
    graph.edges = edges or {}
    return graph


def _make_mock_edge(target, weight=1.0):
    """Create a mock dependency edge."""
    edge = MagicMock()
    edge.target = target
    edge.weight = weight
    return edge


class TestDependencyIntegrationManager:
    """Tests for DependencyIntegrationManager class."""

    def test_init(self):
        cfg = SearchConfig()
        mgr = DependencyIntegrationManager(cfg)
        assert mgr.config is cfg
        assert mgr.dependency_analyzer is None
        assert mgr._logger is None

    def test_set_logger(self):
        mgr = DependencyIntegrationManager(SearchConfig())
        logger = MagicMock()
        mgr.set_logger(logger)
        assert mgr._logger is logger

    def test_analyze_dependencies_with_mock_analyzer(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph(["a", "b"])
        mock_analyzer.analyze_directory.return_value = mock_graph
        mgr.dependency_analyzer = mock_analyzer
        result = mgr.analyze_dependencies(tmp_path)
        assert result is mock_graph
        mock_analyzer.analyze_directory.assert_called_once()

    def test_analyze_dependencies_default_directory(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph()
        mock_analyzer.analyze_directory.return_value = mock_graph
        mgr.dependency_analyzer = mock_analyzer
        result = mgr.analyze_dependencies()
        assert result is mock_graph

    def test_get_dependency_metrics_with_mock(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph()
        mock_analyzer.analyze_directory.return_value = mock_graph
        mock_analyzer.calculate_metrics.return_value = {"total_modules": 5}
        mgr.dependency_analyzer = mock_analyzer
        result = mgr.get_dependency_metrics()
        assert result == {"total_modules": 5}

    def test_get_dependency_metrics_empty_when_no_analyzer_import(self):
        mgr = DependencyIntegrationManager(SearchConfig(paths=["."]))
        with patch.object(mgr, "_ensure_dependency_analyzer"):
            mgr.dependency_analyzer = None
            result = mgr.get_dependency_metrics(graph=_make_mock_graph())
            assert result == {}

    def test_find_dependency_impact_with_mock(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph()
        mock_analyzer.analyze_directory.return_value = mock_graph
        mock_analyzer.find_impact_analysis.return_value = {
            "direct_dependents": ["b"],
            "total_affected": 2,
        }
        mgr.dependency_analyzer = mock_analyzer
        result = mgr.find_dependency_impact("a")
        assert result["total_affected"] == 2

    def test_find_dependency_impact_no_analyzer(self):
        mgr = DependencyIntegrationManager(SearchConfig(paths=["."]))
        with patch.object(mgr, "_ensure_dependency_analyzer"):
            mgr.dependency_analyzer = None
            result = mgr.find_dependency_impact("a", graph=_make_mock_graph())
            assert result == {}

    def test_suggest_refactoring_with_mock(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph()
        mock_analyzer.analyze_directory.return_value = mock_graph
        mock_analyzer.suggest_refactoring_opportunities.return_value = [{"type": "split"}]
        mgr.dependency_analyzer = mock_analyzer
        result = mgr.suggest_refactoring_opportunities()
        assert len(result) == 1

    def test_suggest_refactoring_no_analyzer(self):
        mgr = DependencyIntegrationManager(SearchConfig(paths=["."]))
        with patch.object(mgr, "_ensure_dependency_analyzer"):
            mgr.dependency_analyzer = None
            result = mgr.suggest_refactoring_opportunities(graph=_make_mock_graph())
            assert result == []

    def test_detect_circular_dependencies_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        mgr = DependencyIntegrationManager(cfg)
        mock_analyzer = MagicMock()
        mock_graph = _make_mock_graph(nodes=["a", "b"], edges={})
        mock_analyzer.analyze_directory.return_value = mock_graph
        mgr.dependency_analyzer = mock_analyzer
        # CircularDependencyDetector import may fail, returns []
        result = mgr.detect_circular_dependencies()
        assert isinstance(result, list)

    def test_get_module_coupling_metrics(self):
        mgr = DependencyIntegrationManager(SearchConfig(paths=["."]))
        edge_ab = _make_mock_edge("b")
        graph = _make_mock_graph(nodes=["a", "b"], edges={"a": [edge_ab], "b": []})
        with patch.object(mgr, "_ensure_dependency_analyzer"):
            result = mgr.get_module_coupling_metrics(graph=graph)
            assert "a" in result
            assert result["a"]["efferent_coupling"] == 1
            assert "b" in result

    def test_find_dead_code(self):
        mgr = DependencyIntegrationManager(SearchConfig(paths=["."]))
        edge_ab = _make_mock_edge("b")
        graph = _make_mock_graph(nodes=["a", "b", "orphan"], edges={"a": [edge_ab], "b": []})
        with patch.object(mgr, "_ensure_dependency_analyzer"):
            result = mgr.find_dead_code(graph=graph)
            assert "orphan" in result
            # "b" has incoming from "a", so it's not dead
            assert "b" not in result

    def test_export_dependency_graph_dot(self):
        mgr = DependencyIntegrationManager(SearchConfig())
        edge = _make_mock_edge("b")
        graph = _make_mock_graph(edges={"a": [edge]})
        result = mgr.export_dependency_graph(graph, format="dot")
        assert "digraph" in result
        assert '"a" -> "b"' in result

    def test_export_dependency_graph_json(self):
        mgr = DependencyIntegrationManager(SearchConfig())
        edge = _make_mock_edge("b", weight=1.0)
        graph = _make_mock_graph(nodes=["a", "b"], edges={"a": [edge]})
        result = mgr.export_dependency_graph(graph, format="json")
        assert '"nodes"' in result
        assert '"edges"' in result

    def test_export_dependency_graph_csv(self):
        mgr = DependencyIntegrationManager(SearchConfig())
        edge = _make_mock_edge("b", weight=2.0)
        graph = _make_mock_graph(edges={"a": [edge]})
        result = mgr.export_dependency_graph(graph, format="csv")
        assert "source,target,weight" in result
        assert '"a","b",2.0' in result

    def test_export_dependency_graph_invalid_format(self):
        mgr = DependencyIntegrationManager(SearchConfig())
        graph = _make_mock_graph()
        result = mgr.export_dependency_graph(graph, format="invalid")
        assert result == ""
