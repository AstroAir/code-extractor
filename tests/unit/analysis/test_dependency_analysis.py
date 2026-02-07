"""Tests for pysearch.analysis.dependency_analysis module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.analysis.dependency_analysis import (
    CircularDependencyDetector,
    DependencyGraph,
    DependencyMetrics,
    ImportNode,
)


class TestImportNode:
    """Tests for ImportNode dataclass."""

    def test_basic_import(self):
        node = ImportNode(module="os")
        assert node.module == "os"
        assert node.alias is None
        assert node.from_module is None

    def test_from_import(self):
        node = ImportNode(module="path", from_module="os")
        assert node.module == "path"
        assert node.from_module == "os"

    def test_aliased_import(self):
        node = ImportNode(module="numpy", alias="np")
        assert node.alias == "np"


class TestDependencyGraph:
    """Tests for DependencyGraph class."""

    def test_empty_graph(self):
        g = DependencyGraph()
        assert g.nodes == set()
        assert isinstance(g.edges, dict)

    def test_add_node(self):
        g = DependencyGraph()
        g.add_node("module_a")
        assert "module_a" in g.nodes

    def test_add_edge(self):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("b")
        imp = ImportNode(module="b")
        g.add_edge("a", "b", imp)
        assert len(g.edges.get("a", [])) > 0

    def test_add_duplicate_node(self):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("a")
        # nodes is a set, so no duplicates
        assert len([n for n in g.nodes if n == "a"]) == 1


class TestCircularDependencyDetector:
    """Tests for CircularDependencyDetector class."""

    def test_no_cycles(self):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", ImportNode(module="b"))
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert cycles == []

    def test_simple_cycle(self):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "a", ImportNode(module="a"))
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert len(cycles) > 0


class TestDependencyMetrics:
    """Tests for DependencyMetrics class."""

    def test_empty_graph_metrics(self):
        # DependencyMetrics is a dataclass, not constructed from a graph
        metrics = DependencyMetrics()
        assert metrics.total_modules == 0
        assert metrics.total_dependencies == 0
