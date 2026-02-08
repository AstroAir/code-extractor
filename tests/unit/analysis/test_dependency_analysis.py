"""Tests for pysearch.analysis.dependency_analysis module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.analysis.dependency_analysis import (
    CircularDependencyDetector,
    DependencyAnalyzer,
    DependencyEdge,
    DependencyGraph,
    DependencyMetrics,
    ImportNode,
)
from pysearch.core.types import Language


# ---------------------------------------------------------------------------
# ImportNode
# ---------------------------------------------------------------------------
class TestImportNode:
    """Tests for ImportNode dataclass."""

    def test_basic_import(self):
        node = ImportNode(module="os")
        assert node.module == "os"
        assert node.alias is None
        assert node.from_module is None
        assert node.is_relative is False
        assert node.import_type == "import"
        assert node.language == Language.PYTHON

    def test_from_import(self):
        node = ImportNode(module="path", from_module="os", import_type="from")
        assert node.module == "path"
        assert node.from_module == "os"
        assert node.import_type == "from"

    def test_aliased_import(self):
        node = ImportNode(module="numpy", alias="np")
        assert node.alias == "np"

    def test_relative_import(self):
        node = ImportNode(module="utils", from_module=".", is_relative=True)
        assert node.is_relative is True

    def test_metadata(self):
        node = ImportNode(module="x", metadata={"level": 2})
        assert node.metadata == {"level": 2}


# ---------------------------------------------------------------------------
# DependencyEdge
# ---------------------------------------------------------------------------
class TestDependencyEdge:
    """Tests for DependencyEdge dataclass."""

    def test_creation(self):
        edge = DependencyEdge(source="a", target="b")
        assert edge.source == "a"
        assert edge.target == "b"
        assert edge.weight == 1
        assert edge.edge_type == "direct"
        assert edge.import_nodes == []

    def test_with_import_nodes(self):
        imp = ImportNode(module="b")
        edge = DependencyEdge(source="a", target="b", import_nodes=[imp], weight=1)
        assert len(edge.import_nodes) == 1


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------
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

    def test_add_duplicate_node(self):
        g = DependencyGraph()
        g.add_node("a")
        g.add_node("a")
        assert len([n for n in g.nodes if n == "a"]) == 1

    def test_add_edge(self):
        g = DependencyGraph()
        imp = ImportNode(module="b")
        g.add_edge("a", "b", imp)
        assert "a" in g.nodes
        assert "b" in g.nodes
        assert len(g.edges["a"]) == 1
        assert g.edges["a"][0].target == "b"

    def test_add_duplicate_edge_increments_weight(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("a", "b", ImportNode(module="b2"))
        assert len(g.edges["a"]) == 1
        assert g.edges["a"][0].weight == 2
        assert len(g.edges["a"][0].import_nodes) == 2

    def test_get_dependencies(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("a", "c", ImportNode(module="c"))
        deps = g.get_dependencies("a")
        assert set(deps) == {"b", "c"}

    def test_get_dependencies_empty(self):
        g = DependencyGraph()
        g.add_node("a")
        assert g.get_dependencies("a") == []

    def test_get_dependents(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("c", "b", ImportNode(module="b"))
        dependents = g.get_dependents("b")
        assert set(dependents) == {"a", "c"}

    def test_get_transitive_dependencies(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "c", ImportNode(module="c"))
        g.add_edge("c", "d", ImportNode(module="d"))
        trans = g.get_transitive_dependencies("a")
        assert trans == {"b", "c", "d"}

    def test_get_transitive_dependencies_max_depth(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "c", ImportNode(module="c"))
        g.add_edge("c", "d", ImportNode(module="d"))
        # max_depth=2: a(0)→b(1), b skipped at depth≥2, so only b reached
        trans = g.get_transitive_dependencies("a", max_depth=2)
        assert "b" in trans
        assert "d" not in trans
        # max_depth=3: reaches b(1)→c(2), c skipped at depth≥3
        trans3 = g.get_transitive_dependencies("a", max_depth=3)
        assert "b" in trans3
        assert "c" in trans3
        assert "d" not in trans3

    def test_has_path_direct(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        assert g.has_path("a", "b") is True

    def test_has_path_transitive(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "c", ImportNode(module="c"))
        assert g.has_path("a", "c") is True

    def test_has_path_no_path(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        assert g.has_path("b", "a") is False

    def test_has_path_self(self):
        g = DependencyGraph()
        g.add_node("a")
        assert g.has_path("a", "a") is True

    def test_to_dict(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        d = g.to_dict()
        assert set(d["nodes"]) == {"a", "b"}
        assert len(d["edges"]) == 1
        assert d["edges"][0]["source"] == "a"
        assert d["edges"][0]["target"] == "b"


# ---------------------------------------------------------------------------
# CircularDependencyDetector
# ---------------------------------------------------------------------------
class TestCircularDependencyDetector:
    """Tests for CircularDependencyDetector class."""

    def test_no_cycles(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert cycles == []

    def test_simple_cycle(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "a", ImportNode(module="a"))
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b"}

    def test_three_node_cycle(self):
        g = DependencyGraph()
        g.add_edge("a", "b", ImportNode(module="b"))
        g.add_edge("b", "c", ImportNode(module="c"))
        g.add_edge("c", "a", ImportNode(module="a"))
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert len(cycles) == 1
        assert set(cycles[0]) == {"a", "b", "c"}

    def test_empty_graph(self):
        g = DependencyGraph()
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert cycles == []

    def test_single_node_no_cycle(self):
        g = DependencyGraph()
        g.add_node("a")
        detector = CircularDependencyDetector(g)
        cycles = detector.find_cycles()
        assert cycles == []


# ---------------------------------------------------------------------------
# DependencyMetrics
# ---------------------------------------------------------------------------
class TestDependencyMetrics:
    """Tests for DependencyMetrics dataclass."""

    def test_defaults(self):
        metrics = DependencyMetrics()
        assert metrics.total_modules == 0
        assert metrics.total_dependencies == 0
        assert metrics.circular_dependencies == 0
        assert metrics.max_depth == 0
        assert metrics.average_dependencies_per_module == 0.0
        assert metrics.coupling_metrics == {}
        assert metrics.dead_modules == []
        assert metrics.highly_coupled_modules == []


# ---------------------------------------------------------------------------
# DependencyAnalyzer
# ---------------------------------------------------------------------------
class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer class."""

    def test_init(self):
        analyzer = DependencyAnalyzer()
        assert analyzer.graph is not None
        assert Language.PYTHON in analyzer.language_parsers

    def test_supported_languages(self):
        analyzer = DependencyAnalyzer()
        expected = {
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.JAVA,
            Language.CSHARP,
            Language.GO,
            Language.RUST,
            Language.PHP,
            Language.RUBY,
            Language.KOTLIN,
            Language.SWIFT,
            Language.SCALA,
            Language.C,
            Language.CPP,
            Language.DART,
            Language.LUA,
            Language.PERL,
            Language.ELIXIR,
            Language.HASKELL,
        }
        assert set(analyzer.language_parsers.keys()) == expected

    def test_analyze_python_file(self, tmp_path: Path):
        f = tmp_path / "example.py"
        f.write_text("import os\nfrom pathlib import Path\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert len(imports) >= 2
        modules = [i.module for i in imports]
        assert "os" in modules

    def test_analyze_javascript_file(self, tmp_path: Path):
        f = tmp_path / "example.js"
        f.write_text(
            "import React from 'react';\n"
            "const fs = require('fs');\n",
            encoding="utf-8",
        )
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert len(imports) >= 2

    def test_analyze_java_file(self, tmp_path: Path):
        f = tmp_path / "Example.java"
        f.write_text(
            "import java.util.List;\n"
            "import static java.lang.Math.PI;\n",
            encoding="utf-8",
        )
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert len(imports) >= 2
        types = [i.import_type for i in imports]
        assert "static" in types

    def test_analyze_csharp_file(self, tmp_path: Path):
        f = tmp_path / "Example.cs"
        f.write_text("using System;\nusing System.Collections.Generic;\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert len(imports) >= 2
        assert all(i.import_type == "using" for i in imports)

    def test_analyze_go_file(self, tmp_path: Path):
        f = tmp_path / "main.go"
        f.write_text(
            'package main\n\nimport (\n\t"fmt"\n\t"os"\n)\n',
            encoding="utf-8",
        )
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert len(imports) >= 2

    def test_analyze_unsupported_language(self, tmp_path: Path):
        f = tmp_path / "style.css"
        f.write_text("body { color: red; }\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert imports == []

    def test_analyze_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        assert imports == []

    def test_analyze_directory(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("import os\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("import sys\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(tmp_path)
        assert isinstance(graph, DependencyGraph)
        assert len(graph.nodes) > 0

    def test_calculate_metrics(self, tmp_path: Path):
        (tmp_path / "a.py").write_text("import os\nimport sys\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        analyzer.analyze_directory(tmp_path)
        metrics = analyzer.calculate_metrics()
        assert isinstance(metrics, DependencyMetrics)
        assert metrics.total_modules > 0

    def test_find_impact_analysis(self):
        analyzer = DependencyAnalyzer()
        analyzer.graph.add_edge("a", "b", ImportNode(module="b"))
        analyzer.graph.add_edge("c", "a", ImportNode(module="a"))
        impact = analyzer.find_impact_analysis("a")
        assert impact["module"] == "a"
        assert "c" in impact["direct_dependents"]

    def test_find_impact_analysis_missing_module(self):
        analyzer = DependencyAnalyzer()
        result = analyzer.find_impact_analysis("nonexistent")
        assert "error" in result

    def test_suggest_refactoring_opportunities(self):
        analyzer = DependencyAnalyzer()
        # Create a cycle
        analyzer.graph.add_edge("a", "b", ImportNode(module="b"))
        analyzer.graph.add_edge("b", "a", ImportNode(module="a"))
        suggestions = analyzer.suggest_refactoring_opportunities()
        types = [s["type"] for s in suggestions]
        assert "break_circular_dependency" in types

    def test_python_regex_fallback(self, tmp_path: Path):
        f = tmp_path / "bad_syntax.py"
        f.write_text("import os\ndef bad(\n", encoding="utf-8")
        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(f)
        # Regex fallback should still find the import
        assert any(i.module == "os" for i in imports)
