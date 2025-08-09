from __future__ import annotations

from pathlib import Path

from pysearch.dependency_analysis import (
    DependencyAnalyzer,
    DependencyGraph,
    ImportNode,
)
from pysearch.types import Language


def write(tmp: Path, rel: str, text: str) -> Path:
    p = tmp / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_dependency_graph_basic() -> None:
    g = DependencyGraph()
    g.add_edge("a", "b", ImportNode(module="b"))
    g.add_edge("b", "c", ImportNode(module="c"))
    assert g.get_dependencies("a") == ["b"]
    assert g.get_dependents("c") == ["b"]
    assert g.has_path("a", "c") is True
    d = g.to_dict()
    assert "nodes" in d and "edges" in d


def test_analyze_python_files_and_metrics(tmp_path: Path) -> None:
    # create small package with circular dep
    write(tmp_path, "pkg/__init__.py", "")
    write(tmp_path, "pkg/a.py", "import pkg.b\n")
    write(tmp_path, "pkg/b.py", "from pkg import a as a_mod\n")

    analyzer = DependencyAnalyzer()
    graph = analyzer.analyze_directory(tmp_path)
    # sanity: allow fully-qualified module naming variations
    assert any(n.endswith("pkg.a") for n in graph.nodes)
    assert any(n.endswith("pkg.b") for n in graph.nodes)

    metrics = analyzer.calculate_metrics()
    assert metrics.total_modules >= 2
    assert metrics.total_dependencies >= 2
    # circular detection can vary by naming; rely on dedicated detector test for cycles

    target = next((n for n in graph.nodes if n.endswith("pkg.a")), None)
    assert target, graph.nodes
    impact = analyzer.find_impact_analysis(target)
    assert "impact_score" in impact and 0.0 <= impact["impact_score"] <= 1.0


def test_language_parsers_non_python(tmp_path: Path) -> None:
    js = write(tmp_path, "web/src/app.js", "import x from 'y';\nexport {z} from './z.js'\n")
    ts = write(tmp_path, "web/src/app.ts", "import {A} from 'lib';\nexport * from './w'\n")
    java = write(tmp_path, "App.java", "import java.util.List;\nclass App{}\n")
    cs = write(tmp_path, "Program.cs", "using System;\nclass P{}\n")
    go = write(tmp_path, "main.go", "package main\nimport (\n \"fmt\"\n)\nfunc main(){}\n")

    az = DependencyAnalyzer()
    # Directly exercise parser internals via analyze_file routing
    for p, lang in [(js, Language.JAVASCRIPT), (ts, Language.TYPESCRIPT), (java, Language.JAVA), (cs, Language.CSHARP), (go, Language.GO)]:
        nodes = az.analyze_file(p)
        assert isinstance(nodes, list)

