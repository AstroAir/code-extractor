from __future__ import annotations

from pathlib import Path

from pysearch import PySearch, SearchConfig


def test_api_dependency_wrappers(tmp_path: Path) -> None:
    # small package
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "pkg" / "a.py").write_text("import pkg.b\n", encoding="utf-8")
    (tmp_path / "pkg" / "b.py").write_text("from pkg import a as a_mod\n", encoding="utf-8")

    eng = PySearch(
        SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False)
    )

    graph = eng.analyze_dependencies()
    metrics = eng.get_dependency_metrics(graph)
    assert metrics.total_modules >= 2

    # impact
    target = next((n for n in graph.nodes if n.endswith("pkg.a")), None)
    if target:
        impact = eng.find_dependency_impact(target, graph)
        assert (
            "direct_dependencies" in impact
            and "direct_dependents" in impact
            and "impact_score" in impact
        )
