from __future__ import annotations

from pysearch.analysis.dependency_analysis import CircularDependencyDetector, DependencyGraph, ImportNode


def test_circular_dependency_detector_finds_cycle():
    g = DependencyGraph()
    g.add_edge("a", "b", ImportNode(module="b"))
    g.add_edge("b", "a", ImportNode(module="a"))
    detector = CircularDependencyDetector(g)
    cycles = detector.find_cycles()
    assert any(set(c) == {"a", "b"} for c in cycles)

