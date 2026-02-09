#!/usr/bin/env python3
"""
Example 06: Dependency Analysis — 依赖分析示例

Demonstrates:
- Building dependency graphs / 构建依赖图
- Dependency metrics / 依赖指标
- Circular dependency detection / 循环依赖检测
- Change impact analysis / 变更影响分析
- Refactoring suggestions / 重构建议
- Module coupling metrics / 模块耦合度
- Dead code detection / 死代码检测
- Graph export (DOT, JSON) / 依赖图导出
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Build dependency graph / 构建依赖图
# ---------------------------------------------------------------------------


def demo_dependency_graph(engine: PySearch) -> None:
    section("1. Build Dependency Graph / 构建依赖图")

    graph = engine.analyze_dependencies()

    if graph is None:
        print("  (Dependency analysis returned None — check your search path)")
        return

    print(f"  Modules: {len(graph.nodes)}")
    print(f"  Edges: {sum(len(e) for e in graph.edges.values())}")
    print()

    # Show first 10 modules / 展示前 10 个模块
    print("  Sample modules:")
    for node in sorted(graph.nodes)[:10]:
        deps = graph.get_dependencies(node)
        print(f"    {node} → {len(deps)} dependencies")

    return graph


# ---------------------------------------------------------------------------
# 2. Dependency metrics / 依赖指标
# ---------------------------------------------------------------------------


def demo_dependency_metrics(engine: PySearch) -> None:
    section("2. Dependency Metrics / 依赖指标")

    metrics = engine.get_dependency_metrics()

    if metrics is None:
        print("  (No metrics available)")
        return

    print(f"  Total modules: {metrics.total_modules}")
    print(f"  Total dependencies: {metrics.total_dependencies}")
    print(f"  Circular dependencies: {metrics.circular_dependencies}")
    print(f"  Max depth: {metrics.max_depth}")
    print(f"  Avg deps per module: {metrics.average_dependencies_per_module:.2f}")

    if metrics.highly_coupled_modules:
        print("\n  Highly coupled modules:")
        for module in metrics.highly_coupled_modules[:5]:
            print(f"    - {module}")

    if metrics.dead_modules:
        print("\n  Potentially dead modules:")
        for module in metrics.dead_modules[:5]:
            print(f"    - {module}")


# ---------------------------------------------------------------------------
# 3. Circular dependency detection / 循环依赖检测
# ---------------------------------------------------------------------------


def demo_circular_deps(engine: PySearch) -> None:
    section("3. Circular Dependency Detection / 循环依赖检测")

    cycles = engine.detect_circular_dependencies()

    if not cycles:
        print("  No circular dependencies found! ✓")
    else:
        print(f"  Found {len(cycles)} circular dependency chains:\n")
        for i, cycle in enumerate(cycles[:5], 1):
            chain = " → ".join(cycle)
            print(f"    {i}. {chain}")

        if len(cycles) > 5:
            print(f"    ... and {len(cycles) - 5} more")


# ---------------------------------------------------------------------------
# 4. Change impact analysis / 变更影响分析
# ---------------------------------------------------------------------------


def demo_impact_analysis(engine: PySearch) -> None:
    section("4. Change Impact Analysis / 变更影响分析")

    # Analyze impact of changing the core config module
    # 分析修改核心配置模块的影响
    graph = engine.analyze_dependencies()
    if graph is None:
        print("  (Dependency analysis not available)")
        return

    # Pick a module that exists in the graph
    target_module = None
    for node in graph.nodes:
        if "config" in node.lower():
            target_module = node
            break

    if target_module is None:
        # Fallback: use the first module
        nodes_list = sorted(graph.nodes)
        target_module = nodes_list[0] if nodes_list else None

    if target_module is None:
        print("  (No modules found)")
        return

    impact = engine.find_dependency_impact(target_module)

    print(f"  Module: {target_module}")
    if isinstance(impact, dict):
        for key, value in list(impact.items())[:6]:
            if isinstance(value, int | float | str | bool):
                print(f"    {key}: {value}")
            elif isinstance(value, list):
                print(f"    {key}: {len(value)} items")
                for item in value[:3]:
                    print(f"      - {item}")
    else:
        print(f"  Impact result: {impact}")


# ---------------------------------------------------------------------------
# 5. Refactoring suggestions / 重构建议
# ---------------------------------------------------------------------------


def demo_refactoring_suggestions(engine: PySearch) -> None:
    section("5. Refactoring Suggestions / 重构建议")

    suggestions = engine.suggest_refactoring_opportunities()

    if not suggestions:
        print("  No refactoring suggestions — code structure looks clean! ✓")
    else:
        print(f"  Found {len(suggestions)} suggestions:\n")
        for suggestion in suggestions[:5]:
            priority = suggestion.get("priority", "medium").upper()
            stype = suggestion.get("type", "unknown")
            desc = suggestion.get("description", "")
            rationale = suggestion.get("rationale", "")
            print(f"    [{priority}] {stype}")
            print(f"      {desc}")
            if rationale:
                print(f"      Rationale: {rationale}")
            print()


# ---------------------------------------------------------------------------
# 6. Module coupling metrics / 模块耦合度
# ---------------------------------------------------------------------------


def demo_coupling_metrics(engine: PySearch) -> None:
    section("6. Module Coupling Metrics / 模块耦合度")

    coupling = engine.get_module_coupling_metrics()

    if not coupling:
        print("  (No coupling metrics available)")
        return

    print(f"  Analyzed {len(coupling)} modules:\n")

    # Show top 5 most coupled modules
    items = list(coupling.items())[:5]
    for module, metrics in items:
        if isinstance(metrics, dict):
            afferent = metrics.get("afferent_coupling", 0)
            efferent = metrics.get("efferent_coupling", 0)
            instability = metrics.get("instability", 0)
            print(f"    {module}:")
            print(
                f"      Afferent: {afferent}, Efferent: {efferent}, "
                f"Instability: {instability:.2f}"
            )
        else:
            print(f"    {module}: {metrics}")


# ---------------------------------------------------------------------------
# 7. Dead code detection / 死代码检测
# ---------------------------------------------------------------------------


def demo_dead_code(engine: PySearch) -> None:
    section("7. Dead Code Detection / 死代码检测")

    dead_modules = engine.find_dead_code()

    if not dead_modules:
        print("  No potentially dead modules found! ✓")
    else:
        print(f"  Found {len(dead_modules)} potentially unused modules:\n")
        for module in dead_modules[:10]:
            print(f"    - {module}")

        if len(dead_modules) > 10:
            print(f"    ... and {len(dead_modules) - 10} more")


# ---------------------------------------------------------------------------
# 8. Dependency path check / 依赖路径检查
# ---------------------------------------------------------------------------


def demo_dependency_path(engine: PySearch) -> None:
    section("8. Dependency Path Check / 依赖路径检查")

    graph = engine.analyze_dependencies()
    if graph is None or len(graph.nodes) < 2:
        print("  (Not enough modules for path check)")
        return

    nodes = sorted(graph.nodes)
    source, target = nodes[0], nodes[min(1, len(nodes) - 1)]

    has_path = engine.check_dependency_path(source, target)
    print(f"  {source} → {target}: {'PATH EXISTS' if has_path else 'NO PATH'}")

    # Reverse direction
    has_path_rev = engine.check_dependency_path(target, source)
    print(f"  {target} → {source}: {'PATH EXISTS' if has_path_rev else 'NO PATH'}")


# ---------------------------------------------------------------------------
# 9. Export dependency graph / 导出依赖图
# ---------------------------------------------------------------------------


def demo_export_graph(engine: PySearch) -> None:
    section("9. Export Dependency Graph / 导出依赖图")

    graph = engine.analyze_dependencies()
    if graph is None:
        print("  (No graph to export)")
        return

    # Export as DOT format (for Graphviz)
    dot_output = engine.export_dependency_graph(graph, format="dot")
    print("  DOT format (first 500 chars):")
    print(f"    {dot_output[:500]}")

    # Export as JSON
    json_output = engine.export_dependency_graph(graph, format="json")
    print("\n  JSON format (first 500 chars):")
    print(f"    {json_output[:500]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    search_path = sys.argv[1] if len(sys.argv) > 1 else "./src"

    if not Path(search_path).exists():
        print(f"Error: search path '{search_path}' does not exist.")
        sys.exit(1)

    config = SearchConfig(paths=[search_path], include=["**/*.py"], context=2)
    engine = PySearch(config)

    demo_dependency_graph(engine)
    demo_dependency_metrics(engine)
    demo_circular_deps(engine)
    demo_impact_analysis(engine)
    demo_refactoring_suggestions(engine)
    demo_coupling_metrics(engine)
    demo_dead_code(engine)
    demo_dependency_path(engine)
    demo_export_graph(engine)

    print("\n✓ All dependency analysis examples completed.")


if __name__ == "__main__":
    main()
