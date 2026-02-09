#!/usr/bin/env python3
"""
Example 11: IDE Integration — IDE 集成示例

Demonstrates:
- Enable IDE integration / 启用 IDE 集成
- Jump to definition / 跳转到定义
- Find references / 查找引用
- Auto-completion / 自动补全
- Hover information / 悬停信息
- Document symbols / 文档符号
- Workspace symbols / 工作区符号
- Diagnostics / 诊断
- Structured query / 结构化查询
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig
from pysearch.core.types import Query

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. Enable IDE integration / 启用 IDE 集成
# ---------------------------------------------------------------------------


def demo_enable_ide(engine: PySearch) -> None:
    section("1. Enable IDE Integration / 启用 IDE 集成")

    success = engine.enable_ide_integration()
    print(f"  IDE integration enabled: {success}")
    print(f"  Is IDE enabled: {engine.is_ide_enabled()}")

    if not success:
        print("  (IDE integration could not be enabled)")


# ---------------------------------------------------------------------------
# 2. Jump to definition / 跳转到定义
# ---------------------------------------------------------------------------


def demo_jump_to_definition(engine: PySearch) -> None:
    section("2. Jump to Definition / 跳转到定义")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Try to find definition of 'SearchConfig'
    # 尝试查找 'SearchConfig' 的定义
    test_cases = [
        ("src/pysearch/core/api.py", 10, "SearchConfig"),
        ("src/pysearch/core/api.py", 10, "PySearch"),
        ("src/pysearch/core/api.py", 10, "Indexer"),
    ]

    for file_path, line, symbol in test_cases:
        location = engine.jump_to_definition(file_path, line, symbol)
        if location:
            print(f"  '{symbol}' defined at:")
            print(f"    File: {location.get('file', 'unknown')}")
            print(f"    Line: {location.get('line', 'unknown')}")
        else:
            print(f"  '{symbol}': definition not found")


# ---------------------------------------------------------------------------
# 3. Find references / 查找引用
# ---------------------------------------------------------------------------


def demo_find_references(engine: PySearch) -> None:
    section("3. Find References / 查找引用")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Find all references to 'SearchConfig'
    refs = engine.find_references(
        file_path="src/pysearch/core/config.py",
        line=58,
        symbol="SearchConfig",
        include_definition=True,
    )
    print(f"  References to 'SearchConfig': {len(refs)} found")
    for ref in refs[:5]:
        print(f"    {ref.get('file', '?')}:{ref.get('line', '?')}")

    if len(refs) > 5:
        print(f"    ... and {len(refs) - 5} more")


# ---------------------------------------------------------------------------
# 4. Auto-completion / 自动补全
# ---------------------------------------------------------------------------


def demo_completion(engine: PySearch) -> None:
    section("4. Auto-Completion / 自动补全")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Get completions for a partial identifier
    # 获取部分标识符的自动补全
    completions = engine.provide_completion(
        file_path="src/pysearch/core/api.py",
        line=100,
        column=10,
        prefix="search",
    )
    print(f"  Completions for 'search': {len(completions)} items")
    for comp in completions[:10]:
        label = comp.get("label", "?")
        kind = comp.get("kind", "?")
        print(f"    {label} ({kind})")

    # Completions for 'config'
    completions2 = engine.provide_completion(
        file_path="src/pysearch/core/api.py",
        line=100,
        column=10,
        prefix="config",
    )
    print(f"\n  Completions for 'config': {len(completions2)} items")
    for comp in completions2[:5]:
        print(f"    {comp.get('label', '?')} ({comp.get('kind', '?')})")


# ---------------------------------------------------------------------------
# 5. Hover information / 悬停信息
# ---------------------------------------------------------------------------


def demo_hover(engine: PySearch) -> None:
    section("5. Hover Information / 悬停信息")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    hover_info = engine.provide_hover(
        file_path="src/pysearch/core/api.py",
        line=89,
        column=10,
        symbol="PySearch",
    )
    if hover_info:
        print("  Hover info for 'PySearch':")
        for key, value in hover_info.items():
            val_str = str(value)[:100]
            print(f"    {key}: {val_str}")
    else:
        print("  No hover info available for 'PySearch'")


# ---------------------------------------------------------------------------
# 6. Document symbols / 文档符号
# ---------------------------------------------------------------------------


def demo_document_symbols(engine: PySearch) -> None:
    section("6. Document Symbols / 文档符号")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Get all symbols in a file
    # 获取文件中的所有符号
    symbols = engine.get_document_symbols("src/pysearch/core/config.py")
    print(f"  Symbols in config.py: {len(symbols)}")

    for sym in symbols[:10]:
        name = sym.get("name", "?")
        kind = sym.get("kind", "?")
        line = sym.get("line", "?")
        print(f"    L{line}: {name} ({kind})")

    if len(symbols) > 10:
        print(f"    ... and {len(symbols) - 10} more")


# ---------------------------------------------------------------------------
# 7. Workspace symbols / 工作区符号
# ---------------------------------------------------------------------------


def demo_workspace_symbols(engine: PySearch) -> None:
    section("7. Workspace Symbols / 工作区符号")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Search for symbols across the workspace
    # 在工作区中搜索符号
    symbols = engine.get_workspace_symbols("Config")
    print(f"  Workspace symbols matching 'Config': {len(symbols)}")

    for sym in symbols[:10]:
        name = sym.get("name", "?")
        kind = sym.get("kind", "?")
        file_path = sym.get("file", "?")
        print(f"    {name} ({kind}) — {file_path}")

    if len(symbols) > 10:
        print(f"    ... and {len(symbols) - 10} more")


# ---------------------------------------------------------------------------
# 8. Diagnostics / 诊断
# ---------------------------------------------------------------------------


def demo_diagnostics(engine: PySearch) -> None:
    section("8. Diagnostics / 诊断")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # Run diagnostics on a file (checks for TODO/FIXME/HACK, self-imports, etc.)
    # 对文件运行诊断（检查 TODO/FIXME/HACK、自导入等）
    diagnostics = engine.get_diagnostics("src/pysearch/core/api.py")
    print(f"  Diagnostics for api.py: {len(diagnostics)} issues")

    for diag in diagnostics[:10]:
        severity = diag.get("severity", "info")
        message = diag.get("message", "?")
        line = diag.get("line", "?")
        print(f"    L{line} [{severity}]: {message}")

    if len(diagnostics) > 10:
        print(f"    ... and {len(diagnostics) - 10} more")


# ---------------------------------------------------------------------------
# 9. Structured query for IDE / 结构化查询
# ---------------------------------------------------------------------------


def demo_structured_query(engine: PySearch) -> None:
    section("9. Structured Query / 结构化查询")

    if not engine.is_ide_enabled():
        print("  (IDE integration not enabled, skipping)")
        return

    # IDE-friendly structured query that returns JSON-serializable results
    # IDE 友好的结构化查询，返回可 JSON 序列化的结果
    query = Query(pattern="def search", use_regex=False, context=1)
    result = engine.ide_structured_query(query)

    items = result.get("items", [])
    stats = result.get("stats", {})

    print("  Structured query 'def search':")
    print(f"    Items: {len(items)}")
    print(f"    Stats: {stats}")

    for item in items[:3]:
        print(f"    → {item}")


# ---------------------------------------------------------------------------
# 10. Cleanup / 清理
# ---------------------------------------------------------------------------


def demo_disable_ide(engine: PySearch) -> None:
    section("10. Disable IDE Integration / 禁用 IDE 集成")

    engine.disable_ide_integration()
    print(f"  IDE integration disabled: {not engine.is_ide_enabled()}")


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

    demo_enable_ide(engine)
    demo_jump_to_definition(engine)
    demo_find_references(engine)
    demo_completion(engine)
    demo_hover(engine)
    demo_document_symbols(engine)
    demo_workspace_symbols(engine)
    demo_diagnostics(engine)
    demo_structured_query(engine)
    demo_disable_ide(engine)

    print("\n✓ All IDE integration examples completed.")


if __name__ == "__main__":
    main()
