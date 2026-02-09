#!/usr/bin/env python3
"""
Example 10: Workspace & Multi-Repository Search — 工作区与多仓库搜索示例

Demonstrates:
- WorkspaceManager for workspace lifecycle / 工作区生命周期管理
- Repository auto-discovery / 仓库自动发现
- Multi-repository search / 多仓库搜索
- Cross-repository search / 跨仓库搜索
- Repository configuration / 仓库配置
- Workspace persistence (TOML) / 工作区持久化
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from pysearch import PySearch, SearchConfig
from pysearch.core.types import Query
from pysearch.core.workspace import WorkspaceManager

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# 1. WorkspaceManager basics / WorkspaceManager 基础
# ---------------------------------------------------------------------------


def demo_workspace_manager() -> None:
    section("1. WorkspaceManager Basics / WorkspaceManager 基础")

    manager = WorkspaceManager()

    # Create a workspace
    ws = manager.create_workspace(
        name="demo-workspace",
        root_path=".",
        description="Demo workspace for examples",
    )
    print(f"  Workspace created: {ws.name}")
    print(f"  Root path: {ws.root_path}")
    print(f"  Description: {ws.description}")
    print(f"  Repositories: {len(ws.repositories)}")

    # Save workspace to a temp file
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False, mode="w") as f:
        ws_path = f.name

    manager.save_workspace(ws, ws_path)
    print(f"\n  Saved to: {ws_path}")

    # Load workspace back
    loaded_ws = manager.load_workspace(ws_path)
    if loaded_ws:
        print(f"  Loaded workspace: {loaded_ws.name}")
        print(f"  Repositories: {len(loaded_ws.repositories)}")

    # Clean up
    Path(ws_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Repository auto-discovery / 仓库自动发现
# ---------------------------------------------------------------------------


def demo_auto_discovery() -> None:
    section("2. Repository Auto-Discovery / 仓库自动发现")

    manager = WorkspaceManager()
    ws = manager.create_workspace("auto-discover", ".")

    # Discover repositories (looks for .git directories)
    # 自动发现仓库（查找 .git 目录）
    discovered = manager.discover_repositories(ws, max_depth=2)

    print(f"  Discovered {len(discovered)} repositories:")
    for repo in discovered:
        print(f"    - {repo.name}: {repo.path}")
        print(f"      Type: {repo.project_type}, Priority: {repo.priority}")

    # Also show workspace repos (auto_add=True by default)
    print(f"\n  Workspace now has {len(ws.repositories)} repositories.")
    for repo in ws.repositories:
        print(f"    - {repo.name}: {repo.path}")


# ---------------------------------------------------------------------------
# 3. Create workspace via PySearch API / 通过 PySearch API 创建工作区
# ---------------------------------------------------------------------------


def demo_pysearch_workspace(engine: PySearch) -> None:
    section("3. PySearch Workspace API / PySearch 工作区 API")

    # Create workspace through engine
    success = engine.create_workspace(
        name="pysearch-workspace",
        root_path=".",
        description="Created via PySearch API",
        auto_discover=True,
        max_depth=2,
    )
    print(f"  Workspace created: {success}")

    # Get workspace summary
    summary = engine.get_workspace_summary()
    print("\n  Workspace summary:")
    for key, value in summary.items():
        if isinstance(value, int | float | str | bool):
            print(f"    {key}: {value}")

    # Save workspace
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        ws_path = f.name
    saved = engine.save_workspace(ws_path)
    print(f"\n  Workspace saved: {saved}")
    Path(ws_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 4. Multi-repository search / 多仓库搜索
# ---------------------------------------------------------------------------


def demo_multi_repo_search(engine: PySearch) -> None:
    section("4. Multi-Repository Search / 多仓库搜索")

    # Enable multi-repo search
    success = engine.enable_multi_repo(max_workers=4)
    print(f"  Multi-repo enabled: {success}")
    print(f"  Is multi-repo active: {engine.is_multi_repo_enabled()}")

    if not success:
        print("  (Skipping multi-repo demos)")
        return

    # Add repositories (using project subdirectories as examples)
    src_path = Path("./src")
    tests_path = Path("./tests")

    if src_path.exists():
        added = engine.add_repository(
            name="source",
            path=src_path,
            priority="high",
        )
        print(f"  Added 'source' repo: {added}")

    if tests_path.exists():
        added = engine.add_repository(
            name="tests",
            path=tests_path,
            priority="normal",
        )
        print(f"  Added 'tests' repo: {added}")

    # List repositories
    repos = engine.list_repositories()
    print(f"\n  Repositories: {repos}")

    # Get repository info
    for name in repos[:3]:
        info = engine.get_repository_info(name)
        if info:
            print(f"\n  Repository '{name}':")
            print(f"    Path: {info.path}")
            print(f"    Priority: {info.priority}")

    # Search across all repositories
    results = engine.search_all_repositories(
        pattern="def search",
        use_regex=False,
        context=1,
        max_results=20,
    )
    if results:
        print("\n  Cross-repo search for 'def search':")
        print(f"    Successful repos: {results.successful_repositories}")
        print(f"    Total matches: {results.total_matches}")
        print(f"    Search time: {results.search_time_ms:.0f}ms")
        for repo_name, repo_result in results.repository_results.items():
            print(f"      {repo_name}: {len(repo_result.items)} matches")

    # Configure a repository
    engine.configure_repository("source", priority="high")
    print("\n  Configured 'source' priority to 'high'")

    # Sync repositories
    sync_status = engine.sync_repositories()
    print(f"  Sync status: {sync_status}")

    # Health check
    health = engine.get_multi_repo_health()
    print(f"  Health: {len(health)} repos checked")

    # Stats
    stats = engine.get_multi_repo_stats()
    print(f"  Stats: {stats}")

    # Cleanup
    for name in repos:
        engine.remove_repository(name)
    engine.disable_multi_repo()
    print("\n  Multi-repo disabled and repos removed.")


# ---------------------------------------------------------------------------
# 5. Search specific repositories / 搜索特定仓库
# ---------------------------------------------------------------------------


def demo_search_specific_repos(engine: PySearch) -> None:
    section("5. Search Specific Repositories / 搜索特定仓库")

    engine.enable_multi_repo()

    src_path = Path("./src")
    if src_path.exists():
        engine.add_repository("source", src_path)

    tests_path = Path("./tests")
    if tests_path.exists():
        engine.add_repository("tests", tests_path)

    repos = engine.list_repositories()
    if len(repos) >= 1:
        query = Query(pattern="import", use_regex=False, context=0)
        results = engine.search_specific_repositories(
            repositories=repos[:1],
            query=query,
            max_results=10,
        )
        if results:
            print(f"  Search in {repos[:1]}: {results.total_matches} matches")

    # Cleanup
    for name in repos:
        engine.remove_repository(name)
    engine.disable_multi_repo()


# ---------------------------------------------------------------------------
# 6. Discover repositories via engine / 通过引擎发现仓库
# ---------------------------------------------------------------------------


def demo_discover_via_engine(engine: PySearch) -> None:
    section("6. Discover Repositories / 发现仓库")

    discovered = engine.discover_repositories(
        root_path=".",
        max_depth=2,
        auto_add=False,  # Don't auto-add, just discover
    )
    print(f"  Discovered {len(discovered)} repositories:")
    for repo_info in discovered[:5]:
        print(
            f"    - {repo_info.get('name', 'unknown')}: "
            f"{repo_info.get('project_type', 'unknown')}"
        )

    # Cleanup
    if engine.is_multi_repo_enabled():
        engine.disable_multi_repo()


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

    demo_workspace_manager()
    demo_auto_discovery()
    demo_pysearch_workspace(engine)
    demo_multi_repo_search(engine)
    demo_search_specific_repos(engine)
    demo_discover_via_engine(engine)

    print("\n✓ All workspace & multi-repo examples completed.")


if __name__ == "__main__":
    main()
