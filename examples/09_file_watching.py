#!/usr/bin/env python3
"""
Example 09: File Watching — 文件监控示例

Demonstrates:
- Auto-watch for real-time index updates / 实时索引更新
- Custom file watchers with callbacks / 自定义文件监控回调
- Watcher management (list, pause, resume, remove) / 监控管理
- Watch statistics and performance metrics / 监控统计
- Watch filters / 监控过滤器

Note: File watching requires the 'watch' optional dependency:
    pip install -e ".[watch]"
"""

from __future__ import annotations

import sys
import time
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
# 1. Enable auto-watch / 启用自动监控
# ---------------------------------------------------------------------------


def demo_auto_watch(engine: PySearch) -> None:
    section("1. Auto-Watch / 自动监控")

    # Enable auto-watch — index updates automatically when files change
    # 启用自动监控 — 文件变更时自动更新索引
    success = engine.enable_auto_watch(
        debounce_delay=0.5,  # Wait 0.5s before processing changes
        batch_size=50,  # Batch up to 50 changes together
        max_batch_delay=5.0,  # Max wait before processing batch
    )
    print(f"  Auto-watch enabled: {success}")
    print(f"  Is auto-watch active: {engine.is_auto_watch_enabled()}")

    if success:
        # Search with auto-watch active — no manual index refresh needed
        # 在自动监控活跃时搜索 — 无需手动刷新索引
        results = engine.search("def search")
        print(f"  Search with auto-watch: {len(results.items)} matches")

        # Disable auto-watch
        engine.disable_auto_watch()
        print("  Auto-watch disabled.")
    else:
        print("  (watchdog may not be installed — pip install -e '.[watch]')")


# ---------------------------------------------------------------------------
# 2. Custom file watcher / 自定义文件监控
# ---------------------------------------------------------------------------


def demo_custom_watcher(engine: PySearch) -> None:
    section("2. Custom File Watcher / 自定义文件监控")

    # Track file change events
    events_log: list[str] = []

    def my_handler(events: list) -> None:
        """Custom handler that logs file events."""
        for event in events:
            msg = f"{event.event_type.value}: {event.path}"
            events_log.append(msg)
            print(f"    [EVENT] {msg}")

    # Add custom watcher for the src directory
    search_path = engine.cfg.paths[0] if engine.cfg.paths else "."
    success = engine.add_custom_watcher(
        name="my_watcher",
        path=search_path,
        change_handler=my_handler,
    )
    print(f"  Custom watcher added: {success}")

    if success:
        # List active watchers
        watchers = engine.list_watchers()
        print(f"  Active watchers: {watchers}")

        # Get watcher status
        status = engine.get_watcher_status("my_watcher")
        print(f"  Watcher status: {status}")

        # Wait briefly for any events
        print("  Waiting 1s for events...")
        time.sleep(1)
        print(f"  Events captured: {len(events_log)}")

        # Remove watcher
        removed = engine.remove_watcher("my_watcher")
        print(f"  Watcher removed: {removed}")
    else:
        print("  (watchdog may not be installed)")


# ---------------------------------------------------------------------------
# 3. Watch filters / 监控过滤器
# ---------------------------------------------------------------------------


def demo_watch_filters(engine: PySearch) -> None:
    section("3. Watch Filters / 监控过滤器")

    # Set file patterns for what to watch
    # 设置要监控的文件模式
    engine.set_watch_filters(
        include_patterns=["**/*.py", "**/*.toml"],
        exclude_patterns=["**/__pycache__/**", "**/.git/**"],
    )
    print("  Watch filters set:")
    print("    Include: *.py, *.toml")
    print("    Exclude: __pycache__, .git")


# ---------------------------------------------------------------------------
# 4. Pause and resume watching / 暂停和恢复监控
# ---------------------------------------------------------------------------


def demo_pause_resume(engine: PySearch) -> None:
    section("4. Pause & Resume Watching / 暂停与恢复监控")

    success = engine.enable_auto_watch()
    if not success:
        print("  (Auto-watch not available)")
        return

    print(f"  Auto-watch active: {engine.is_auto_watch_enabled()}")

    # Pause all watchers
    engine.pause_watching()
    print("  Watchers paused.")

    # Resume watching
    engine.resume_watching()
    print("  Watchers resumed.")

    # Force rescan
    rescan_ok = engine.force_rescan()
    print(f"  Force rescan: {rescan_ok}")

    engine.disable_auto_watch()
    print("  Auto-watch disabled.")


# ---------------------------------------------------------------------------
# 5. Watch statistics / 监控统计
# ---------------------------------------------------------------------------


def demo_watch_stats(engine: PySearch) -> None:
    section("5. Watch Statistics / 监控统计")

    success = engine.enable_auto_watch()
    if not success:
        print("  (Auto-watch not available)")
        return

    # Get watch stats
    stats = engine.get_watch_stats()
    print("  Watch stats:")
    for name, watcher_stats in stats.items():
        print(f"    {name}: {watcher_stats}")

    # Performance metrics
    perf = engine.get_watch_performance_metrics()
    print("\n  Performance metrics:")
    for key, value in perf.items():
        if isinstance(value, int | float | str | bool):
            print(f"    {key}: {value}")

    engine.disable_auto_watch()


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

    demo_auto_watch(engine)
    demo_custom_watcher(engine)
    demo_watch_filters(engine)
    demo_pause_resume(engine)
    demo_watch_stats(engine)

    print("\n✓ All file watching examples completed.")


if __name__ == "__main__":
    main()
