#!/usr/bin/env python3
"""
Example 07: Search History & Bookmarks — 搜索历史与书签示例

Demonstrates:
- Search history viewing and filtering / 搜索历史查看与过滤
- Bookmark management (create/delete/folders) / 书签管理
- Search analytics (frequency, trends, failed patterns) / 搜索分析
- History export/import/backup/restore / 历史导出/导入/备份/恢复
- Session management / 会话管理
"""

from __future__ import annotations

import sys
import tempfile
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
# 1. Generate some search history / 生成一些搜索历史
# ---------------------------------------------------------------------------


def generate_search_history(engine: PySearch) -> None:
    section("1. Generating Search History / 生成搜索历史")

    # Perform several searches to build history
    # 执行多次搜索以构建历史记录
    patterns = [
        "def search",
        "class Config",
        "import pathlib",
        "async def",
        "def __init__",
        "raise ValueError",
        "TODO",
    ]

    for pattern in patterns:
        results = engine.search(pattern)
        print(f"  Searched '{pattern}': {len(results.items)} matches")

    print(f"\n  Generated {len(patterns)} history entries.")


# ---------------------------------------------------------------------------
# 2. View search history / 查看搜索历史
# ---------------------------------------------------------------------------


def demo_view_history(engine: PySearch) -> None:
    section("2. View Search History / 查看搜索历史")

    # Get recent history
    history = engine.get_search_history(limit=10)
    print(f"  Total history entries: {len(history)}\n")

    for entry in history[:5]:
        print(f"  Pattern: '{entry.query_pattern}'")
        print(f"    Results: {entry.items_count}, Time: {entry.elapsed_ms:.0f}ms")
        print(
            f"    Category: {entry.category.value if hasattr(entry.category, 'value') else entry.category}"
        )
        print()


# ---------------------------------------------------------------------------
# 3. Frequent and recent patterns / 频繁和最近的模式
# ---------------------------------------------------------------------------


def demo_frequent_recent(engine: PySearch) -> None:
    section("3. Frequent & Recent Patterns / 频繁与最近模式")

    # Most frequently searched patterns
    frequent = engine.get_frequent_patterns(limit=5)
    print("  Most frequent patterns:")
    for pattern, count in frequent:
        print(f"    '{pattern}' — {count} times")

    # Recently used patterns
    recent = engine.get_recent_patterns(days=30, limit=5)
    print("\n  Recent patterns (last 30 days):")
    for pattern in recent:
        print(f"    '{pattern}'")


# ---------------------------------------------------------------------------
# 4. Pattern suggestions / 模式建议
# ---------------------------------------------------------------------------


def demo_pattern_suggestions(engine: PySearch) -> None:
    section("4. Pattern Suggestions / 模式建议")

    partials = ["def", "class", "imp"]
    for partial in partials:
        suggestions = engine.get_pattern_suggestions(partial, limit=3)
        print(f"  '{partial}' → {suggestions}")


# ---------------------------------------------------------------------------
# 5. Search ratings and tags / 搜索评分与标签
# ---------------------------------------------------------------------------


def demo_ratings_and_tags(engine: PySearch) -> None:
    section("5. Ratings & Tags / 评分与标签")

    # Rate a search
    engine.rate_last_search("def search", rating=5)
    print("  Rated 'def search' → 5 stars")

    # Add tags
    engine.add_search_tags("def search", ["core", "important"])
    print("  Tagged 'def search' → ['core', 'important']")

    # Search by tags
    tagged = engine.search_history_by_tags(["core"])
    print(f"  Entries tagged 'core': {len(tagged)}")


# ---------------------------------------------------------------------------
# 6. Bookmark management / 书签管理
# ---------------------------------------------------------------------------


def demo_bookmarks(engine: PySearch) -> None:
    section("6. Bookmarks / 书签管理")

    # Create a bookmark
    query = Query(pattern="def search", use_regex=False, context=2)
    results = engine.run(query)
    engine.add_bookmark("search-functions", query, results)
    print("  Created bookmark: 'search-functions'")

    # Create bookmark folder
    engine.create_bookmark_folder("core-searches", description="Core search patterns")
    print("  Created folder: 'core-searches'")

    # Add bookmark to folder
    engine.add_bookmark_to_folder("search-functions", "core-searches")
    print("  Added 'search-functions' to folder 'core-searches'")

    # List bookmarks
    bookmarks = engine.get_bookmarks()
    print(f"\n  Total bookmarks: {len(bookmarks)}")
    for name, entry in bookmarks.items():
        print(f"    '{name}': pattern='{entry.query_pattern}', " f"results={entry.items_count}")

    # List folders
    folders = engine.get_bookmark_folders()
    print(f"\n  Bookmark folders: {len(folders)}")
    for name, folder_info in folders.items():
        print(f"    '{name}': {folder_info}")

    # Get bookmarks in folder
    folder_bookmarks = engine.get_bookmarks_in_folder("core-searches")
    print(f"\n  Bookmarks in 'core-searches': {len(folder_bookmarks)}")

    # Cleanup
    engine.remove_bookmark_from_folder("search-functions", "core-searches")
    engine.delete_bookmark_folder("core-searches")
    engine.remove_bookmark("search-functions")
    print("  Cleaned up bookmark and folder.")


# ---------------------------------------------------------------------------
# 7. Search analytics / 搜索分析
# ---------------------------------------------------------------------------


def demo_analytics(engine: PySearch) -> None:
    section("7. Search Analytics / 搜索分析")

    analytics = engine.get_search_analytics(days=30)
    print("  Analytics (last 30 days):")
    for key, value in analytics.items():
        if isinstance(value, int | float | str):
            print(f"    {key}: {value}")

    # Detailed stats
    stats = engine.get_detailed_history_stats()
    print("\n  Detailed stats:")
    for key, value in list(stats.items())[:10]:
        if isinstance(value, int | float | str):
            print(f"    {key}: {value}")

    # Search trends
    trends = engine.get_search_trends(days=30)
    if trends:
        print(f"\n  Search trends: {len(trends)} data points")

    # Failed patterns
    failed = engine.get_top_failed_patterns(limit=5)
    if failed:
        print("\n  Top failed patterns:")
        for item in failed:
            print(f"    {item}")


# ---------------------------------------------------------------------------
# 8. Session management / 会话管理
# ---------------------------------------------------------------------------


def demo_sessions(engine: PySearch) -> None:
    section("8. Session Management / 会话管理")

    current = engine.get_current_session()
    if current:
        print(f"  Current session ID: {current.session_id}")
        print(f"    Total searches: {current.total_searches}")
        print(f"    Start time: {current.start_time}")

    sessions = engine.get_search_sessions(limit=3)
    print(f"\n  Total sessions: {len(sessions)}")
    for session in sessions[:3]:
        print(f"    Session {session.session_id}: " f"{session.total_searches} searches")

    # End current session
    engine.end_current_session()
    print("\n  Ended current session.")


# ---------------------------------------------------------------------------
# 9. History export/import / 历史导出/导入
# ---------------------------------------------------------------------------


def demo_export_import(engine: PySearch) -> None:
    section("9. Export / Import / Backup / 导出与备份")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Export to JSON
        json_path = str(Path(tmpdir) / "history.json")
        count = engine.export_history(json_path, fmt="json")
        print(f"  Exported {count} entries to {json_path}")

        # Export to string
        json_str = engine.export_history_to_string(fmt="json")
        print(f"  Export to string: {len(json_str)} chars")

        # Export to CSV
        csv_path = str(Path(tmpdir) / "history.csv")
        count = engine.export_history(csv_path, fmt="csv")
        print(f"  Exported {count} entries to CSV")

        # Backup
        backup_path = str(Path(tmpdir) / "backup.json")
        backup_counts = engine.backup_history(backup_path)
        print(f"  Backup created: {backup_counts}")

        # Validate backup
        validation = engine.validate_backup(backup_path)
        print(f"  Backup valid: {validation.get('valid', False)}")

        # Import (merge)
        import_count = engine.import_history(json_path, merge=True)
        print(f"  Imported (merge): {import_count} total entries")


# ---------------------------------------------------------------------------
# 10. History filtering / 历史过滤
# ---------------------------------------------------------------------------


def demo_history_filtering(engine: PySearch) -> None:
    section("10. History Filtering / 历史过滤")

    # Filter by category
    text_searches = engine.get_history_by_category("text", limit=3)
    print(f"  Text searches: {len(text_searches)}")

    # Search in history
    matching = engine.search_in_history("search", limit=5)
    print(f"  History entries matching 'search': {len(matching)}")

    # Filter by language
    python_searches = engine.get_history_by_language("python", limit=3)
    print(f"  Python-related searches: {len(python_searches)}")

    # Cleanup old history
    removed = engine.cleanup_old_history(days=365)
    print(f"  Cleaned up entries older than 365 days: {removed}")

    # Deduplicate
    deduped = engine.deduplicate_history()
    print(f"  Deduplicated entries: {deduped}")


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

    generate_search_history(engine)
    demo_view_history(engine)
    demo_frequent_recent(engine)
    demo_pattern_suggestions(engine)
    demo_ratings_and_tags(engine)
    demo_bookmarks(engine)
    demo_analytics(engine)
    demo_sessions(engine)
    demo_export_import(engine)
    demo_history_filtering(engine)

    print("\n✓ All history & bookmark examples completed.")


if __name__ == "__main__":
    main()
