"""
Command-line interface for pysearch.

This module provides the CLI commands and argument parsing for the pysearch tool.
It serves as the primary entry point for command-line usage, offering a comprehensive
set of options for configuring and executing searches.

Commands:
    find:       Execute a search with specified parameters (text, regex, fuzzy, boolean)
    semantic:   Semantic code search based on concepts
    history:    View and manage search history, analytics, and sessions
    bookmarks:  Manage search bookmarks and bookmark folders
    index:      Manage search index (stats, cleanup, rebuild)
    deps:       Analyze code dependencies and suggest refactoring
    watch:      Manage file watching for real-time index updates
    cache:      Manage search result caching
    config:     Display and validate configuration

Key Features:
    - Comprehensive argument parsing with validation
    - Support for all search modes (text, regex, AST, fuzzy, semantic, boolean)
    - Flexible output formatting (text, JSON, highlighted)
    - Advanced filtering options (AST filters, metadata filters)
    - Performance statistics and debugging options
    - Search history, analytics, and bookmarking
    - Index, cache, and file-watch management
    - Dependency analysis and refactoring suggestions

Example Usage:
    Basic text search:
        $ pysearch find "def main" --path . --context 2

    Regex search with filters:
        $ pysearch find "def .*_handler" --regex \\
          --filter-func-name ".*handler" --format json

    Semantic search:
        $ pysearch semantic "database connection" --path .

    Index management:
        $ pysearch index --stats
        $ pysearch index --rebuild

    Dependency analysis:
        $ pysearch deps --metrics
        $ pysearch deps --impact src.core.api

For more information, run: pysearch --help or pysearch <command> --help
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from ..core.api import PySearch
from ..core.config import SearchConfig
from ..core.types import ASTFilters, Language, OutputFormat, Query
from ..utils.formatter import format_result, render_highlight_console
from ..utils.metadata_filters import create_metadata_filters

try:
    from importlib.metadata import version as _pkg_version

    __cli_version__ = _pkg_version("pysearch")
except Exception:
    __cli_version__ = "0.1.0"


@click.group()
@click.version_option(version=__cli_version__, prog_name="pysearch")
def cli() -> None:
    """pysearch - Context-aware search engine for Python codebases"""
    pass


@cli.command("find")
@click.option("--path", "paths", multiple=True, default=["."], help="搜索路径，可多次提供")
@click.option("--include", multiple=True, help="包含的 glob 模式 (默认: 所有支持的文件类型)")
@click.option("--exclude", multiple=True, help="排除的 glob 模式")
@click.option(
    "--language",
    "languages",
    multiple=True,
    type=click.Choice([lang.value for lang in Language if lang != Language.UNKNOWN]),
    help="限制搜索的编程语言",
)
@click.option("--regex", is_flag=True, default=False, help="启用正则匹配")
@click.option("--fuzzy", is_flag=True, default=False, help="启用模糊搜索")
@click.option("--fuzzy-distance", type=int, default=2, help="模糊搜索最大编辑距离")
@click.option(
    "--fuzzy-algorithm",
    type=click.Choice(
        ["levenshtein", "damerau_levenshtein", "jaro_winkler", "soundex", "metaphone"]
    ),
    help="模糊搜索算法",
)
@click.option("--fuzzy-similarity", type=float, default=0.6, help="模糊搜索最小相似度 (0.0-1.0)")
@click.option("--context", type=int, default=2, help="上下文行数")
@click.option(
    "--format",
    "fmt",
    type=click.Choice([e.value for e in OutputFormat]),
    default=OutputFormat.TEXT.value,
    help="输出格式",
)
@click.option(
    "--filter-func-name", "filter_func", type=str, default=None, help="AST 过滤: 函数名正则"
)
@click.option(
    "--filter-class-name", "filter_class", type=str, default=None, help="AST 过滤: 类名正则"
)
@click.option(
    "--filter-decorator", "filter_deco", type=str, default=None, help="AST 过滤: 装饰器正则"
)
@click.option(
    "--filter-import",
    "filter_import",
    type=str,
    default=None,
    help="AST 过滤: 导入名正则（包含模块前缀）",
)
@click.option("--no-docstrings", is_flag=True, default=False, help="禁用 docstring 搜索")
@click.option("--no-comments", is_flag=True, default=False, help="禁用注释搜索")
@click.option("--no-strings", is_flag=True, default=False, help="禁用字符串字面量搜索")
@click.option("--stats", is_flag=True, default=False, help="打印性能统计")
# Metadata filters
@click.option("--min-size", help="最小文件大小 (例如: 1KB, 5MB)")
@click.option("--max-size", help="最大文件大小 (例如: 10MB, 1GB)")
@click.option("--modified-after", help="修改时间晚于 (例如: 2023-12-01, 1d, 1w)")
@click.option("--modified-before", help="修改时间早于")
@click.option("--created-after", help="创建时间晚于")
@click.option("--created-before", help="创建时间早于")
@click.option("--min-lines", type=int, help="最小行数")
@click.option("--max-lines", type=int, help="最大行数")
@click.option("--author", help="作者名称正则模式")
@click.option("--encoding", help="文件编码正则模式")
# Logging and debugging options
@click.option("--debug", is_flag=True, default=False, help="启用调试日志")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
    default="INFO",
    help="日志级别",
)
@click.option("--log-file", help="日志文件路径")
@click.option(
    "--log-format",
    type=click.Choice(["simple", "detailed", "json", "structured"]),
    default="simple",
    help="日志格式",
)
@click.option("--show-errors", is_flag=True, default=False, help="显示详细错误报告")
# Ranking and result organization options
@click.option(
    "--ranking",
    type=click.Choice(["relevance", "frequency", "recency", "popularity", "hybrid"]),
    default="hybrid",
    help="结果排序策略",
)
@click.option("--cluster", is_flag=True, default=False, help="按相似性聚类结果")
@click.option("--ranking-analysis", is_flag=True, default=False, help="显示排序策略分析")
# New features
@click.option(
    "--logic", "use_logic", is_flag=True, default=False, help="使用布尔逻辑查询 (AND, OR, NOT)"
)
@click.option("--count", is_flag=True, default=False, help="仅显示匹配计数，不显示内容")
@click.option("--max-per-file", type=int, help="每个文件的最大匹配数量")
@click.argument("pattern")
def find_cmd(
    paths: tuple[str, ...],
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    languages: tuple[str, ...],
    pattern: str,
    regex: bool,
    fuzzy: bool,
    fuzzy_distance: int,
    fuzzy_algorithm: str | None,
    fuzzy_similarity: float,
    context: int,
    fmt: str,
    filter_func: str | None,
    filter_class: str | None,
    filter_deco: str | None,
    filter_import: str | None,
    no_docstrings: bool,
    no_comments: bool,
    no_strings: bool,
    stats: bool,
    min_size: str | None,
    max_size: str | None,
    modified_after: str | None,
    modified_before: str | None,
    created_after: str | None,
    created_before: str | None,
    min_lines: int | None,
    max_lines: int | None,
    author: str | None,
    encoding: str | None,
    debug: bool,
    log_level: str,
    log_file: str | None,
    log_format: str,
    show_errors: bool,
    ranking: str,
    cluster: bool,
    ranking_analysis: bool,
    use_logic: bool,
    count: bool,
    max_per_file: int | None,
) -> None:
    """Execute a search with specified parameters."""
    # Parse language filters
    language_set = None
    if languages:
        language_set = {Language(lang) for lang in languages}

    cfg = SearchConfig(
        paths=list(paths) or ["."],
        # Use None to trigger auto-detection
        include=list(include) if include else None,
        exclude=list(exclude) if exclude else None,
        languages=language_set,
        context=context,
        output_format=OutputFormat(fmt),
        enable_docstrings=not no_docstrings,
        enable_comments=not no_comments,
        enable_strings=not no_strings,
    )
    # Check for conflicting options early (before engine initialization)
    if fuzzy and regex:
        click.echo("Error: --fuzzy and --regex cannot be used together", err=True)
        sys.exit(1)

    if use_logic and fuzzy:
        click.echo("Error: --logic and --fuzzy cannot be used together", err=True)
        sys.exit(1)

    if count and fmt == OutputFormat.HIGHLIGHT.value:
        click.echo("Error: --count cannot be used with highlight format", err=True)
        sys.exit(1)

    # Configure logging
    if debug:
        log_level = "DEBUG"

    from ..utils.logging_config import LogFormat, LogLevel, configure_logging

    try:
        configure_logging(
            level=LogLevel(log_level),
            format_type=LogFormat(log_format),
            log_file=Path(log_file) if log_file else None,
            enable_file=bool(log_file),
            enable_console=True,
        )
    except ValueError as e:
        click.echo(f"Error configuring logging: {e}", err=True)
        sys.exit(1)

    engine = PySearch(cfg)

    # AST filters
    ast_filters = None
    if any([filter_func, filter_class, filter_deco, filter_import]):
        ast_filters = ASTFilters(
            func_name=filter_func,
            class_name=filter_class,
            decorator=filter_deco,
            imported=filter_import,
        )

    # Metadata filters
    metadata_filters = None
    if any(
        [
            min_size,
            max_size,
            modified_after,
            modified_before,
            created_after,
            created_before,
            min_lines,
            max_lines,
            author,
            encoding,
        ]
    ):
        try:
            metadata_filters = create_metadata_filters(
                min_size=min_size,
                max_size=max_size,
                modified_after=modified_after,
                modified_before=modified_before,
                created_after=created_after,
                created_before=created_before,
                min_lines=min_lines,
                max_lines=max_lines,
                author_pattern=author,
                encoding_pattern=encoding,
                languages=language_set,
            )
        except ValueError as e:
            click.echo(f"Error in metadata filters: {e}", err=True)
            sys.exit(1)

    # Handle count-only search
    if count:
        count_result = engine.search_count_only(
            pattern=pattern,
            regex=regex,
            use_boolean=use_logic,
            filters=ast_filters,
            metadata_filters=metadata_filters,
        )
        # Output count results
        if OutputFormat(fmt) == OutputFormat.JSON:
            import orjson

            output_data = {
                "total_matches": count_result.total_matches,
                "files_matched": count_result.files_matched,
                "stats": {
                    "files_scanned": count_result.stats.files_scanned,
                    "elapsed_ms": count_result.stats.elapsed_ms,
                    "indexed_files": count_result.stats.indexed_files,
                },
            }
            sys.stdout.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2).decode())
        else:
            click.echo(f"Total matches: {count_result.total_matches}")
            click.echo(f"Files matched: {count_result.files_matched}")
            if stats:
                click.echo(f"Files scanned: {count_result.stats.files_scanned}")
                click.echo(f"Elapsed time: {count_result.stats.elapsed_ms:.2f}ms")
        return

    # Execute search based on type
    if fuzzy:
        # Use fuzzy search
        result = engine.fuzzy_search(
            pattern=pattern,
            max_distance=fuzzy_distance,
            min_similarity=fuzzy_similarity,
            algorithm=fuzzy_algorithm,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
        )
    elif ranking != "hybrid" or cluster:
        # Use advanced ranking search
        result = engine.search_with_ranking(
            pattern=pattern,
            ranking_strategy=ranking,
            cluster_results=cluster,
            regex=regex,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
        )
    else:
        # Use regular search
        query = Query(
            pattern=pattern,
            use_regex=regex,
            use_boolean=use_logic,
            use_ast=ast_filters is not None,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
            search_docstrings=cfg.enable_docstrings,
            search_comments=cfg.enable_comments,
            search_strings=cfg.enable_strings,
            count_only=count,
            max_per_file=max_per_file,
        )
        result = engine.run(query)

    if OutputFormat(fmt) == OutputFormat.HIGHLIGHT and sys.stdout.isatty():
        # 交互终端渲染彩色高亮
        render_highlight_console(result)
    else:
        sys.stdout.write(format_result(result, OutputFormat(fmt)))
        sys.stdout.write("\n")

    if stats and OutputFormat(fmt) != OutputFormat.TEXT:
        # 附加统计信息（文本）
        s = result.stats
        sys.stderr.write(
            f"# files_scanned={s.files_scanned} files_matched={s.files_matched} "
            f"items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}\n"
        )

    # Show error report if requested or if there are critical errors
    if show_errors or engine.has_critical_errors():
        error_summary = engine.get_error_summary()
        if error_summary["total_errors"] > 0:
            click.echo("\n" + "=" * 50, err=True)
            click.echo("ERROR REPORT", err=True)
            click.echo("=" * 50, err=True)
            click.echo(engine.get_error_report(), err=True)

            if engine.has_critical_errors():
                click.echo(
                    "\nCritical errors were encountered. Some results may be incomplete.", err=True
                )
                sys.exit(1)

    # Show ranking analysis if requested
    if ranking_analysis:
        analysis = engine.get_ranking_suggestions(pattern, result)
        click.echo("\n" + "=" * 50, err=True)
        click.echo("RANKING ANALYSIS", err=True)
        click.echo("=" * 50, err=True)
        click.echo(f"Query type: {analysis['query_type']}", err=True)
        click.echo(f"Recommended strategy: {analysis['recommended_strategy']}", err=True)
        click.echo(f"File spread: {analysis['file_spread']} files", err=True)
        click.echo(f"Result diversity: {analysis['result_diversity']:.2f}", err=True)
        if analysis["suggestions"]:
            click.echo("Suggestions:", err=True)
            for suggestion in analysis["suggestions"]:
                click.echo(f"  - {suggestion}", err=True)


@cli.command("history")
@click.option("--limit", type=int, default=20, help="限制显示的历史记录数量")
@click.option("--pattern", help="过滤包含特定模式的历史记录")
@click.option("--analytics", is_flag=True, help="显示搜索分析统计")
@click.option("--days", type=int, default=30, help="分析统计的天数范围 (默认: 30)")
@click.option("--sessions", is_flag=True, help="显示搜索会话")
@click.option("--tags", help="按标签过滤历史记录")
@click.option("--clear", "clear_history", is_flag=True, help="清除搜索历史")
@click.option("--frequent", is_flag=True, help="显示最常搜索的模式")
@click.option("--recent", is_flag=True, help="显示最近搜索的模式")
def history_cmd(
    limit: int,
    pattern: str | None,
    analytics: bool,
    days: int,
    sessions: bool,
    tags: str | None,
    clear_history: bool,
    frequent: bool,
    recent: bool,
) -> None:
    """显示搜索历史记录和分析。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if clear_history:
        engine.history.clear_history()
        click.echo("Search history cleared.")
        return

    if frequent:
        freq_patterns = engine.get_frequent_patterns(limit=limit)
        if not freq_patterns:
            click.echo("No search patterns found.")
            return
        click.echo("Most Frequent Search Patterns")
        click.echo("=" * 40)
        for pat, count in freq_patterns:
            click.echo(f"  {count:4d}x  {pat}")
        return

    if recent:
        recent_patterns = engine.get_recent_patterns(days=days, limit=limit)
        if not recent_patterns:
            click.echo("No recent search patterns found.")
            return
        click.echo(f"Recent Search Patterns (Last {days} days)")
        click.echo("=" * 40)
        for pat in recent_patterns:
            click.echo(f"  {pat}")
        return

    if analytics:
        # Show analytics
        stats = engine.get_search_analytics(days=days)
        click.echo(f"Search Analytics (Last {days} days)")
        click.echo("=" * 40)
        click.echo(f"Total searches: {stats['total_searches']}")
        click.echo(f"Successful searches: {stats['successful_searches']}")
        click.echo(f"Success rate: {stats['success_rate']:.1%}")
        click.echo(f"Average success score: {stats['average_success_score']:.2f}")
        click.echo(f"Average search time: {stats['average_search_time']:.1f}ms")
        click.echo(f"Sessions: {stats['session_count']}")

        if stats["most_common_categories"]:
            click.echo("\nMost common categories:")
            for category, count in stats["most_common_categories"]:
                click.echo(f"  {category}: {count}")

        if stats["most_used_languages"]:
            click.echo("\nMost used languages:")
            for language, count in stats["most_used_languages"]:
                click.echo(f"  {language}: {count}")

        return

    if sessions:
        # Show sessions
        session_list = engine.get_search_sessions(limit=10)
        if not session_list:
            click.echo("No search sessions found.")
            return

        click.echo("Recent Search Sessions")
        click.echo("=" * 40)
        for session in session_list:
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(session.start_time))
            duration = "ongoing"
            if session.end_time:
                duration = f"{(session.end_time - session.start_time) / 60:.1f} min"

            click.echo(f"Session {session.session_id[:8]}: {start_time} ({duration})")
            click.echo(
                f"  Searches: {session.total_searches} (success: {session.successful_searches})"
            )
            if session.queries:
                click.echo(f"  Patterns: {', '.join(session.queries[:3])}")
                if len(session.queries) > 3:
                    click.echo(f"    ... and {len(session.queries) - 3} more")
            click.echo()

        return

    if tags:
        # Search by tags
        tag_list = [tag.strip() for tag in tags.split(",")]
        history_entries = engine.search_history_by_tags(tag_list)
    else:
        # Regular history
        history_entries = engine.get_search_history(limit)

    if not history_entries:
        click.echo("No search history found.")
        return

    for entry in history_entries:
        if pattern and pattern.lower() not in entry.query_pattern.lower():
            continue

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
        category_str = f"[{entry.category.value}]" if hasattr(entry, "category") else ""
        rating_str = (
            f"★{entry.user_rating}" if hasattr(entry, "user_rating") and entry.user_rating else ""
        )
        tags_str = f"#{','.join(entry.tags)}" if hasattr(entry, "tags") and entry.tags else ""

        click.echo(
            f"{timestamp} {category_str} - '{entry.query_pattern}' "
            f"({entry.files_matched} files, {entry.items_count} items, "
            f"{entry.elapsed_ms:.1f}ms) {rating_str} {tags_str}"
        )


@cli.command("bookmarks")
@click.option("--add", help="添加书签名称")
@click.option("--remove", help="删除书签名称")
@click.option("--pattern", help="搜索模式（用于添加书签）")
@click.option("--folder", help="书签文件夹名称")
@click.option("--create-folder", help="创建新的书签文件夹")
@click.option("--delete-folder", help="删除书签文件夹")
@click.option("--description", help="文件夹描述（用于创建文件夹）")
@click.option("--list-folders", is_flag=True, help="列出所有书签文件夹")
def bookmarks_cmd(
    add: str | None,
    remove: str | None,
    pattern: str | None,
    folder: str | None,
    create_folder: str | None,
    delete_folder: str | None,
    description: str | None,
    list_folders: bool,
) -> None:
    """管理搜索书签和文件夹。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if create_folder:
        if engine.create_bookmark_folder(create_folder, description):
            click.echo(f"Folder '{create_folder}' created")
        else:
            click.echo(f"Folder '{create_folder}' already exists")
    elif delete_folder:
        if engine.delete_bookmark_folder(delete_folder):
            click.echo(f"Folder '{delete_folder}' deleted")
        else:
            click.echo(f"Folder '{delete_folder}' not found")
    elif list_folders:
        folders = engine.get_bookmark_folders()
        if not folders:
            click.echo("No bookmark folders found.")
            return

        for name, folder_obj in folders.items():
            bookmark_count = len(folder_obj.bookmarks) if folder_obj.bookmarks else 0
            click.echo(f"{name}: {bookmark_count} bookmarks")
            if folder_obj.description:
                click.echo(f"  Description: {folder_obj.description}")
    elif add:
        if not pattern:
            click.echo("Error: --pattern is required when adding a bookmark", err=True)
            sys.exit(1)
        # Add bookmark by running the search first
        result = engine.search(pattern)
        query = Query(pattern=pattern, output=OutputFormat.TEXT)
        engine.add_bookmark(add, query, result)

        # Add to folder if specified
        if folder:
            engine.add_bookmark_to_folder(add, folder)

        click.echo(f"Bookmark '{add}' added for pattern '{pattern}'")
        if folder:
            click.echo(f"Added to folder '{folder}'")
    elif remove:
        if engine.remove_bookmark(remove):
            click.echo(f"Bookmark '{remove}' removed")
        else:
            click.echo(f"Bookmark '{remove}' not found")
    elif folder:
        # List bookmarks in folder
        bookmarks = engine.get_bookmarks_in_folder(folder)
        if not bookmarks:
            click.echo(f"No bookmarks found in folder '{folder}'.")
            return

        click.echo(f"Bookmarks in folder '{folder}':")
        for entry in bookmarks:
            click.echo(
                f"  '{entry.query_pattern}' "
                f"({entry.files_matched} files, {entry.items_count} items)"
            )
    else:
        # List all bookmarks
        all_bookmarks = engine.get_bookmarks()
        if not all_bookmarks or not isinstance(all_bookmarks, dict):
            click.echo("No bookmarks found.")
            return

        for name, entry in all_bookmarks.items():
            click.echo(
                f"{name}: '{entry.query_pattern}' "
                f"({entry.files_matched} files, {entry.items_count} items)"
            )


@cli.command("semantic")
@click.option("--path", "paths", multiple=True, default=["."], help="搜索路径，可多次提供")
@click.option("--threshold", type=float, default=0.1, help="语义相似度阈值 (0.0-1.0)")
@click.option("--max-results", type=int, default=100, help="最大结果数量")
@click.option(
    "--format",
    "fmt",
    type=click.Choice([e.value for e in OutputFormat]),
    default=OutputFormat.TEXT.value,
    help="输出格式",
)
@click.option("--stats", is_flag=True, default=False, help="打印性能统计")
@click.option("--context", type=int, default=2, help="上下文行数")
@click.argument("query")
def semantic_cmd(
    paths: tuple[str, ...],
    threshold: float,
    max_results: int,
    fmt: str,
    stats: bool,
    context: int,
    query: str,
) -> None:
    """基于代码概念的语义搜索。"""
    cfg = SearchConfig(paths=list(paths) or ["."], context=context)
    engine = PySearch(cfg)

    try:
        result = engine.semantic_search(
            concept=query,
            context=context,
            output=OutputFormat(fmt),
        )
    except Exception as e:
        click.echo(f"Error during semantic search: {e}", err=True)
        sys.exit(1)

    # Limit results
    if len(result.items) > max_results:
        result.items = result.items[:max_results]

    if OutputFormat(fmt) == OutputFormat.HIGHLIGHT and sys.stdout.isatty():
        render_highlight_console(result)
    else:
        sys.stdout.write(format_result(result, OutputFormat(fmt)))
        sys.stdout.write("\n")

    if stats:
        s = result.stats
        sys.stderr.write(
            f"# files_matched={s.files_matched} items={s.items} "
            f"elapsed_ms={s.elapsed_ms:.2f}\n"
        )


@cli.command("index")
@click.option("--path", "paths", multiple=True, default=["."], help="索引路径")
@click.option("--stats", is_flag=True, default=False, help="显示索引统计信息")
@click.option("--cleanup", type=int, help="清理指定天数之前的旧缓存条目")
@click.option("--rebuild", is_flag=True, default=False, help="强制重建索引")
def index_cmd(
    paths: tuple[str, ...],
    stats: bool,
    cleanup: int | None,
    rebuild: bool,
) -> None:
    """管理搜索索引。"""
    cfg = SearchConfig(paths=list(paths) or ["."])
    engine = PySearch(cfg)

    if stats:
        idx_stats = engine.get_indexer_stats()
        click.echo("Index Statistics")
        click.echo("=" * 40)
        for key, value in idx_stats.items():
            click.echo(f"  {key}: {value}")
        return

    if cleanup is not None:
        removed = engine.cleanup_old_cache_entries(days_old=cleanup)
        click.echo(f"Cleaned up {removed} old cache entries (older than {cleanup} days)")
        return

    if rebuild:
        engine.indexer.clear()
        changed, removed, total = engine.indexer.scan()
        engine.indexer.save()
        click.echo(f"Index rebuilt: {total} files indexed, {len(changed or [])} changed")
        return

    # Default: show index summary
    try:
        changed, removed, total = engine.indexer.scan()
        engine.indexer.save()
        click.echo(
            f"Index status: {total} files tracked, "
            f"{len(changed or [])} changed, {len(removed or [])} removed"
        )
    except Exception as e:
        click.echo(f"Error scanning index: {e}", err=True)
        sys.exit(1)


@cli.command("deps")
@click.option("--path", "search_path", default=".", help="分析路径")
@click.option("--recursive/--no-recursive", default=True, help="是否递归分析子目录")
@click.option("--metrics", is_flag=True, default=False, help="显示依赖度量指标")
@click.option("--impact", help="分析指定模块的变更影响")
@click.option("--suggest", is_flag=True, default=False, help="建议重构机会")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def deps_cmd(
    search_path: str,
    recursive: bool,
    metrics: bool,
    impact: str | None,
    suggest: bool,
    fmt: str,
) -> None:
    """分析代码依赖关系。"""
    cfg = SearchConfig(paths=[search_path])
    engine = PySearch(cfg)

    if metrics:
        try:
            dep_metrics = engine.get_dependency_metrics()
        except Exception as e:
            click.echo(f"Error calculating metrics: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(dep_metrics, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Dependency Metrics")
            click.echo("=" * 40)
            if hasattr(dep_metrics, "__dict__"):
                for key, value in vars(dep_metrics).items():
                    click.echo(f"  {key}: {value}")
            else:
                click.echo(str(dep_metrics))
        return

    if impact:
        try:
            impact_result = engine.find_dependency_impact(impact)
        except Exception as e:
            click.echo(f"Error analyzing impact: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(impact_result, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Impact Analysis for '{impact}'")
            click.echo("=" * 40)
            for key, value in impact_result.items():
                if isinstance(value, list):
                    click.echo(f"  {key}:")
                    for item in value[:20]:
                        click.echo(f"    - {item}")
                else:
                    click.echo(f"  {key}: {value}")
        return

    if suggest:
        try:
            suggestions = engine.suggest_refactoring_opportunities()
        except Exception as e:
            click.echo(f"Error generating suggestions: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(suggestions, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            if not suggestions:
                click.echo("No refactoring suggestions found.")
                return
            click.echo("Refactoring Suggestions")
            click.echo("=" * 40)
            for s in suggestions:
                priority = s.get("priority", "medium").upper()
                click.echo(f"  [{priority}] {s.get('type', 'unknown')}")
                click.echo(f"    {s.get('description', '')}")
                if s.get("rationale"):
                    click.echo(f"    Rationale: {s['rationale']}")
                click.echo()
        return

    # Default: analyze and show summary
    try:
        graph = engine.analyze_dependencies(Path(search_path), recursive)
        if graph:
            click.echo("Dependency Analysis")
            click.echo("=" * 40)
            if hasattr(graph, "nodes"):
                click.echo(f"  Modules: {len(graph.nodes)}")
            if hasattr(graph, "edges"):
                click.echo(f"  Dependencies: {len(graph.edges)}")
            click.echo("\nUse --metrics, --impact, or --suggest for detailed analysis.")
        else:
            click.echo("No dependencies found or analysis not available.")
    except Exception as e:
        click.echo(f"Error analyzing dependencies: {e}", err=True)
        sys.exit(1)


@cli.command("watch")
@click.option("--path", "paths", multiple=True, default=["."], help="监视路径")
@click.option("--enable", is_flag=True, default=False, help="启用文件监视")
@click.option("--disable", "disable_watch", is_flag=True, default=False, help="禁用文件监视")
@click.option("--status", is_flag=True, default=False, help="显示监视状态")
@click.option("--debounce", type=float, default=0.5, help="去抖延迟（秒）")
def watch_cmd(
    paths: tuple[str, ...],
    enable: bool,
    disable_watch: bool,
    status: bool,
    debounce: float,
) -> None:
    """管理文件监视以实现实时索引更新。"""
    cfg = SearchConfig(paths=list(paths) or ["."])
    engine = PySearch(cfg)

    if status:
        is_enabled = engine.is_auto_watch_enabled()
        click.echo(f"File watching: {'ENABLED' if is_enabled else 'DISABLED'}")
        if is_enabled:
            watchers = engine.list_watchers()
            click.echo(f"Active watchers: {len(watchers)}")
            for name in watchers:
                click.echo(f"  - {name}")
            watch_stats = engine.get_watch_stats()
            if watch_stats:
                for name, watcher_stats in watch_stats.items():
                    click.echo(f"\n  {name}:")
                    for key, value in watcher_stats.items():
                        click.echo(f"    {key}: {value}")
        return

    if enable:
        success = engine.enable_auto_watch(debounce_delay=debounce)
        if success:
            click.echo("File watching enabled. Index will update automatically.")
        else:
            click.echo("Failed to enable file watching.", err=True)
            sys.exit(1)
        return

    if disable_watch:
        engine.disable_auto_watch()
        click.echo("File watching disabled.")
        return

    # Default: show status
    click.echo(f"File watching: {'ENABLED' if engine.is_auto_watch_enabled() else 'DISABLED'}")
    click.echo("Use --enable or --disable to manage file watching.")


@cli.command("cache")
@click.option(
    "--enable",
    "enable_backend",
    type=click.Choice(["memory", "disk"]),
    help="启用缓存 (memory/disk)",
)
@click.option("--disable", "disable_cache", is_flag=True, default=False, help="禁用缓存")
@click.option("--clear", "clear_cache", is_flag=True, default=False, help="清除所有缓存")
@click.option("--stats", is_flag=True, default=False, help="显示缓存统计")
@click.option("--cache-dir", help="磁盘缓存目录（仅 disk 模式）")
@click.option("--max-size", type=int, default=1000, help="最大缓存条目数")
@click.option("--ttl", type=float, default=3600, help="缓存过期时间（秒）")
def cache_cmd(
    enable_backend: str | None,
    disable_cache: bool,
    clear_cache: bool,
    stats: bool,
    cache_dir: str | None,
    max_size: int,
    ttl: float,
) -> None:
    """管理搜索结果缓存。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if stats:
        cache_stats = engine.get_cache_stats()
        if not cache_stats:
            click.echo("Caching is not enabled. No statistics available.")
            return
        click.echo("Cache Statistics")
        click.echo("=" * 40)
        for key, value in cache_stats.items():
            click.echo(f"  {key}: {value}")
        return

    if clear_cache:
        engine.clear_cache()
        engine.clear_caches()
        click.echo("All caches cleared.")
        return

    if enable_backend:
        success = engine.enable_caching(
            backend=enable_backend,
            cache_dir=cache_dir,
            max_size=max_size,
            default_ttl=ttl,
        )
        if success:
            click.echo(f"Caching enabled with {enable_backend} backend.")
        else:
            click.echo("Failed to enable caching.", err=True)
            sys.exit(1)
        return

    if disable_cache:
        engine.disable_caching()
        click.echo("Caching disabled.")
        return

    # Default: show status
    click.echo(f"Caching: {'ENABLED' if engine.is_caching_enabled() else 'DISABLED'}")
    click.echo("Use --enable, --disable, --clear, or --stats for cache management.")


@cli.command("config")
@click.option("--path", "paths", multiple=True, default=["."], help="项目路径")
@click.option("--validate", is_flag=True, default=False, help="验证高级功能配置")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def config_cmd(
    paths: tuple[str, ...],
    validate: bool,
    fmt: str,
) -> None:
    """显示和验证配置信息。"""
    cfg = SearchConfig(paths=list(paths) or ["."])

    if validate:
        issues = cfg.validate_optional_config()
        if issues:
            click.echo("Configuration Issues:", err=True)
            for issue in issues:
                click.echo(f"  - {issue}", err=True)
            sys.exit(1)
        else:
            click.echo("Configuration is valid.")
        return

    # Display configuration
    if fmt == "json":
        import orjson

        config_dict: dict[str, object] = {}
        for field_name in cfg.__dataclass_fields__:
            value = getattr(cfg, field_name)
            if isinstance(value, set):
                value = sorted(v.value if hasattr(v, "value") else str(v) for v in value)
            elif isinstance(value, Path):
                value = str(value)
            elif hasattr(value, "value"):  # Enum
                value = value.value
            config_dict[field_name] = value
        sys.stdout.write(
            orjson.dumps(config_dict, option=orjson.OPT_INDENT_2, default=str).decode()
        )
        sys.stdout.write("\n")
    else:
        click.echo("Current Configuration")
        click.echo("=" * 40)
        click.echo(f"  Paths: {cfg.paths}")
        include_patterns = cfg.get_include_patterns()
        click.echo(f"  Include patterns: {len(include_patterns)} patterns")
        exclude_patterns = cfg.get_exclude_patterns()
        click.echo(f"  Exclude patterns: {len(exclude_patterns)} patterns")
        click.echo(f"  Languages: {cfg.languages or 'auto-detect (all supported)'}")
        click.echo(f"  Context lines: {cfg.context}")
        click.echo(f"  Output format: {cfg.output_format.value}")
        click.echo(f"  Parallel: {cfg.parallel} (workers: {cfg.workers or 'auto'})")
        click.echo(f"  Cache dir: {cfg.resolve_cache_dir()}")
        click.echo(f"  File size limit: {cfg.file_size_limit / 1024 / 1024:.1f}MB")
        click.echo(f"  Follow symlinks: {cfg.follow_symlinks}")
        click.echo(f"  Strict hash check: {cfg.strict_hash_check}")
        click.echo(f"  Dir prune exclude: {cfg.dir_prune_exclude}")
        click.echo(f"  Content toggles: docstrings={cfg.enable_docstrings}, "
                    f"comments={cfg.enable_comments}, strings={cfg.enable_strings}")
        if cfg.is_optional_features_enabled():
            click.echo("\n  Optional Features:")
            click.echo(f"    GraphRAG: {cfg.enable_graphrag}")
            click.echo(f"    Metadata indexing: {cfg.enable_metadata_indexing}")
            click.echo(f"    Qdrant: {cfg.qdrant_enabled}")


def main() -> None:
    cli(prog_name="pysearch")


if __name__ == "__main__":
    main()
