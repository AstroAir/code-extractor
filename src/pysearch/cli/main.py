"""
Command-line interface for pysearch.

This module provides the CLI commands and argument parsing for the pysearch tool.
It serves as the primary entry point for command-line usage, offering a comprehensive
set of options for configuring and executing searches.

Main Commands:
    find: Execute a search with specified parameters

Key Features:
    - Comprehensive argument parsing with validation
    - Support for all search modes (text, regex, AST, semantic)
    - Flexible output formatting (text, JSON, highlighted)
    - Advanced filtering options (AST filters, metadata filters)
    - Performance statistics and debugging options
    - Integration with configuration files

Example Usage:
    Basic text search:
        $ pysearch find --pattern "def main" --path . --context 2

    Regex search with filters:
        $ pysearch find --pattern "def .*_handler" --regex \\
          --filter-func-name ".*handler" --format json

    AST-based search:
        $ pysearch find --pattern "def" --ast \\
          --filter-decorator "lru_cache" --stats

For more information, run: pysearch find --help
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import click

from ..core.api import PySearch
from ..core.config import SearchConfig
from ..utils.formatter import format_result, render_highlight_console
from ..utils.metadata_filters import create_metadata_filters
from ..core.types import ASTFilters, Language, OutputFormat, Query


@click.group()
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
    type=click.Choice(
        [lang.value for lang in Language if lang != Language.UNKNOWN]),
    help="限制搜索的编程语言",
)
@click.option("--pattern", required=True, help="文本/正则模式")
@click.option("--regex", is_flag=True, default=False, help="启用正则匹配")
@click.option("--fuzzy", is_flag=True, default=False, help="启用模糊搜索")
@click.option("--fuzzy-distance", type=int, default=2, help="模糊搜索最大编辑距离")
@click.option(
    "--fuzzy-algorithm",
    type=click.Choice(
        ["levenshtein", "damerau_levenshtein",
            "jaro_winkler", "soundex", "metaphone"]
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
    type=click.Choice(
        ["relevance", "frequency", "recency", "popularity", "hybrid"]),
    default="hybrid",
    help="结果排序策略",
)
@click.option("--cluster", is_flag=True, default=False, help="按相似性聚类结果")
@click.option("--ranking-analysis", is_flag=True, default=False, help="显示排序策略分析")
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
) -> None:
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

    # Check for conflicting options
    if fuzzy and regex:
        click.echo("Error: --fuzzy and --regex cannot be used together", err=True)
        sys.exit(1)

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
            use_ast=ast_filters is not None,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
            search_docstrings=cfg.enable_docstrings,
            search_comments=cfg.enable_comments,
            search_strings=cfg.enable_strings,
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
            f"# files_scanned={s.files_scanned} files_matched={s.files_matched} items={s.items} elapsed_ms={s.elapsed_ms:.2f} indexed={s.indexed_files}\n"
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
        click.echo(
            f"Recommended strategy: {analysis['recommended_strategy']}", err=True)
        click.echo(f"File spread: {analysis['file_spread']} files", err=True)
        click.echo(
            f"Result diversity: {analysis['result_diversity']:.2f}", err=True)
        if analysis["suggestions"]:
            click.echo("Suggestions:", err=True)
            for suggestion in analysis["suggestions"]:
                click.echo(f"  - {suggestion}", err=True)


@cli.command("history")
@click.option("--limit", type=int, default=20, help="限制显示的历史记录数量")
@click.option("--pattern", help="过滤包含特定模式的历史记录")
@click.option("--analytics", is_flag=True, help="显示搜索分析统计")
@click.option("--sessions", is_flag=True, help="显示搜索会话")
@click.option("--tags", help="按标签过滤历史记录")
def history_cmd(
    limit: int, pattern: str | None, analytics: bool, sessions: bool, tags: str | None
) -> None:
    """显示搜索历史记录和分析。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if analytics:
        # Show analytics
        stats = engine.get_search_analytics(days=30)
        click.echo("Search Analytics (Last 30 days)")
        click.echo("=" * 40)
        click.echo(f"Total searches: {stats['total_searches']}")
        click.echo(f"Successful searches: {stats['successful_searches']}")
        click.echo(f"Success rate: {stats['success_rate']:.1%}")
        click.echo(
            f"Average success score: {stats['average_success_score']:.2f}")
        click.echo(
            f"Average search time: {stats['average_search_time']:.1f}ms")
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
            start_time = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(session.start_time))
            duration = "ongoing"
            if session.end_time:
                duration = f"{(session.end_time - session.start_time) / 60:.1f} min"

            click.echo(
                f"Session {session.session_id[:8]}: {start_time} ({duration})")
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

        timestamp = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp))
        category_str = f"[{entry.category.value}]" if hasattr(
            entry, "category") else ""
        rating_str = (
            f"★{entry.user_rating}" if hasattr(
                entry, "user_rating") and entry.user_rating else ""
        )
        tags_str = f"#{','.join(entry.tags)}" if hasattr(
            entry, "tags") and entry.tags else ""

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
@click.option("--list-folders", is_flag=True, help="列出所有书签文件夹")
def bookmarks_cmd(
    add: str | None,
    remove: str | None,
    pattern: str | None,
    folder: str | None,
    create_folder: str | None,
    list_folders: bool,
) -> None:
    """管理搜索书签和文件夹。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if create_folder:
        if engine.create_bookmark_folder(create_folder):
            click.echo(f"Folder '{create_folder}' created")
        else:
            click.echo(f"Folder '{create_folder}' already exists")
    elif list_folders:
        folders = engine.get_bookmark_folders()
        if not folders:
            click.echo("No bookmark folders found.")
            return

        for name, folder_obj in folders.items():
            bookmark_count = len(
                folder_obj.bookmarks) if folder_obj.bookmarks else 0
            click.echo(f"{name}: {bookmark_count} bookmarks")
            if folder_obj.description:
                click.echo(f"  Description: {folder_obj.description}")
    elif add and pattern:
        # Add bookmark by running the search first
        result = engine.search(pattern)
        from ..core.types import OutputFormat, Query

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


def main() -> None:
    cli(prog_name="pysearch")


if __name__ == "__main__":
    main()
