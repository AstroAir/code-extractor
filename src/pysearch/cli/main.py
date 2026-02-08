"""
Command-line interface for pysearch.

This module provides the CLI commands and argument parsing for the pysearch tool.
It serves as the primary entry point for command-line usage, offering a comprehensive
set of options for configuring and executing searches.

Commands:
    find:       Execute a search with specified parameters (text, regex, fuzzy, boolean, phonetic, multi-fuzzy)
    semantic:   Semantic code search based on concepts (embedding-based similarity)
    history:    View and manage search history, analytics, sessions, ratings, and tags
    bookmarks:  Manage search bookmarks and bookmark folders
    index:      Manage search index (stats, cleanup, rebuild)
    deps:       Analyze code dependencies and suggest refactoring
    watch:      Manage file watching for real-time index updates
    cache:      Manage search result caching (memory/disk, compression, invalidation)
    config:     Display and validate configuration
    repo:       Manage multi-repository search (enable, add, remove, configure, search across repos)
    ide:        IDE integration (jump-to-definition, find-references, completion, hover, symbols, diagnostics)
    distributed: Manage distributed indexing (enable, index, scale workers, stats, metrics)
    errors:     View and manage error diagnostics (summary, category, suppress, report)
    suggest:    Spelling correction suggestions based on codebase identifiers

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
@click.option(
    "--multi-fuzzy",
    is_flag=True,
    default=False,
    help="使用多算法模糊搜索（合并多种算法结果）",
)
@click.option(
    "--fuzzy-algorithms",
    multiple=True,
    type=click.Choice(
        ["levenshtein", "damerau_levenshtein", "jaro_winkler", "soundex", "metaphone"]
    ),
    help="多算法模糊搜索使用的算法列表",
)
@click.option(
    "--phonetic",
    is_flag=True,
    default=False,
    help="使用语音相似搜索（搜索发音相似的词）",
)
@click.option(
    "--phonetic-algorithm",
    type=click.Choice(["soundex", "metaphone"]),
    default="soundex",
    help="语音搜索算法",
)
@click.option(
    "--word-fuzzy",
    is_flag=True,
    default=False,
    help="使用词级模糊搜索（基于真实编辑距离算法逐词比较，而非正则近似）",
)
@click.option(
    "--group-by-file",
    is_flag=True,
    default=False,
    help="按文件分组显示搜索结果",
)
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
    multi_fuzzy: bool,
    fuzzy_algorithms: tuple[str, ...],
    phonetic: bool,
    phonetic_algorithm: str,
    word_fuzzy: bool,
    group_by_file: bool,
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

    if phonetic and (fuzzy or multi_fuzzy or regex):
        click.echo("Error: --phonetic cannot be used with --fuzzy, --multi-fuzzy, or --regex", err=True)
        sys.exit(1)

    if multi_fuzzy and regex:
        click.echo("Error: --multi-fuzzy and --regex cannot be used together", err=True)
        sys.exit(1)

    if word_fuzzy and (regex or fuzzy or multi_fuzzy or phonetic):
        click.echo("Error: --word-fuzzy cannot be used with --regex, --fuzzy, --multi-fuzzy, or --phonetic", err=True)
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
            max_per_file=max_per_file,
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
    if word_fuzzy:
        # Use word-level fuzzy search with real similarity algorithms
        result = engine.word_level_fuzzy_search(
            pattern=pattern,
            max_distance=fuzzy_distance,
            min_similarity=fuzzy_similarity,
            algorithms=list(fuzzy_algorithms) if fuzzy_algorithms else None,
            max_results=max_per_file * 100 if max_per_file else 1000,
            context=context,
        )
    elif phonetic:
        # Use phonetic (sound-alike) search
        result = engine.phonetic_search(
            pattern=pattern,
            algorithm=phonetic_algorithm,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
        )
    elif multi_fuzzy:
        # Use multi-algorithm fuzzy search
        result = engine.multi_algorithm_fuzzy_search(
            pattern=pattern,
            algorithms=list(fuzzy_algorithms) if fuzzy_algorithms else None,
            max_distance=fuzzy_distance,
            min_similarity=fuzzy_similarity,
            context=context,
            output=OutputFormat(fmt),
            filters=ast_filters,
            metadata_filters=metadata_filters,
        )
    elif fuzzy:
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

    if group_by_file:
        # Display results grouped by file
        grouped = engine.get_results_grouped_by_file(result)
        if OutputFormat(fmt) == OutputFormat.JSON:
            import orjson

            grouped_data = {
                str(file_path): [
                    {
                        "start_line": item.start_line,
                        "end_line": item.end_line,
                        "lines": item.lines,
                        "match_spans": item.match_spans,
                    }
                    for item in items
                ]
                for file_path, items in grouped.items()
            }
            sys.stdout.write(orjson.dumps(grouped_data, option=orjson.OPT_INDENT_2).decode())
            sys.stdout.write("\n")
        else:
            for file_path, items in grouped.items():
                click.echo(f"\n{'='*60}")
                click.echo(f"  {file_path} ({len(items)} matches)")
                click.echo(f"{'='*60}")
                for item in items:
                    click.echo(f"  Lines {item.start_line}-{item.end_line}:")
                    for line in item.lines:
                        click.echo(f"    {line.rstrip()}")
    elif OutputFormat(fmt) == OutputFormat.HIGHLIGHT and sys.stdout.isatty():
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
@click.option("--rate", nargs=2, type=(str, int), default=(None, None), help="评价搜索 (模式 评分1-5)")
@click.option("--add-tags", nargs=2, type=(str, str), default=(None, None), help="为搜索添加标签 (模式 '标签1,标签2')")
@click.option("--suggest", "suggest_pattern", help="基于历史记录获取搜索建议 (输入部分模式)")
@click.option("--end-session", is_flag=True, help="手动结束当前搜索会话")
@click.option("--performance-insights", is_flag=True, help="显示搜索性能洞察和建议")
@click.option("--usage-patterns", is_flag=True, help="分析搜索使用模式和趋势")
@click.option("--session-analytics", is_flag=True, help="显示基于会话的分析统计")
@click.option("--cleanup-sessions", type=int, help="清理指定天数之前的旧会话")
@click.option("--bookmark-search", help="在书签中搜索")
@click.option("--bookmark-stats", is_flag=True, help="显示书签统计信息")
@click.option("--export", "export_fmt", type=click.Choice(["json", "csv"]), help="导出历史记录 (json 或 csv)")
@click.option("--export-output", help="导出文件路径 (默认: history_export.<格式>)")
@click.option("--import-file", "import_path", help="从 JSON 文件导入历史记录")
@click.option("--import-replace", is_flag=True, help="导入时替换而非合并")
@click.option("--backup", "backup_path", help="备份历史数据到文件")
@click.option("--restore", "restore_path", help="从备份文件恢复历史数据")
@click.option("--validate-backup", "validate_path", help="验证备份文件")
@click.option("--date-from", help="按开始日期过滤 (格式: YYYY-MM-DD)")
@click.option("--date-to", help="按结束日期过滤 (格式: YYYY-MM-DD)")
@click.option("--category", help="按类别过滤 (function/class/variable/import/comment/string/regex/general)")
@click.option("--language", help="按编程语言过滤")
@click.option("--search", "search_query", help="在历史记录中全文搜索")
@click.option("--cleanup-history", type=int, help="清理指定天数之前的旧历史记录")
@click.option("--deduplicate", is_flag=True, help="去除重复的历史记录")
@click.option("--detailed-stats", is_flag=True, help="显示详细统计信息")
@click.option("--trends", is_flag=True, help="显示搜索趋势")
@click.option("--category-trends", is_flag=True, help="显示类别使用趋势")
@click.option("--failed-patterns", is_flag=True, help="显示最常失败的搜索模式")
@click.option("--session-summary", help="显示指定会话的详细摘要 (会话ID)")
@click.option("--compare-sessions", nargs=2, type=(str, str), default=(None, None), help="比较两个会话 (会话ID1 会话ID2)")
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
    rate: tuple[str | None, int | None],
    add_tags: tuple[str | None, str | None],
    suggest_pattern: str | None,
    end_session: bool,
    performance_insights: bool,
    usage_patterns: bool,
    session_analytics: bool,
    cleanup_sessions: int | None,
    bookmark_search: str | None,
    bookmark_stats: bool,
    export_fmt: str | None,
    export_output: str | None,
    import_path: str | None,
    import_replace: bool,
    backup_path: str | None,
    restore_path: str | None,
    validate_path: str | None,
    date_from: str | None,
    date_to: str | None,
    category: str | None,
    language: str | None,
    search_query: str | None,
    cleanup_history: int | None,
    deduplicate: bool,
    detailed_stats: bool,
    trends: bool,
    category_trends: bool,
    failed_patterns: bool,
    session_summary: str | None,
    compare_sessions: tuple[str | None, str | None],
) -> None:
    """显示搜索历史记录和分析。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if clear_history:
        engine.history.clear_history()
        click.echo("Search history cleared.")
        return

    if end_session:
        engine.end_current_session()
        click.echo("Current search session ended.")
        return

    if rate[0] is not None:
        rate_pattern, rating = rate
        if rating is None or not (1 <= rating <= 5):
            click.echo("Error: Rating must be between 1 and 5", err=True)
            sys.exit(1)
        if engine.rate_last_search(rate_pattern, rating):
            click.echo(f"Rated search '{rate_pattern}' with {rating} star(s).")
        else:
            click.echo(f"Search pattern '{rate_pattern}' not found in history.", err=True)
            sys.exit(1)
        return

    if add_tags[0] is not None:
        tag_pattern, tag_str = add_tags
        tag_list_input = [t.strip() for t in tag_str.split(",") if t.strip()]
        if not tag_list_input:
            click.echo("Error: At least one tag must be provided", err=True)
            sys.exit(1)
        if engine.add_search_tags(tag_pattern, tag_list_input):
            click.echo(f"Tags {tag_list_input} added to search '{tag_pattern}'.")
        else:
            click.echo(f"Search pattern '{tag_pattern}' not found in history.", err=True)
            sys.exit(1)
        return

    if suggest_pattern:
        suggestions = engine.get_pattern_suggestions(suggest_pattern, limit=limit)
        if not suggestions:
            click.echo("No suggestions found.")
            return
        click.echo(f"Search Suggestions for '{suggest_pattern}'")
        click.echo("=" * 40)
        for s in suggestions:
            click.echo(f"  {s}")
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

    if performance_insights:
        insights = engine.get_performance_insights()
        if not insights or (not insights.get("insights") and not insights.get("recommendations")):
            click.echo("No performance insights available. Build more search history first.")
            return
        click.echo("Performance Insights")
        click.echo("=" * 40)
        if insights.get("insights"):
            for insight in insights["insights"]:
                click.echo(f"  - {insight}")
        if insights.get("recommendations"):
            click.echo("\nRecommendations:")
            for rec in insights["recommendations"]:
                click.echo(f"  - {rec}")
        if insights.get("metrics"):
            click.echo("\nMetrics:")
            for key, value in insights["metrics"].items():
                click.echo(f"  {key}: {value}")
        return

    if usage_patterns:
        patterns_data = engine.get_usage_patterns()
        if not patterns_data:
            click.echo("No usage patterns available. Build more search history first.")
            return
        click.echo("Usage Patterns")
        click.echo("=" * 40)
        for key, value in patterns_data.items():
            if isinstance(value, dict):
                click.echo(f"\n  {key}:")
                for sub_key, sub_value in value.items():
                    click.echo(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                click.echo(f"\n  {key}:")
                for item in value[:10]:
                    click.echo(f"    - {item}")
            else:
                click.echo(f"  {key}: {value}")
        return

    if session_analytics:
        sa_stats = engine.get_session_analytics(days=days)
        if not sa_stats:
            click.echo("No session analytics available.")
            return
        click.echo(f"Session Analytics (Last {days} days)")
        click.echo("=" * 40)
        for key, value in sa_stats.items():
            if isinstance(value, dict):
                click.echo(f"\n  {key}:")
                for sub_key, sub_value in value.items():
                    click.echo(f"    {sub_key}: {sub_value}")
            elif isinstance(value, list):
                click.echo(f"  {key}:")
                for item in value[:10]:
                    click.echo(f"    - {item}")
            else:
                click.echo(f"  {key}: {value}")
        return

    if cleanup_sessions is not None:
        removed = engine.cleanup_old_sessions(days=cleanup_sessions)
        click.echo(f"Cleaned up {removed} sessions older than {cleanup_sessions} days.")
        return

    if bookmark_search:
        results = engine.search_bookmarks(bookmark_search)
        if not results:
            click.echo(f"No bookmarks matching '{bookmark_search}'.")
            return
        click.echo(f"Bookmarks matching '{bookmark_search}'")
        click.echo("=" * 40)
        for name, entry in results:
            click.echo(
                f"  {name}: '{entry.query_pattern}' "
                f"({entry.files_matched} files, {entry.items_count} items)"
            )
        return

    if bookmark_stats:
        bm_stats = engine.get_bookmark_stats()
        if not bm_stats:
            click.echo("No bookmark statistics available.")
            return
        click.echo("Bookmark Statistics")
        click.echo("=" * 40)
        for key, value in bm_stats.items():
            click.echo(f"  {key}: {value}")
        return

    # ── New enhanced history options ──

    if export_fmt:
        output_file = export_output or f"history_export.{export_fmt}"
        start_ts = None
        end_ts = None
        if date_from:
            from datetime import datetime as dt
            start_ts = dt.strptime(date_from, "%Y-%m-%d").timestamp()
        if date_to:
            from datetime import datetime as dt
            end_ts = dt.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59).timestamp()
        count = engine.export_history(output_file, export_fmt, start_time=start_ts, end_time=end_ts)
        click.echo(f"Exported {count} history entries to '{output_file}' ({export_fmt} format).")
        return

    if import_path:
        try:
            count = engine.import_history(import_path, merge=not import_replace)
            mode = "replaced" if import_replace else "merged"
            click.echo(f"Imported history ({mode}). Total entries: {count}")
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Import failed: {e}", err=True)
            sys.exit(1)
        return

    if backup_path:
        counts = engine.backup_history(backup_path)
        click.echo(f"Backup created: {backup_path}")
        for key, value in counts.items():
            click.echo(f"  {key}: {value}")
        return

    if restore_path:
        try:
            counts = engine.restore_history(restore_path)
            click.echo(f"Restored from: {restore_path}")
            for key, value in counts.items():
                click.echo(f"  {key}: {value}")
        except (FileNotFoundError, ValueError) as e:
            click.echo(f"Restore failed: {e}", err=True)
            sys.exit(1)
        return

    if validate_path:
        result = engine.validate_backup(validate_path)
        if result.get("valid"):
            click.echo(f"Backup is valid (created: {result.get('backup_time_iso', 'unknown')})")
            for key, value in result.items():
                if key not in ("valid", "backup_time_iso", "version"):
                    click.echo(f"  {key}: {value}")
        else:
            click.echo(f"Backup is invalid: {result.get('error', 'unknown error')}", err=True)
            sys.exit(1)
        return

    if cleanup_history is not None:
        removed = engine.cleanup_old_history(cleanup_history)
        click.echo(f"Removed {removed} history entries older than {cleanup_history} days.")
        return

    if deduplicate:
        removed = engine.deduplicate_history()
        click.echo(f"Removed {removed} duplicate history entries.")
        return

    if detailed_stats:
        stats = engine.get_detailed_history_stats()
        click.echo("Detailed History Statistics")
        click.echo("=" * 40)
        click.echo(f"  Total searches: {stats['total_searches']}")
        click.echo(f"  Unique patterns: {stats['unique_patterns']}")
        click.echo(f"  Total elapsed: {stats['total_elapsed_ms']:.1f}ms")
        click.echo(f"  Avg elapsed: {stats['average_elapsed_ms']:.1f}ms")
        click.echo(f"  Avg success score: {stats['average_success_score']:.2f}")
        click.echo(f"  Total results: {stats['total_results']}")
        click.echo(f"  Avg results: {stats['average_results']:.1f}")
        click.echo(f"  Storage: {stats['storage_bytes']} bytes")
        if stats.get("date_range"):
            dr = stats["date_range"]
            click.echo(f"  Earliest: {time.strftime('%Y-%m-%d %H:%M', time.localtime(dr['earliest']))}")
            click.echo(f"  Latest: {time.strftime('%Y-%m-%d %H:%M', time.localtime(dr['latest']))}")
        if stats.get("categories"):
            click.echo("  Categories:")
            for cat, cnt in stats["categories"].items():
                click.echo(f"    {cat}: {cnt}")
        return

    if trends:
        trend_data = engine.get_search_trends(days=days)
        if not trend_data or not trend_data.get("daily_counts"):
            click.echo("No trend data available. Build more search history first.")
            return
        click.echo(f"Search Trends (Last {days} days)")
        click.echo("=" * 40)
        click.echo(f"  Trend: {trend_data.get('trend', 'unknown')}")
        click.echo(f"  Active days: {trend_data.get('total_days_active', 0)}")
        click.echo(f"  Peak day: {trend_data.get('peak_day', 'N/A')} ({trend_data.get('peak_count', 0)} searches)")
        click.echo("\n  Daily counts:")
        for day in sorted(trend_data.get("daily_counts", {}).keys()):
            cnt = trend_data["daily_counts"][day]
            rate = trend_data.get("daily_success_rates", {}).get(day, 0)
            click.echo(f"    {day}: {cnt} searches (success: {rate:.0%})")
        return

    if category_trends:
        ct_data = engine.get_category_trends(days=days)
        if not ct_data or not ct_data.get("weekly_categories"):
            click.echo("No category trend data available.")
            return
        click.echo(f"Category Trends (Last {days} days)")
        click.echo("=" * 40)
        for week in sorted(ct_data.get("weekly_categories", {}).keys()):
            cats = ct_data["weekly_categories"][week]
            click.echo(f"  {week}: {', '.join(f'{c}={n}' for c, n in cats.items())}")
        if ct_data.get("category_shifts"):
            click.echo("\n  Significant shifts:")
            for shift in ct_data["category_shifts"]:
                click.echo(
                    f"    {shift['category']} {shift['direction']} in {shift['week']} "
                    f"({shift['from_count']} -> {shift['to_count']})"
                )
        return

    if failed_patterns:
        fp_data = engine.get_top_failed_patterns(limit=limit)
        if not fp_data:
            click.echo("No failed patterns found.")
            return
        click.echo("Top Failed Search Patterns")
        click.echo("=" * 40)
        for fp in fp_data:
            click.echo(
                f"  '{fp['pattern']}': {fp['failed_searches']}/{fp['total_searches']} failed "
                f"({fp['failure_rate']:.0%})"
            )
        return

    if session_summary:
        summary = engine.get_session_summary(session_summary)
        if "error" in summary:
            click.echo(f"Error: {summary['error']}", err=True)
            sys.exit(1)
        click.echo(f"Session Summary: {summary['session_id']}")
        click.echo("=" * 40)
        click.echo(f"  Start: {summary.get('start_time_iso', 'N/A')}")
        click.echo(f"  End: {summary.get('end_time_iso', 'ongoing')}")
        click.echo(f"  Active: {summary.get('is_active', False)}")
        click.echo(f"  Duration: {summary.get('duration_minutes', 0):.1f} min")
        click.echo(f"  Searches: {summary.get('total_searches', 0)} (success: {summary.get('successful_searches', 0)}, failed: {summary.get('failed_searches', 0)})")
        click.echo(f"  Success rate: {summary.get('success_rate', 0):.0%}")
        click.echo(f"  Unique queries: {summary.get('unique_queries', 0)}")
        if summary.get("repeated_queries"):
            click.echo(f"  Repeated queries: {', '.join(summary['repeated_queries'])}")
        if summary.get("primary_languages"):
            click.echo(f"  Languages: {', '.join(summary['primary_languages'])}")
        return

    if compare_sessions[0] is not None:
        sid1, sid2 = compare_sessions
        comparison = engine.compare_sessions(sid1, sid2)
        if "error" in comparison:
            click.echo(f"Error: {comparison['error']}", err=True)
            sys.exit(1)
        click.echo(f"Session Comparison: {sid1} vs {sid2}")
        click.echo("=" * 40)
        s1 = comparison["session_1"]
        s2 = comparison["session_2"]
        click.echo(f"  Session 1: {s1['total_searches']} searches, {s1['success_rate']:.0%} success")
        click.echo(f"  Session 2: {s2['total_searches']} searches, {s2['success_rate']:.0%} success")
        click.echo(f"  Search count diff: {comparison['search_count_diff']:+d}")
        click.echo(f"  Success rate diff: {comparison['success_rate_diff']:+.0%}")
        if comparison.get("common_queries"):
            click.echo(f"  Common queries: {', '.join(comparison['common_queries'][:5])}")
        if comparison.get("only_in_session_1"):
            click.echo(f"  Only in session 1: {', '.join(comparison['only_in_session_1'][:5])}")
        if comparison.get("only_in_session_2"):
            click.echo(f"  Only in session 2: {', '.join(comparison['only_in_session_2'][:5])}")
        return

    # ── Apply advanced filters to history listing ──

    if search_query:
        history_entries = engine.search_in_history(search_query, limit)
    elif category:
        history_entries = engine.get_history_by_category(category, limit)
    elif language:
        history_entries = engine.get_history_by_language(language, limit)
    elif date_from or date_to:
        start_ts = None
        end_ts = None
        if date_from:
            from datetime import datetime as dt
            start_ts = dt.strptime(date_from, "%Y-%m-%d").timestamp()
        if date_to:
            from datetime import datetime as dt
            end_ts = dt.strptime(date_to, "%Y-%m-%d").replace(hour=23, minute=59, second=59).timestamp()
        history_entries = engine.get_history_by_date_range(start_ts, end_ts, limit)
    elif tags:
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
@click.option("--remove-from-folder", nargs=2, type=(str, str), default=(None, None), help="从文件夹移除书签 (书签名 文件夹名)")
def bookmarks_cmd(
    add: str | None,
    remove: str | None,
    pattern: str | None,
    folder: str | None,
    create_folder: str | None,
    delete_folder: str | None,
    description: str | None,
    list_folders: bool,
    remove_from_folder: tuple[str | None, str | None],
) -> None:
    """管理搜索书签和文件夹。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if remove_from_folder[0] is not None:
        bm_name, fld_name = remove_from_folder
        if engine.remove_bookmark_from_folder(bm_name, fld_name):
            click.echo(f"Bookmark '{bm_name}' removed from folder '{fld_name}'")
        else:
            click.echo(f"Bookmark '{bm_name}' not found in folder '{fld_name}'", err=True)
            sys.exit(1)
    elif create_folder:
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
        result = engine.search_semantic(
            query=query,
            threshold=threshold,
            max_results=max_results,
            context=context,
            output=OutputFormat(fmt),
        )
    except Exception as e:
        click.echo(f"Error during semantic search: {e}", err=True)
        sys.exit(1)

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
@click.option("--circular", is_flag=True, default=False, help="检测循环依赖")
@click.option("--coupling", is_flag=True, default=False, help="显示模块耦合度指标")
@click.option("--dead-code", is_flag=True, default=False, help="检测未使用的模块（死代码）")
@click.option(
    "--export",
    "export_fmt",
    type=click.Choice(["dot", "json", "csv"]),
    help="导出依赖图 (dot/json/csv)",
)
@click.option(
    "--check-path",
    nargs=2,
    type=(str, str),
    default=(None, None),
    help="检查两个模块间是否存在依赖路径 (源模块 目标模块)",
)
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
    circular: bool,
    coupling: bool,
    dead_code: bool,
    export_fmt: str | None,
    check_path: tuple[str | None, str | None],
    fmt: str,
) -> None:
    """分析代码依赖关系。"""
    cfg = SearchConfig(paths=[search_path])
    engine = PySearch(cfg)

    if check_path[0] is not None:
        source_mod, target_mod = check_path
        try:
            has_path = engine.check_dependency_path(source_mod, target_mod)
        except Exception as e:
            click.echo(f"Error checking dependency path: {e}", err=True)
            sys.exit(1)
        if has_path:
            click.echo(f"Dependency path EXISTS: {source_mod} -> ... -> {target_mod}")
        else:
            click.echo(f"No dependency path from '{source_mod}' to '{target_mod}'.")
        return

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

    if circular:
        try:
            cycles = engine.detect_circular_dependencies()
        except Exception as e:
            click.echo(f"Error detecting circular dependencies: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(cycles, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            if not cycles:
                click.echo("No circular dependencies detected.")
                return
            click.echo(f"Circular Dependencies ({len(cycles)} cycles)")
            click.echo("=" * 40)
            for i, cycle in enumerate(cycles, 1):
                click.echo(f"  Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")
        return

    if coupling:
        try:
            coupling_metrics = engine.get_module_coupling_metrics()
        except Exception as e:
            click.echo(f"Error calculating coupling metrics: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(coupling_metrics, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            if not coupling_metrics:
                click.echo("No coupling metrics available.")
                return
            click.echo("Module Coupling Metrics")
            click.echo("=" * 40)
            # Sort by total coupling descending
            sorted_modules = sorted(
                coupling_metrics.items(),
                key=lambda x: x[1].get("total_coupling", 0),
                reverse=True,
            )
            for module, metrics_data in sorted_modules[:30]:
                ca = metrics_data.get("afferent_coupling", 0)
                ce = metrics_data.get("efferent_coupling", 0)
                instability = metrics_data.get("instability", 0)
                click.echo(f"  {module}: Ca={ca} Ce={ce} I={instability:.2f}")
        return

    if dead_code:
        try:
            dead_modules = engine.find_dead_code()
        except Exception as e:
            click.echo(f"Error detecting dead code: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(dead_modules, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            if not dead_modules:
                click.echo("No potentially unused modules detected.")
                return
            click.echo(f"Potentially Unused Modules ({len(dead_modules)})")
            click.echo("=" * 40)
            for module in dead_modules:
                click.echo(f"  - {module}")
        return

    if export_fmt:
        try:
            graph = engine.analyze_dependencies(Path(search_path), recursive)
            if graph:
                output = engine.export_dependency_graph(graph, export_fmt)
                sys.stdout.write(output)
                sys.stdout.write("\n")
            else:
                click.echo("No dependency graph available to export.", err=True)
                sys.exit(1)
        except Exception as e:
            click.echo(f"Error exporting dependency graph: {e}", err=True)
            sys.exit(1)
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
            click.echo(
                "\nUse --metrics, --impact, --suggest, --circular, "
                "--coupling, --dead-code, or --export for detailed analysis."
            )
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
@click.option("--pause", is_flag=True, default=False, help="暂停所有文件监视器")
@click.option("--resume", is_flag=True, default=False, help="恢复所有文件监视器")
@click.option("--rescan", is_flag=True, default=False, help="强制重新扫描所有监视目录")
@click.option("--performance", is_flag=True, default=False, help="显示监视性能指标")
@click.option("--include-filter", multiple=True, help="监视包含的文件模式")
@click.option("--exclude-filter", multiple=True, help="监视排除的文件模式")
def watch_cmd(
    paths: tuple[str, ...],
    enable: bool,
    disable_watch: bool,
    status: bool,
    debounce: float,
    pause: bool,
    resume: bool,
    rescan: bool,
    performance: bool,
    include_filter: tuple[str, ...],
    exclude_filter: tuple[str, ...],
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
                watcher_status = engine.get_watcher_status(name)
                status_str = watcher_status.get("status", "unknown") if watcher_status else "unknown"
                click.echo(f"  - {name} ({status_str})")
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
@click.option("--hit-rate", is_flag=True, default=False, help="显示缓存命中率")
@click.option("--set-ttl", type=float, help="设置缓存过期时间（秒）")
@click.option("--compression", is_flag=True, default=False, help="启用缓存压缩（减少磁盘/内存占用）")
@click.option("--invalidate-file", help="使指定文件相关的缓存失效")
def cache_cmd(
    enable_backend: str | None,
    disable_cache: bool,
    clear_cache: bool,
    stats: bool,
    cache_dir: str | None,
    max_size: int,
    ttl: float,
    hit_rate: bool,
    set_ttl: float | None,
    compression: bool,
    invalidate_file: str | None,
) -> None:
    """管理搜索结果缓存。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if set_ttl is not None:
        engine.set_cache_ttl(set_ttl)
        click.echo(f"Cache TTL set to {set_ttl} seconds.")
        return

    if hit_rate:
        rate = engine.get_cache_hit_rate()
        click.echo(f"Cache hit rate: {rate:.1f}%")
        return

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

    if invalidate_file:
        engine.invalidate_cache_for_file(invalidate_file)
        click.echo(f"Cache invalidated for file '{invalidate_file}'.")
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
            compression=compression,
        )
        if success:
            click.echo(
                f"Caching enabled with {enable_backend} backend"
                f"{' (compression ON)' if compression else ''}."
            )
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
    click.echo("Use --enable, --disable, --clear, --stats, --hit-rate, or --set-ttl for cache management.")


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


@cli.command("repo")
@click.option("--enable", is_flag=True, default=False, help="启用多仓库搜索")
@click.option("--disable", "disable_repo", is_flag=True, default=False, help="禁用多仓库搜索")
@click.option("--add", nargs=2, type=(str, str), default=(None, None), help="添加仓库 (名称 路径)")
@click.option(
    "--priority",
    type=click.Choice(["high", "normal", "low"]),
    default="normal",
    help="仓库优先级 (用于 --add)",
)
@click.option("--configure", nargs=2, type=(str, str), default=(None, None), help="配置仓库 (名称 key=value)")
@click.option("--remove", "remove_repo", help="移除仓库名称")
@click.option("--list", "list_repos", is_flag=True, default=False, help="列出所有仓库")
@click.option("--info", "repo_info", help="查看指定仓库详细信息")
@click.option("--search", "search_pattern", help="跨所有仓库搜索")
@click.option("--regex", "repo_regex", is_flag=True, default=False, help="跨仓库搜索时使用正则")
@click.option("--max-results", type=int, default=1000, help="跨仓库搜索最大结果数")
@click.option("--timeout", type=float, default=30.0, help="每个仓库搜索超时（秒）")
@click.option("--max-workers", type=int, default=4, help="并行搜索的最大工作线程数")
@click.option("--health", is_flag=True, default=False, help="查看所有仓库健康状态")
@click.option("--stats", "repo_stats", is_flag=True, default=False, help="查看多仓库搜索统计")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def repo_cmd(
    enable: bool,
    disable_repo: bool,
    add: tuple[str | None, str | None],
    priority: str,
    configure: tuple[str | None, str | None],
    remove_repo: str | None,
    list_repos: bool,
    repo_info: str | None,
    search_pattern: str | None,
    repo_regex: bool,
    max_results: int,
    timeout: float,
    max_workers: int,
    health: bool,
    repo_stats: bool,
    fmt: str,
) -> None:
    """管理多仓库搜索。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if enable:
        if engine.enable_multi_repo(max_workers=max_workers):
            click.echo(f"Multi-repository search enabled (max_workers={max_workers}).")
        else:
            click.echo("Failed to enable multi-repository search.", err=True)
            sys.exit(1)
        return

    if disable_repo:
        engine.disable_multi_repo()
        click.echo("Multi-repository search disabled.")
        return

    if add[0] is not None:
        repo_name, repo_path = add
        if not engine.is_multi_repo_enabled():
            click.echo("Error: Multi-repo not enabled. Use --enable first.", err=True)
            sys.exit(1)
        if engine.add_repository(repo_name, repo_path, priority=priority):
            click.echo(f"Repository '{repo_name}' added at '{repo_path}' (priority: {priority}).")
        else:
            click.echo(f"Failed to add repository '{repo_name}'.", err=True)
            sys.exit(1)
        return

    if configure[0] is not None:
        cfg_name, cfg_value = configure
        if not engine.is_multi_repo_enabled():
            click.echo("Error: Multi-repo not enabled. Use --enable first.", err=True)
            sys.exit(1)
        # Parse key=value pairs from cfg_value
        updates: dict[str, Any] = {}
        for pair in cfg_value.split(","):
            pair = pair.strip()
            if "=" not in pair:
                click.echo(f"Invalid config format '{pair}'. Use key=value.", err=True)
                sys.exit(1)
            k, v = pair.split("=", 1)
            k, v = k.strip(), v.strip()
            if v.lower() in ("true", "false"):
                updates[k] = v.lower() == "true"
            else:
                updates[k] = v
        if engine.configure_repository(cfg_name, **updates):
            click.echo(f"Repository '{cfg_name}' configured: {updates}")
        else:
            click.echo(f"Failed to configure repository '{cfg_name}'.", err=True)
            sys.exit(1)
        return

    if remove_repo:
        if engine.remove_repository(remove_repo):
            click.echo(f"Repository '{remove_repo}' removed.")
        else:
            click.echo(f"Repository '{remove_repo}' not found.", err=True)
            sys.exit(1)
        return

    if list_repos:
        repos = engine.list_repositories()
        if not repos:
            click.echo("No repositories configured. Use --enable and --add to set up.")
            return
        click.echo("Configured Repositories")
        click.echo("=" * 40)
        for name in repos:
            info = engine.get_repository_info(name)
            if info:
                click.echo(f"  {name}: {info.path} (priority: {info.priority})")
            else:
                click.echo(f"  {name}")
        return

    if repo_info:
        info = engine.get_repository_info(repo_info)
        if not info:
            click.echo(f"Repository '{repo_info}' not found.", err=True)
            sys.exit(1)
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(vars(info), option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Repository: {repo_info}")
            click.echo("=" * 40)
            for key, value in vars(info).items():
                click.echo(f"  {key}: {value}")
        return

    if health:
        health_data = engine.get_multi_repo_health()
        if not health_data:
            click.echo("Multi-repo not enabled or no health data available.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(health_data, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Repository Health Status")
            click.echo("=" * 40)
            for name, status in health_data.items():
                click.echo(f"  {name}: {status}")
        return

    if repo_stats:
        stats_data = engine.get_multi_repo_stats()
        if not stats_data:
            click.echo("Multi-repo not enabled or no statistics available.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(stats_data, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Multi-Repository Search Statistics")
            click.echo("=" * 40)
            for key, value in stats_data.items():
                click.echo(f"  {key}: {value}")
        return

    if search_pattern:
        if not engine.is_multi_repo_enabled():
            click.echo("Error: Multi-repo not enabled. Use --enable first.", err=True)
            sys.exit(1)
        result = engine.search_all_repositories(
            pattern=search_pattern,
            use_regex=repo_regex,
            max_results=max_results,
            timeout=timeout,
        )
        if not result:
            click.echo("No results or multi-repo search failed.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(vars(result), option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Multi-Repository Search Results")
            click.echo("=" * 40)
            if hasattr(result, "successful_repositories"):
                click.echo(f"  Repositories searched: {result.successful_repositories}")
            if hasattr(result, "total_results"):
                click.echo(f"  Total results: {result.total_results}")
            if hasattr(result, "results") and result.results:
                for repo_name, repo_result in result.results.items():
                    click.echo(f"\n  [{repo_name}]")
                    if hasattr(repo_result, "items"):
                        for item in repo_result.items[:20]:
                            click.echo(f"    {item.file}:{item.start_line}")
                    elif isinstance(repo_result, list):
                        for item in repo_result[:20]:
                            click.echo(f"    {item}")
        return

    # Default: show status
    click.echo(
        f"Multi-repo search: {'ENABLED' if engine.is_multi_repo_enabled() else 'DISABLED'}"
    )
    click.echo("Use --enable, --add, --search for multi-repository management.")


@cli.command("workspace")
@click.option("--init", "init_name", default=None, help="初始化新工作区 (名称)")
@click.option("--open", "open_path", default=None, help="打开工作区配置文件")
@click.option("--save", "save_path", default=None, is_flag=False, flag_value=".", help="保存工作区配置")
@click.option("--discover", "discover_root", default=None, is_flag=False, flag_value=".", help="自动发现仓库")
@click.option("--max-depth", type=int, default=3, help="自动发现最大深度")
@click.option("--add", "add_repo", nargs=2, type=(str, str), default=(None, None), help="添加仓库 (名称 路径)")
@click.option(
    "--priority",
    type=click.Choice(["high", "normal", "low"]),
    default="normal",
    help="仓库优先级 (用于 --add)",
)
@click.option("--remove", "remove_repo", default=None, help="移除仓库")
@click.option("--list", "list_repos", is_flag=True, default=False, help="列出工作区中的仓库")
@click.option("--status", "show_status", is_flag=True, default=False, help="显示工作区状态摘要")
@click.option("--search", "search_pattern", default=None, help="跨工作区搜索")
@click.option("--regex", "ws_regex", is_flag=True, default=False, help="搜索时使用正则")
@click.option("--max-results", type=int, default=1000, help="最大结果数")
@click.option("--timeout", type=float, default=30.0, help="每个仓库搜索超时（秒）")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def workspace_cmd(
    init_name: str | None,
    open_path: str | None,
    save_path: str | None,
    discover_root: str | None,
    max_depth: int,
    add_repo: tuple[str | None, str | None],
    priority: str,
    remove_repo: str | None,
    list_repos: bool,
    show_status: bool,
    search_pattern: str | None,
    ws_regex: bool,
    max_results: int,
    timeout: float,
    fmt: str,
) -> None:
    """管理工作区（多仓库工作环境）。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if init_name:
        root = discover_root or "."
        if engine.create_workspace(init_name, root, auto_discover=bool(discover_root), max_depth=max_depth):
            click.echo(f"Workspace '{init_name}' created at '{root}'.")
            if discover_root:
                repos = engine.discover_repositories(root, max_depth=max_depth)
                click.echo(f"  Discovered {len(repos)} repositories.")
            # Auto-save
            save_file = str(Path(root) / ".pysearch-workspace.toml")
            if engine.save_workspace(save_file):
                click.echo(f"  Saved to {save_file}")
        else:
            click.echo("Failed to create workspace.", err=True)
            sys.exit(1)
        return

    if open_path:
        if engine.open_workspace(open_path):
            summary = engine.get_workspace_summary()
            click.echo(f"Workspace '{summary.get('name', '?')}' opened.")
            click.echo(f"  Repositories: {summary.get('total_repositories', 0)}")
            click.echo(f"  Enabled: {summary.get('enabled_repositories', 0)}")
        else:
            click.echo(f"Failed to open workspace from '{open_path}'.", err=True)
            sys.exit(1)
        return

    if save_path:
        target = save_path if save_path != "." else None
        if engine.save_workspace(target):
            click.echo(f"Workspace saved to {target or 'default location'}.")
        else:
            click.echo("Failed to save workspace (no workspace loaded?).", err=True)
            sys.exit(1)
        return

    if discover_root:
        if not engine.is_multi_repo_enabled():
            engine.enable_multi_repo()
        repos = engine.discover_repositories(discover_root, max_depth=max_depth)
        if not repos:
            click.echo("No repositories discovered.")
            return
        click.echo(f"Discovered {len(repos)} repositories:")
        for repo in repos:
            ptype = repo.get("project_type", "")
            ptype_str = f" [{ptype}]" if ptype else ""
            click.echo(f"  {repo['name']}: {repo['path']}{ptype_str}")
        return

    if add_repo[0] is not None:
        repo_name, repo_path = add_repo
        if not engine.is_multi_repo_enabled():
            engine.enable_multi_repo()
        if engine.add_repository(repo_name, repo_path, priority=priority):
            click.echo(f"Repository '{repo_name}' added (priority: {priority}).")
        else:
            click.echo(f"Failed to add repository '{repo_name}'.", err=True)
            sys.exit(1)
        return

    if remove_repo:
        if engine.remove_repository(remove_repo):
            click.echo(f"Repository '{remove_repo}' removed.")
        else:
            click.echo(f"Repository '{remove_repo}' not found.", err=True)
            sys.exit(1)
        return

    if list_repos:
        summary = engine.get_workspace_summary()
        if not summary:
            click.echo("No workspace loaded. Use --open or --init first.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(summary, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Workspace: {summary.get('name', '?')}")
            click.echo("=" * 40)
            click.echo(f"  Total repos: {summary.get('total_repositories', 0)}")
            click.echo(f"  Enabled: {summary.get('enabled_repositories', 0)}")
            click.echo(f"  Disabled: {summary.get('disabled_repositories', 0)}")
            by_type = summary.get("repositories_by_type", {})
            if by_type:
                click.echo("  By type:")
                for ptype, count in by_type.items():
                    click.echo(f"    {ptype}: {count}")
        return

    if show_status:
        summary = engine.get_workspace_summary()
        if not summary:
            click.echo("No workspace loaded.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(summary, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Workspace: {summary.get('name', '?')}")
            click.echo(f"  Root: {summary.get('root_path', '?')}")
            click.echo(f"  Repositories: {summary.get('total_repositories', 0)}")
            click.echo(f"  Enabled: {summary.get('enabled_repositories', 0)}")
            search_settings = summary.get("search_settings", {})
            if search_settings:
                click.echo(f"  Parallel: {search_settings.get('parallel', '?')}")
                click.echo(f"  Workers: {search_settings.get('workers', '?')}")
                click.echo(f"  Max workers: {search_settings.get('max_workers', '?')}")
        return

    if search_pattern:
        if not engine.is_multi_repo_enabled():
            click.echo("Error: No workspace/multi-repo enabled. Use --open or --init first.", err=True)
            sys.exit(1)
        result = engine.search_all_repositories(
            pattern=search_pattern,
            use_regex=ws_regex,
            max_results=max_results,
            timeout=timeout,
        )
        if not result:
            click.echo("No results found.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(vars(result), option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Workspace Search Results")
            click.echo("=" * 40)
            if hasattr(result, "successful_repositories"):
                click.echo(f"  Repositories searched: {result.successful_repositories}")
            if hasattr(result, "total_matches"):
                click.echo(f"  Total matches: {result.total_matches}")
            if hasattr(result, "repository_results") and result.repository_results:
                for repo_name, repo_result in result.repository_results.items():
                    click.echo(f"\n  [{repo_name}]")
                    if hasattr(repo_result, "items"):
                        for item in repo_result.items[:20]:
                            click.echo(f"    {item.file}:{item.start_line}")
        return

    # Default: show help
    click.echo("Workspace management commands:")
    click.echo("  --init NAME     Create a new workspace")
    click.echo("  --open PATH     Open a workspace config file")
    click.echo("  --discover .    Auto-discover Git repositories")
    click.echo("  --add NAME PATH Add a repository")
    click.echo("  --list          List workspace repositories")
    click.echo("  --status        Show workspace status")
    click.echo("  --search PATTERN  Search across workspace")
    click.echo("  --save [PATH]   Save workspace config")


@cli.command("ide")
@click.option("--definition", nargs=3, type=(str, int, str), default=(None, None, None), help="跳转定义 (文件 行号 符号)")
@click.option("--references", nargs=3, type=(str, int, str), default=(None, None, None), help="查找引用 (文件 行号 符号)")
@click.option("--completion", nargs=4, type=(str, int, int, str), default=(None, None, None, None), help="自动补全 (文件 行 列 前缀)")
@click.option("--hover", nargs=4, type=(str, int, int, str), default=(None, None, None, None), help="悬停信息 (文件 行 列 符号)")
@click.option("--symbols", "symbols_file", help="列出文件中的所有符号")
@click.option("--workspace-symbols", "ws_symbols", help="搜索工作区符号")
@click.option("--diagnostics", "diagnostics_file", help="运行文件诊断")
@click.option("--path", "paths", multiple=True, default=["."], help="搜索路径")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def ide_cmd(
    definition: tuple[str | None, int | None, str | None],
    references: tuple[str | None, int | None, str | None],
    completion: tuple[str | None, int | None, int | None, str | None],
    hover: tuple[str | None, int | None, int | None, str | None],
    symbols_file: str | None,
    ws_symbols: str | None,
    diagnostics_file: str | None,
    paths: tuple[str, ...],
    fmt: str,
) -> None:
    """IDE 集成功能（跳转定义、查找引用、自动补全等）。"""
    cfg = SearchConfig(paths=list(paths) or ["."])
    engine = PySearch(cfg)

    if not engine.enable_ide_integration():
        click.echo("Failed to enable IDE integration.", err=True)
        sys.exit(1)

    if definition[0] is not None:
        file_path, line, symbol = definition
        result = engine.jump_to_definition(file_path, line, symbol)
        if not result:
            click.echo(f"No definition found for '{symbol}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(result, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Definition: {result.get('file', '?')}:{result.get('line', '?')}")
            if result.get("symbol_type"):
                click.echo(f"  Type: {result['symbol_type']}")
        return

    if references[0] is not None:
        file_path, line, symbol = references
        refs = engine.find_references(file_path, line, symbol)
        if not refs:
            click.echo(f"No references found for '{symbol}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(refs, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"References for '{symbol}' ({len(refs)} found)")
            click.echo("=" * 40)
            for ref in refs:
                marker = " [def]" if ref.get("is_definition") else ""
                click.echo(f"  {ref.get('file', '?')}:{ref.get('line', '?')}{marker}")
                if ref.get("context"):
                    click.echo(f"    {ref['context']}")
        return

    if completion[0] is not None:
        file_path, line, column, prefix = completion
        items = engine.provide_completion(file_path, line, column, prefix)
        if not items:
            click.echo(f"No completions found for prefix '{prefix}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(items, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Completions for '{prefix}' ({len(items)} found)")
            click.echo("=" * 40)
            for item in items:
                click.echo(f"  {item.get('label', '?')} ({item.get('kind', '?')})")
                if item.get("detail"):
                    click.echo(f"    {item['detail']}")
        return

    if hover[0] is not None:
        file_path, line, column, symbol = hover
        info = engine.provide_hover(file_path, line, column, symbol)
        if not info:
            click.echo(f"No hover information for '{symbol}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(info, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Hover: {info.get('symbol_name', symbol)} ({info.get('symbol_type', '?')})")
            click.echo("=" * 40)
            click.echo(info.get("contents", ""))
        return

    if symbols_file:
        symbols = engine.get_document_symbols(symbols_file)
        if not symbols:
            click.echo(f"No symbols found in '{symbols_file}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(symbols, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Symbols in {symbols_file} ({len(symbols)} found)")
            click.echo("=" * 40)
            for sym in symbols:
                click.echo(f"  L{sym.get('line', '?')}: {sym.get('kind', '?')} {sym.get('name', '?')}")
        return

    if ws_symbols:
        symbols = engine.get_workspace_symbols(ws_symbols)
        if not symbols:
            click.echo(f"No workspace symbols found for '{ws_symbols}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(symbols, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Workspace symbols matching '{ws_symbols}' ({len(symbols)} found)")
            click.echo("=" * 40)
            for sym in symbols:
                detail = f" ({sym['detail']})" if sym.get("detail") else ""
                click.echo(f"  L{sym.get('line', '?')}: {sym.get('kind', '?')} {sym.get('name', '?')}{detail}")
        return

    if diagnostics_file:
        diags = engine.get_diagnostics(diagnostics_file)
        if not diags:
            click.echo(f"No diagnostics for '{diagnostics_file}'.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(diags, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo(f"Diagnostics for {diagnostics_file} ({len(diags)} found)")
            click.echo("=" * 40)
            for diag in diags:
                sev = diag.get("severity", "info").upper()
                click.echo(f"  L{diag.get('line', '?')} [{sev}] {diag.get('message', '')}")
        return

    # Default: show usage
    click.echo("IDE Integration (enabled)")
    click.echo("Use --definition, --references, --completion, --hover, --symbols,")
    click.echo("--workspace-symbols, or --diagnostics to access IDE features.")


@cli.command("distributed")
@click.option("--enable", is_flag=True, default=False, help="启用分布式索引")
@click.option("--disable", "disable_dist", is_flag=True, default=False, help="禁用分布式索引")
@click.option("--index", "index_dirs", multiple=True, help="执行分布式索引的目录（可多次指定）")
@click.option("--workers", type=int, default=None, help="Worker 数量")
@click.option("--scale", "scale_to", type=int, default=None, help="动态调整 Worker 数量")
@click.option("--stats", "show_stats", is_flag=True, default=False, help="查看 Worker 统计")
@click.option("--metrics", is_flag=True, default=False, help="查看性能指标")
@click.option("--queue", "show_queue", is_flag=True, default=False, help="查看工作队列状态")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def distributed_cmd(
    enable: bool,
    disable_dist: bool,
    index_dirs: tuple[str, ...],
    workers: int | None,
    scale_to: int | None,
    show_stats: bool,
    metrics: bool,
    show_queue: bool,
    fmt: str,
) -> None:
    """管理分布式索引。"""
    import asyncio

    cfg = SearchConfig()
    engine = PySearch(cfg)

    if enable:
        if engine.enable_distributed_indexing(num_workers=workers):
            w = workers or "auto"
            click.echo(f"Distributed indexing enabled (workers={w}).")
        else:
            click.echo("Failed to enable distributed indexing.", err=True)
            sys.exit(1)
        return

    if disable_dist:
        engine.disable_distributed_indexing()
        click.echo("Distributed indexing disabled.")
        return

    if index_dirs:
        if not engine.is_distributed_indexing_enabled():
            # Auto-enable with default settings
            if not engine.enable_distributed_indexing(num_workers=workers):
                click.echo("Failed to enable distributed indexing.", err=True)
                sys.exit(1)

        click.echo(f"Starting distributed indexing for: {', '.join(index_dirs)}")

        async def _run_index() -> list[dict[str, Any]]:
            return await engine.distributed_index_codebase(directories=list(index_dirs))

        try:
            updates = asyncio.run(_run_index())
        except Exception as e:
            click.echo(f"Distributed indexing failed: {e}", err=True)
            sys.exit(1)

        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(updates, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            if updates:
                last = updates[-1]
                click.echo(f"Indexing complete: {last.get('description', 'done')}")
                click.echo(f"  Status: {last.get('status', 'unknown')}")
            else:
                click.echo("No progress updates received.")
        return

    if scale_to is not None:
        if not engine.is_distributed_indexing_enabled():
            click.echo("Error: Distributed indexing not enabled. Use --enable first.", err=True)
            sys.exit(1)

        async def _scale() -> bool:
            return await engine.scale_distributed_workers(scale_to)

        try:
            success = asyncio.run(_scale())
        except Exception as e:
            click.echo(f"Scaling failed: {e}", err=True)
            sys.exit(1)

        if success:
            click.echo(f"Workers scaled to {scale_to}.")
        else:
            click.echo("Failed to scale workers.", err=True)
            sys.exit(1)
        return

    if show_stats:
        if not engine.is_distributed_indexing_enabled():
            click.echo("Distributed indexing not enabled.")
            return

        async def _stats() -> list[dict[str, Any]]:
            return await engine.get_distributed_worker_stats()

        try:
            stats_data = asyncio.run(_stats())
        except Exception:
            stats_data = []

        if not stats_data:
            click.echo("No worker statistics available.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(stats_data, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Worker Statistics")
            click.echo("=" * 40)
            for s in stats_data:
                click.echo(f"  {s['worker_id']}: processed={s['items_processed']}, "
                           f"failed={s['items_failed']}, mem={s['memory_usage_mb']:.1f}MB")
        return

    if metrics:
        if not engine.is_distributed_indexing_enabled():
            click.echo("Distributed indexing not enabled.")
            return

        async def _metrics() -> dict[str, Any]:
            return await engine.get_distributed_performance_metrics()

        try:
            metrics_data = asyncio.run(_metrics())
        except Exception:
            metrics_data = {}

        if not metrics_data:
            click.echo("No performance metrics available.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(metrics_data, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Performance Metrics")
            click.echo("=" * 40)
            for section, data in metrics_data.items():
                click.echo(f"\n  [{section}]")
                if isinstance(data, dict):
                    for k, v in data.items():
                        click.echo(f"    {k}: {v}")
                else:
                    click.echo(f"    {data}")
        return

    if show_queue:
        if not engine.is_distributed_indexing_enabled():
            click.echo("Distributed indexing not enabled.")
            return
        queue_data = engine.get_distributed_queue_stats()
        if not queue_data:
            click.echo("No queue statistics available.")
            return
        if fmt == "json":
            import orjson

            sys.stdout.write(
                orjson.dumps(queue_data, option=orjson.OPT_INDENT_2, default=str).decode()
            )
            sys.stdout.write("\n")
        else:
            click.echo("Work Queue Statistics")
            click.echo("=" * 40)
            for k, v in queue_data.items():
                click.echo(f"  {k}: {v}")
        return

    # Default: show status
    enabled = engine.is_distributed_indexing_enabled()
    click.echo(f"Distributed indexing: {'ENABLED' if enabled else 'DISABLED'}")
    click.echo("Use --enable, --index, --stats, --metrics for distributed indexing management.")


@cli.command("errors")
@click.option("--summary", is_flag=True, default=False, help="显示错误摘要")
@click.option("--category", help="按类别过滤错误 (file_access, parse, search, config, index, encoding)")
@click.option("--clear", "clear_errors", is_flag=True, default=False, help="清除所有错误")
@click.option("--suppress", help="抑制指定类别的错误")
@click.option("--report", is_flag=True, default=False, help="显示完整错误报告")
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
def errors_cmd(
    summary: bool,
    category: str | None,
    clear_errors: bool,
    suppress: str | None,
    report: bool,
    fmt: str,
) -> None:
    """管理和查看错误诊断信息。"""
    cfg = SearchConfig()
    engine = PySearch(cfg)

    if clear_errors:
        engine.clear_errors()
        click.echo("All errors cleared.")
        return

    if suppress:
        engine.suppress_error_category(suppress)
        click.echo(f"Error category '{suppress}' suppressed.")
        return

    if category:
        errors = engine.get_errors_by_category(category)
        if not errors:
            click.echo(f"No errors found in category '{category}'.")
            return
        click.echo(f"Errors in category '{category}'")
        click.echo("=" * 40)
        for err in errors:
            click.echo(f"  - {err}")
        return

    if report:
        error_report = engine.get_error_report()
        if not error_report:
            click.echo("No errors to report.")
            return
        click.echo(error_report)
        return

    # Default: show summary
    error_summary = engine.get_error_summary()
    if fmt == "json":
        import orjson

        sys.stdout.write(
            orjson.dumps(error_summary, option=orjson.OPT_INDENT_2, default=str).decode()
        )
        sys.stdout.write("\n")
    else:
        click.echo("Error Summary")
        click.echo("=" * 40)
        if error_summary.get("total_errors", 0) == 0:
            click.echo("  No errors recorded.")
        else:
            for key, value in error_summary.items():
                click.echo(f"  {key}: {value}")
        click.echo(
            f"\n  Critical errors: {'YES' if engine.has_critical_errors() else 'No'}"
        )


@cli.command("suggest")
@click.option("--path", "paths", multiple=True, default=["."], help="搜索路径，可多次提供")
@click.option("--max-suggestions", type=int, default=10, help="最大建议数量")
@click.option(
    "--algorithm",
    type=click.Choice(
        ["levenshtein", "damerau_levenshtein", "jaro_winkler", "soundex", "metaphone"]
    ),
    default="damerau_levenshtein",
    help="相似度算法",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "json"]),
    default="text",
    help="输出格式",
)
@click.argument("word")
def suggest_cmd(
    paths: tuple[str, ...],
    max_suggestions: int,
    algorithm: str,
    fmt: str,
    word: str,
) -> None:
    """基于代码库标识符的拼写纠正建议。

    扫描索引文件中的所有标识符，返回与给定词最相似的匹配项。
    适用于搜索时不确定确切拼写的场景。

    Example:
        $ pysearch suggest "conection"
        $ pysearch suggest "requst" --algorithm jaro_winkler
    """
    cfg = SearchConfig(paths=list(paths) or ["."])
    engine = PySearch(cfg)

    try:
        suggestions = engine.suggest_corrections(
            word=word,
            max_suggestions=max_suggestions,
            algorithm=algorithm,
        )
    except Exception as e:
        click.echo(f"Error generating suggestions: {e}", err=True)
        sys.exit(1)

    if not suggestions:
        click.echo(f"No similar identifiers found for '{word}'.")
        return

    if fmt == "json":
        import orjson

        output_data = {
            "query": word,
            "algorithm": algorithm,
            "suggestions": [
                {"identifier": s, "similarity": round(score, 4)}
                for s, score in suggestions
            ],
        }
        sys.stdout.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2).decode())
        sys.stdout.write("\n")
    else:
        click.echo(f"Spelling Suggestions for '{word}' (algorithm: {algorithm})")
        click.echo("=" * 50)
        for suggestion, score in suggestions:
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            click.echo(f"  {suggestion:<30s} {bar} {score:.2%}")


def main() -> None:
    cli(prog_name="pysearch")


if __name__ == "__main__":
    main()
