from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from .api import PySearch
from .config import SearchConfig
from .formatter import format_result, render_highlight_console
from .types import ASTFilters, OutputFormat, Query


@click.group()
def cli() -> None:
    """pysearch - 面向 Python 代码库的上下文感知搜索引擎"""
    pass


@cli.command("find")
@click.option("--path", "paths", multiple=True, default=["."], help="搜索路径，可多次提供")
@click.option("--include", multiple=True, default=["**/*.py"], help="包含的 glob 模式")
@click.option("--exclude", multiple=True, default=["**/.venv/**", "**/.git/**", "**/build/**", "**/dist/**", "**/__pycache__/**"], help="排除的 glob 模式")
@click.option("--pattern", required=True, help="文本/正则模式")
@click.option("--regex", is_flag=True, default=False, help="启用正则匹配")
@click.option("--context", type=int, default=2, help="上下文行数")
@click.option("--format", "fmt", type=click.Choice([e.value for e in OutputFormat]), default=OutputFormat.TEXT.value, help="输出格式")
@click.option("--filter-func-name", "filter_func", type=str, default=None, help="AST 过滤: 函数名正则")
@click.option("--filter-class-name", "filter_class", type=str, default=None, help="AST 过滤: 类名正则")
@click.option("--filter-decorator", "filter_deco", type=str, default=None, help="AST 过滤: 装饰器正则")
@click.option("--filter-import", "filter_import", type=str, default=None, help="AST 过滤: 导入名正则（包含模块前缀）")
@click.option("--no-docstrings", is_flag=True, default=False, help="禁用 docstring 搜索")
@click.option("--no-comments", is_flag=True, default=False, help="禁用注释搜索")
@click.option("--no-strings", is_flag=True, default=False, help="禁用字符串字面量搜索")
@click.option("--stats", is_flag=True, default=False, help="打印性能统计")
def find_cmd(
    paths: tuple[str, ...],
    include: tuple[str, ...],
    exclude: tuple[str, ...],
    pattern: str,
    regex: bool,
    context: int,
    fmt: str,
    filter_func: Optional[str],
    filter_class: Optional[str],
    filter_deco: Optional[str],
    filter_import: Optional[str],
    no_docstrings: bool,
    no_comments: bool,
    no_strings: bool,
    stats: bool,
) -> None:
    cfg = SearchConfig(
        paths=list(paths) or ["."],
        include=list(include) or ["**/*.py"],
        exclude=list(exclude),
        context=context,
        output_format=OutputFormat(fmt),
        enable_docstrings=not no_docstrings,
        enable_comments=not no_comments,
        enable_strings=not no_strings,
    )
    engine = PySearch(cfg)

    filters = None
    if any([filter_func, filter_class, filter_deco, filter_import]):
        filters = ASTFilters(
            func_name=filter_func,
            class_name=filter_class,
            decorator=filter_deco,
            imported=filter_import,
        )

    query = Query(
        pattern=pattern,
        use_regex=regex,
        use_ast=filters is not None,
        context=context,
        output=OutputFormat(fmt),
        filters=filters,
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


def main() -> None:
    cli(prog_name="pysearch")


if __name__ == "__main__":
    main()