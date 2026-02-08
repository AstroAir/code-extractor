"""Performance benchmark tests for pysearch core modules.

Benchmarks cover the critical hot paths:
- File iteration with directory pruning
- Regex matching with compiled-pattern caching
- Indexer scan (strict hash vs lazy mtime)
- In-file search (text & regex via search_in_file)
- AST block extraction
- Result scoring and sorting
- Safe file reading
- End-to-end PySearch.run()
"""

from __future__ import annotations

import random
import string
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

import pytest

try:
    import pytest_benchmark  # noqa: F401

    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False

pytestmark = [
    pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed"),
    pytest.mark.benchmark,
]

from pysearch import PySearch, SearchConfig
from pysearch.core.types import ASTFilters, Query, SearchItem
from pysearch.indexing.indexer import Indexer
from pysearch.search.matchers import find_ast_blocks, find_text_regex_matches, search_in_file
from pysearch.search.scorer import score_item, sort_items
from pysearch.utils.helpers import iter_files, read_text_safely

T = TypeVar("T")


class _Benchmark(Protocol):
    def __call__(self, func: Callable[[], T]) -> T: ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _rand_text(lines: int = 50, width: int = 80) -> str:
    alphabet = string.ascii_letters + string.digits + " _-()[]{}.,;:/'\""
    return "\n".join(
        "".join(random.choice(alphabet) for _ in range(width)) for _ in range(lines)
    )


_PYTHON_TEMPLATE = """\
import os
import sys
from pathlib import Path

def function_{idx}(x: int, y: str = "default") -> str:
    '''Docstring for function_{idx}.'''
    result = f"{{x}}: {{y}}"
    if x > 0:
        return result
    return ""

class Class_{idx}:
    '''Class docstring for Class_{idx}.'''

    def method_a(self) -> None:
        pass

    def method_b(self, value: int) -> int:
        return value * 2

async def async_handler_{idx}(request):
    '''Async handler.'''
    data = await request.json()
    return data
"""


def _populate_project(root: Path, n_files: int = 100) -> None:
    for i in range(n_files):
        _write(root / "src" / f"mod{i}.py", _PYTHON_TEMPLATE.format(idx=i))


# ---------------------------------------------------------------------------
# 1. File iteration – directory pruning
# ---------------------------------------------------------------------------


def test_benchmark_iter_files_prune(benchmark: _Benchmark, tmp_path: Path) -> None:
    for i in range(50):
        _write(tmp_path / "src" / f"m{i}" / f"f{i}.py", _rand_text(10, 60))
        _write(tmp_path / "src" / f"m{i}" / f"{i}.txt", "x")
    for i in range(30):
        _write(tmp_path / ".venv" / "lib" / f"v{i}.py", _rand_text(10, 60))
    for i in range(30):
        _write(tmp_path / ".git" / "objects" / f"o{i}", "x")

    include = ["**/*.py"]
    exclude = ["**/.venv/**", "**/.git/**"]

    def run() -> list[Path]:
        return list(
            iter_files(
                roots=[str(tmp_path)],
                include=include,
                exclude=exclude,
                follow_symlinks=False,
                prune_excluded_dirs=True,
            )
        )

    # Warm-up
    run()

    result = benchmark(run)
    assert all(".venv" not in str(p) and ".git" not in str(p) for p in result)
    assert len(result) == 50


# ---------------------------------------------------------------------------
# 2. Regex matching with compiled-pattern cache
# ---------------------------------------------------------------------------


def test_benchmark_regex_cache(benchmark: _Benchmark, tmp_path: Path) -> None:
    n_files = 100
    pattern = r"(foo|bar)[0-9]{2,4}"
    for i in range(n_files):
        content = "\n".join(f"line {j} foo{j % 100}" for j in range(200))
        _write(tmp_path / "pkg" / f"f{i}.py", content)

    texts = [
        (tmp_path / "pkg" / f"f{i}.py").read_text(encoding="utf-8") for i in range(n_files)
    ]

    def run() -> int:
        total = 0
        for t in texts:
            ms = find_text_regex_matches(t, pattern=pattern, use_regex=True)
            total += len(ms)
        return total

    tot = benchmark(run)
    assert tot > 0


# ---------------------------------------------------------------------------
# 3. Indexer scan – strict hash vs lazy mtime
# ---------------------------------------------------------------------------


def test_benchmark_indexer_strict_vs_lazy(benchmark: _Benchmark, tmp_path: Path) -> None:
    for i in range(100):
        _write(tmp_path / "src" / f"mod{i}.py", _rand_text(20, 60))

    cfg_lazy = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[])
    cfg_lazy.strict_hash_check = False
    idx_lazy = Indexer(cfg_lazy)

    cfg_strict = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[])
    cfg_strict.strict_hash_check = True
    idx_strict = Indexer(cfg_strict)

    def run_lazy() -> tuple[int, int]:
        changed, _removed, total = idx_lazy.scan()
        idx_lazy.save()
        return len(changed), total

    def run_strict() -> tuple[int, int]:
        changed, _removed, total = idx_strict.scan()
        idx_strict.save()
        return len(changed), total

    # Warm-up: initial index build
    run_lazy()
    run_strict()

    # Benchmark re-scan (no modifications)
    c_lazy, total_lazy = benchmark(run_lazy)

    c_strict, total_strict = run_strict()
    assert total_lazy == total_strict
    assert c_strict == 0


# ---------------------------------------------------------------------------
# 4. In-file search (text & regex via search_in_file)
# ---------------------------------------------------------------------------


def test_benchmark_search_in_file(benchmark: _Benchmark, tmp_path: Path) -> None:
    content = _PYTHON_TEMPLATE.format(idx=0) * 20  # ~600 lines
    fpath = tmp_path / "big_module.py"
    _write(fpath, content)

    query_text = Query(pattern="def", use_regex=False, context=2)
    query_regex = Query(pattern=r"def \w+\(", use_regex=True, context=2)

    # Warm-up
    search_in_file(fpath, content, query_text)
    search_in_file(fpath, content, query_regex)

    def run() -> int:
        items_text = search_in_file(fpath, content, query_text)
        items_regex = search_in_file(fpath, content, query_regex)
        return len(items_text) + len(items_regex)

    total = benchmark(run)
    assert total > 0


# ---------------------------------------------------------------------------
# 5. AST block extraction
# ---------------------------------------------------------------------------


def test_benchmark_ast_search(benchmark: _Benchmark) -> None:
    source = _PYTHON_TEMPLATE.format(idx=0) * 10  # ~300 lines

    filters_func = ASTFilters(func_name=r"function_\d+")
    filters_class = ASTFilters(class_name=r"Class_\d+")

    # Warm-up
    find_ast_blocks(source, filters_func)
    find_ast_blocks(source, filters_class)

    def run() -> int:
        blocks_f = find_ast_blocks(source, filters_func)
        blocks_c = find_ast_blocks(source, filters_class)
        return len(blocks_f) + len(blocks_c)

    total = benchmark(run)
    assert total > 0


# ---------------------------------------------------------------------------
# 6. Result scoring and sorting
# ---------------------------------------------------------------------------


def test_benchmark_score_and_sort(benchmark: _Benchmark, tmp_path: Path) -> None:
    _populate_project(tmp_path, n_files=50)
    cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[])

    # Build a set of SearchItems by searching across files
    items: list[SearchItem] = []
    query = Query(pattern="def", use_regex=False, context=1)
    for i in range(50):
        fpath = tmp_path / "src" / f"mod{i}.py"
        text = fpath.read_text(encoding="utf-8")
        items.extend(search_in_file(fpath, text, query))

    assert len(items) > 0, "Need items to benchmark scoring"

    def run() -> list[SearchItem]:
        # Score every item individually then sort
        for item in items:
            score_item(item, cfg, query_text="def")
        return sort_items(items, cfg, query_text="def")

    sorted_items = benchmark(run)
    assert len(sorted_items) == len(items)


# ---------------------------------------------------------------------------
# 7. Safe file reading
# ---------------------------------------------------------------------------


def test_benchmark_read_text_safely(benchmark: _Benchmark, tmp_path: Path) -> None:
    for i in range(100):
        _write(tmp_path / f"file_{i}.py", _rand_text(100, 80))

    paths = [tmp_path / f"file_{i}.py" for i in range(100)]

    def run() -> int:
        count = 0
        for p in paths:
            text = read_text_safely(p)
            if text is not None:
                count += 1
        return count

    count = benchmark(run)
    assert count == 100


# ---------------------------------------------------------------------------
# 8. End-to-end PySearch.run()
# ---------------------------------------------------------------------------


def test_benchmark_pysearch_end_to_end(benchmark: _Benchmark, tmp_path: Path) -> None:
    _populate_project(tmp_path, n_files=80)
    cfg = SearchConfig(
        paths=[str(tmp_path)],
        include=["**/*.py"],
        exclude=[],
        parallel=False,
    )
    engine = PySearch(cfg)

    query = Query(pattern=r"class \w+", use_regex=True, context=2)

    # Warm-up (builds index)
    engine.run(query)

    def run() -> int:
        result = engine.run(query)
        return len(result.items)

    total = benchmark(run)
    assert total > 0
