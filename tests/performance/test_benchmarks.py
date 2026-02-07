from __future__ import annotations

import random
import string
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeVar

from pysearch import SearchConfig

T = TypeVar("T")


class _Benchmark(Protocol):
    def __call__(self, func: Callable[[], T]) -> T: ...


from pysearch.indexing.indexer import Indexer
from pysearch.search.matchers import find_text_regex_matches
from pysearch.utils.helpers import iter_files


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _rand_text(lines: int = 50, width: int = 80) -> str:
    alphabet = string.ascii_letters + string.digits + " _-()[]{}.,;:/'\""
    result: list[str] = []
    for _ in range(lines):
        result.append("".join(random.choice(alphabet) for _ in range(width)))
    return "\n".join(result)


def test_benchmark_iter_files_prune(benchmark: _Benchmark, tmp_path: Path) -> None:
    # layout with excluded trees
    for i in range(50):
        _write(tmp_path / "src" / f"m{i}" / f"f{i}.py", _rand_text(10, 60))
        _write(tmp_path / "src" / f"m{i}" / f"{i}.txt", "x")
    for i in range(30):
        _write(tmp_path / ".venv" / "lib" / f"v{i}.py", _rand_text(10, 60))
    for i in range(30):
        _write(tmp_path / ".git" / "objects" / f"o{i}", "x")

    include = ["**/*.py"]
    exclude = ["**/.venv/**", "**/.git/**"]

    def run_prune_on() -> list[Path]:
        return list(
            iter_files(
                roots=[str(tmp_path)],
                include=include,
                exclude=exclude,
                follow_symlinks=False,
                prune_excluded_dirs=True,
            )
        )

    def run_prune_off() -> list[Path]:
        return list(
            iter_files(
                roots=[str(tmp_path)],
                include=include,
                exclude=exclude,
                follow_symlinks=False,
                prune_excluded_dirs=False,
            )
        )

    # Warm-up
    list(run_prune_on())
    list(run_prune_off())

    # Benchmark pruning ON
    result_on = benchmark(run_prune_on)
    assert all(".venv" not in str(p) and ".git" not in str(p) for p in result_on)

    # Optional side check for pruning OFF to ensure correctness (not benchmarked)
    res_off = run_prune_off()
    assert all(".venv" not in str(p) and ".git" not in str(p) for p in res_off)


def test_benchmark_regex_cache(benchmark: _Benchmark, tmp_path: Path) -> None:
    # Create N files with repeated pattern occurrences
    N = 100
    pattern = r"(foo|bar)[0-9]{2,4}"
    for i in range(N):
        content = "\n".join([f"line {j} foo{j%100}" for j in range(200)])
        _write(tmp_path / "pkg" / f"f{i}.py", content)

    texts = [(tmp_path / "pkg" / f"f{i}.py").read_text(encoding="utf-8") for i in range(N)]

    # First call warms cache; benchmark focuses on repeated use across many texts
    def run() -> int:
        total = 0
        for t in texts:
            ms = find_text_regex_matches(t, pattern=pattern, use_regex=True)
            total += len(ms)
        return total

    tot = benchmark(run)
    assert tot > 0


def test_benchmark_indexer_strict_vs_lazy(benchmark: _Benchmark, tmp_path: Path) -> None:
    # Prepare repo
    for i in range(100):
        _write(tmp_path / "src" / f"mod{i}.py", _rand_text(20, 60))

    cfg_lazy = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[])
    cfg_lazy.strict_hash_check = False
    idx_lazy = Indexer(cfg_lazy)

    cfg_strict = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[])
    cfg_strict.strict_hash_check = True
    idx_strict = Indexer(cfg_strict)

    def run_lazy() -> tuple[int, int]:
        changed, removed, total = idx_lazy.scan()
        idx_lazy.save()
        return len(changed), total

    def run_strict() -> tuple[int, int]:
        changed, removed, total = idx_strict.scan()
        idx_strict.save()
        return len(changed), total

    # Warm-up initial index build
    run_lazy()
    run_strict()

    # Re-scan without modifications - just test one mode with benchmark
    c_lazy, total_lazy = benchmark(run_lazy)

    # Test strict mode without benchmark
    c_strict, total_strict = run_strict()

    assert total_lazy == total_strict
    # On a second pass with no file changes expected, strict mode should usually report 0 changes
    assert c_strict == 0
