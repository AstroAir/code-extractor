"""
basic_usage.py - 最小示例：演示通过 API 使用 pysearch
运行：
  python examples/basic_usage.py
"""
from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import OutputFormat, Query


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    cfg = SearchConfig(
        paths=[str(repo_root / "src")],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/.git/**", "**/__pycache__/**"],
        context=1,
        output_format=OutputFormat.JSON,
        enable_docstrings=True,
        enable_comments=True,
        enable_strings=False,
    )
    engine = PySearch(cfg)
    q = Query(pattern="def ", use_regex=True, use_ast=False, context=1, output=OutputFormat.JSON)
    res = engine.run(q)
    print(f"Scanned files: {res.stats.files_scanned}, hits: {len(res.items)}")


if __name__ == "__main__":
    main()