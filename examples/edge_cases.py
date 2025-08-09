"""
Edge case examples demonstrating query toggles for comments/strings/docstrings and empty results.
Run: python examples/edge_cases.py
"""
from __future__ import annotations

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Query


def toggles_example() -> None:
    cfg = SearchConfig(paths=["./src"], include=["**/*.py"], context=0)
    eng = PySearch(cfg)

    _text = '"use as string" # as comment'  # Example text for testing

    for d, c, s in [(True, True, True), (False, True, True), (True, False, True), (True, True, False)]:
        q = Query(pattern="as", use_regex=False, context=0,
                  search_docstrings=d, search_comments=c, search_strings=s)
        r = eng.run(q)
        print("docstrings/comments/strings:", d, c, s, "-> items:", r.stats.items)


if __name__ == "__main__":
    toggles_example()

