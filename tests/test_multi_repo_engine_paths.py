from __future__ import annotations

from pathlib import Path

from pysearch.config import SearchConfig
from pysearch.multi_repo import MultiRepoSearchEngine


def test_multi_repo_no_repos_and_aggregate(tmp_path: Path) -> None:
    eng = MultiRepoSearchEngine()
    # no repos: expect empty result
    r = eng.search_all("noop", use_regex=False, aggregate_results=True, max_results=5)
    assert r.total_repositories == 0

    # add two simple repos with basic structure
    p1 = tmp_path / "r1"; p1.mkdir(); (p1 / "a.py").write_text("x=1\n", encoding="utf-8")
    p2 = tmp_path / "r2"; p2.mkdir(); (p2 / "b.py").write_text("def f():\n pass\n", encoding="utf-8")

    assert eng.add_repository("r1", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"]))
    assert eng.add_repository("r2", p2, config=SearchConfig(paths=[str(p2)], include=["**/*.py"]))

    # search with aggregation
    res = eng.search_all("def ", use_regex=True, aggregate_results=True, max_results=10)
    assert res.total_repositories >= 2

    stats = eng.get_search_statistics()
    assert "total_searches" in stats

    # health status combines manager summary and stats
    hs = eng.get_health_status()
    assert "total" in hs and "average_search_time" in hs

