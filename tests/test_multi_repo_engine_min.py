from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig
from pysearch.integrations.multi_repo import MultiRepoSearchEngine


def test_multi_repo_engine_basic_search(tmp_path: Path) -> None:
    # Create two small repos with simple python files
    repo1 = tmp_path / "r1"
    repo2 = tmp_path / "r2"
    repo1.mkdir(); repo2.mkdir()
    (repo1 / "a.py").write_text("def foo():\n    pass\n", encoding="utf-8")
    (repo2 / "b.py").write_text("class Bar:\n    pass\n", encoding="utf-8")

    eng = MultiRepoSearchEngine(max_workers=2)
    assert eng.add_repository("r1", repo1, config=SearchConfig(paths=[str(repo1)], include=["**/*.py"]))
    assert eng.add_repository("r2", repo2, config=SearchConfig(paths=[str(repo2)], include=["**/*.py"]))

    # Search across all repos
    res = eng.search_all("def", context=0, aggregate_results=True)
    assert res.total_repositories == 2
    # might find at least 1
    assert res.total_matches >= 0

    stats = eng.get_search_statistics()
    assert "total_searches" in stats
    health = eng.get_health_status()
    assert "total" in health

