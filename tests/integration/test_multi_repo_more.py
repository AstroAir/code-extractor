from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig
from pysearch.integrations.multi_repo import MultiRepoSearchResult, RepositoryManager
from pysearch import SearchResult, SearchStats


def test_multi_repo_aggregation_and_health(tmp_path: Path) -> None:
    r1 = SearchResult(stats=SearchStats(files_scanned=1, files_matched=1, items=1, elapsed_ms=1.0))
    r2 = SearchResult(stats=SearchStats(files_scanned=2, files_matched=0, items=0, elapsed_ms=2.0))
    agg = MultiRepoSearchResult(repository_results={"a": r1, "b": r2})
    # aggregate-like checks using properties
    total = agg.total_matches
    assert total >= 1
    assert agg.success_rate >= 0.0

    rm = RepositoryManager()
    # Existing repo path for add_repository success
    repo_path = tmp_path / "present"
    repo_path.mkdir()
    assert rm.add_repository("r1", repo_path, config=SearchConfig(paths=[str(repo_path)]))

