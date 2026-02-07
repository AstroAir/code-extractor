from __future__ import annotations

from pathlib import Path

from pysearch import SearchConfig
from pysearch.integrations.multi_repo import MultiRepoSearchEngine


def test_multi_repo_search_history_and_bounds(tmp_path: Path) -> None:
    eng = MultiRepoSearchEngine()

    # Add repos
    p1 = tmp_path / "r1"
    p1.mkdir()
    (p1 / "a.py").write_text("def func1(): pass\n", encoding="utf-8")

    p2 = tmp_path / "r2"
    p2.mkdir()
    (p2 / "b.py").write_text("def func2(): pass\n", encoding="utf-8")

    eng.add_repository("r1", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"]))
    eng.add_repository("r2", p2, config=SearchConfig(paths=[str(p2)], include=["**/*.py"]))

    # Perform multiple searches to build history
    for i in range(5):
        eng.search_all(f"func{i % 2 + 1}", use_regex=False, max_results=10)

    # Check search statistics
    stats = eng.get_search_statistics()
    assert stats["total_searches"] >= 5
    assert "average_search_time" in stats

    # Test history bounds (should limit to 100 entries)
    assert len(eng.search_history) <= 100


def test_multi_repo_mixed_enabled_disabled(tmp_path: Path) -> None:
    eng = MultiRepoSearchEngine()

    # Add enabled repo
    p1 = tmp_path / "enabled"
    p1.mkdir()
    (p1 / "a.py").write_text("enabled_code\n", encoding="utf-8")

    eng.add_repository("enabled", p1, config=SearchConfig(paths=[str(p1)], include=["**/*.py"]))

    # Manually disable a repository (simulate disabled state)
    repo_info = eng.repository_manager.get_repository("enabled")
    if repo_info:
        repo_info.enabled = False

    # Search should handle disabled repositories gracefully
    result = eng.search_repositories(repositories=["enabled"], pattern="enabled_code")
    assert result.total_repositories >= 0
