from __future__ import annotations

from pathlib import Path

from pysearch.integrations.multi_repo import RepositoryManager


def test_repository_manager_add_list_remove(tmp_path: Path) -> None:
    rm = RepositoryManager()
    p = tmp_path / "repo"
    p.mkdir()
    assert rm.add_repository("r1", p) is True
    assert "r1" in rm.list_repositories()
    assert rm.get_repository("r1") is not None
    assert rm.remove_repository("r1") is True
    assert rm.get_repository("r1") is None
