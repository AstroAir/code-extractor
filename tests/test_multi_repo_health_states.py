from __future__ import annotations

from pathlib import Path

from pysearch.config import SearchConfig
from pysearch.multi_repo import RepositoryInfo, RepositoryManager


def test_repository_info_health_states(tmp_path: Path) -> None:
    # Test missing path -> error status
    missing_path = tmp_path / "nonexistent"
    repo_missing = RepositoryInfo(
        name="missing",
        path=missing_path,
        config=SearchConfig(paths=[str(missing_path)])
    )
    assert repo_missing.health_status == "error"
    
    # Test existing path but no git -> warning status
    no_git_path = tmp_path / "no_git"
    no_git_path.mkdir()
    repo_no_git = RepositoryInfo(
        name="no_git",
        path=no_git_path,
        config=SearchConfig(paths=[str(no_git_path)])
    )
    assert repo_no_git.health_status == "warning"
    
    # Test refresh_status method
    repo_no_git.refresh_status()
    assert repo_no_git.health_status in ["warning", "error"]


def test_repository_manager_health_summary(tmp_path: Path) -> None:
    rm = RepositoryManager()
    
    # Add repos with different health states
    healthy_path = tmp_path / "healthy"
    healthy_path.mkdir()
    (healthy_path / ".git").mkdir()  # Fake git dir
    
    warning_path = tmp_path / "warning"
    warning_path.mkdir()  # No .git
    
    rm.add_repository("healthy", healthy_path, config=SearchConfig(paths=[str(healthy_path)]))
    rm.add_repository("warning", warning_path, config=SearchConfig(paths=[str(warning_path)]))
    
    # Get health summary
    summary = rm.get_health_summary()
    assert summary["total"] >= 2
    assert "healthy" in summary
    assert "warning" in summary
    assert "error" in summary
    assert "enabled" in summary
    assert "disabled" in summary
    
    # Test refresh_all_status
    rm.refresh_all_status()
    
    # Summary should still be valid after refresh
    summary2 = rm.get_health_summary()
    assert summary2["total"] >= 2


def test_repository_manager_error_logging(tmp_path: Path) -> None:
    rm = RepositoryManager()
    
    # Test adding duplicate repository (should log warning and return False)
    valid_path = tmp_path / "valid"
    valid_path.mkdir()
    
    assert rm.add_repository("test", valid_path, config=SearchConfig(paths=[str(valid_path)])) is True
    assert rm.add_repository("test", valid_path, config=SearchConfig(paths=[str(valid_path)])) is False
    
    # Test adding repository with non-existent path (should log error and return False)
    invalid_path = tmp_path / "invalid"
    assert rm.add_repository("invalid", invalid_path, config=SearchConfig(paths=[str(invalid_path)])) is False
