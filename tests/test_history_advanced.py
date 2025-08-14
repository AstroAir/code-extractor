import tempfile
import time
from pathlib import Path

import pytest

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch.core.history import SearchCategory, SearchHistory
from pysearch import OutputFormat, Query, SearchResult, SearchStats


def test_search_categorization():
    """Test automatic search categorization."""
    cfg = SearchConfig()
    history = SearchHistory(cfg)

    # Test function categorization
    query = Query(pattern="def process_data", output=OutputFormat.TEXT)
    category = history._categorize_search(query)
    assert category == SearchCategory.FUNCTION

    # Test class categorization
    query = Query(pattern="class DataProcessor", output=OutputFormat.TEXT)
    category = history._categorize_search(query)
    assert category == SearchCategory.CLASS

    # Test import categorization
    query = Query(pattern="import numpy", output=OutputFormat.TEXT)
    category = history._categorize_search(query)
    assert category == SearchCategory.IMPORT

    # Test variable categorization
    query = Query(pattern="data_processor = ", output=OutputFormat.TEXT)
    category = history._categorize_search(query)
    assert category == SearchCategory.VARIABLE

    # Test regex categorization
    query = Query(pattern="def.*process", use_regex=True, output=OutputFormat.TEXT)
    category = history._categorize_search(query)
    assert category == SearchCategory.REGEX


def test_success_score_calculation():
    """Test search success score calculation."""
    cfg = SearchConfig()
    history = SearchHistory(cfg)

    # Test no results
    result = SearchResult(
        items=[],
        stats=SearchStats(
            files_scanned=10, files_matched=0, items=0, elapsed_ms=50.0, indexed_files=100
        ),
    )
    score = history._calculate_success_score(result)
    assert score == 0.0

    # Test good results
    result = SearchResult(
        items=[],  # We don't need actual items for this test
        stats=SearchStats(
            files_scanned=10, files_matched=3, items=5, elapsed_ms=50.0, indexed_files=100
        ),
    )
    score = history._calculate_success_score(result)
    assert score > 0.5  # Should be a good score

    # Test too many results
    result = SearchResult(
        items=[],
        stats=SearchStats(
            files_scanned=10, files_matched=50, items=500, elapsed_ms=50.0, indexed_files=100
        ),
    )
    score = history._calculate_success_score(result)
    assert 0.0 < score < 0.8  # Should be lower score due to too many results


def test_session_management():
    """Test search session management."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(cache_dir=Path(tmpdir))
        history = SearchHistory(cfg)

        # Create a mock result
        result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=2, items=3, elapsed_ms=100.0, indexed_files=50
            ),
        )

        # Add first search - should create new session
        query1 = Query(pattern="test1", output=OutputFormat.TEXT)
        history.add_search(query1, result)

        session1 = history.get_current_session()
        assert session1 is not None
        assert session1.total_searches == 1
        assert session1.successful_searches == 1

        # Add second search quickly - should use same session
        query2 = Query(pattern="test2", output=OutputFormat.TEXT)
        history.add_search(query2, result)

        session2 = history.get_current_session()
        assert session2 is not None
        assert session2.session_id == session1.session_id
        assert session2.total_searches == 2

        # Simulate timeout and add another search - should create new session
        history._last_search_time = time.time() - 2000  # 2000 seconds ago
        query3 = Query(pattern="test3", output=OutputFormat.TEXT)
        history.add_search(query3, result)

        session3 = history.get_current_session()
        assert session3 is not None
        assert session3.session_id != session1.session_id
        assert session3.total_searches == 1


def test_bookmark_folders():
    """Test bookmark folder management."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(cache_dir=Path(tmpdir))
        history = SearchHistory(cfg)

        # Create folders
        assert history.create_folder("work", "Work-related searches")
        assert history.create_folder("personal", "Personal projects")
        assert not history.create_folder("work")  # Should fail - already exists

        # Check folders
        folders = history.get_folders()
        assert len(folders) == 2
        assert "work" in folders
        assert "personal" in folders
        assert folders["work"].description == "Work-related searches"

        # Add bookmark and assign to folder
        result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=2, items=3, elapsed_ms=100.0, indexed_files=50
            ),
        )
        query = Query(pattern="important_function", output=OutputFormat.TEXT)
        history.add_bookmark("important", query, result)

        # Add bookmark to folder
        assert history.add_bookmark_to_folder("important", "work")
        assert not history.add_bookmark_to_folder("nonexistent", "work")
        assert not history.add_bookmark_to_folder("important", "nonexistent")

        # Check bookmarks in folder
        work_bookmarks = history.get_bookmarks_in_folder("work")
        assert len(work_bookmarks) == 1
        assert work_bookmarks[0].query_pattern == "important_function"

        # Remove bookmark from folder
        assert history.remove_bookmark_from_folder("important", "work")
        work_bookmarks = history.get_bookmarks_in_folder("work")
        assert len(work_bookmarks) == 0

        # Delete folder
        assert history.delete_folder("personal")
        folders = history.get_folders()
        assert len(folders) == 1
        assert "personal" not in folders


def test_search_analytics():
    """Test search analytics functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(cache_dir=Path(tmpdir))
        history = SearchHistory(cfg)

        # Add several searches with different characteristics
        searches = [
            ("def process", SearchCategory.FUNCTION, 5, 100.0),
            ("class Data", SearchCategory.CLASS, 3, 150.0),
            ("import os", SearchCategory.IMPORT, 1, 50.0),
            ("variable_name", SearchCategory.VARIABLE, 0, 200.0),  # Failed search
        ]

        for pattern, category, items, elapsed in searches:
            result = SearchResult(
                items=[],
                stats=SearchStats(
                    files_scanned=10,
                    files_matched=2,
                    items=items,
                    elapsed_ms=elapsed,
                    indexed_files=100,
                ),
            )
            query = Query(pattern=pattern, output=OutputFormat.TEXT)
            history.add_search(query, result)

        # Get analytics
        analytics = history.get_search_analytics(days=1)

        assert analytics["total_searches"] == 4
        assert analytics["successful_searches"] == 3  # One failed search
        assert analytics["success_rate"] == 0.75
        assert analytics["average_search_time"] == 125.0  # (100+150+50+200)/4
        assert analytics["session_count"] == 1  # All in same session

        # Check category breakdown
        categories = dict(analytics["most_common_categories"])
        assert categories["function"] == 1
        assert categories["class"] == 1
        assert categories["import"] == 1
        assert categories["variable"] == 1


def test_pattern_suggestions():
    """Test pattern suggestion functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(cache_dir=Path(tmpdir))
        history = SearchHistory(cfg)

        # Add some search history
        patterns = [
            "process_data",
            "process_info",
            "handle_request",
            "handle_response",
            "test_process",
        ]
        result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=2, items=3, elapsed_ms=100.0, indexed_files=50
            ),
        )

        for pattern in patterns:
            query = Query(pattern=pattern, output=OutputFormat.TEXT)
            history.add_search(query, result)

        # Test suggestions
        suggestions = history.get_pattern_suggestions("process", limit=3)
        assert len(suggestions) <= 3
        assert any("process" in s for s in suggestions)

        suggestions = history.get_pattern_suggestions("handle", limit=5)
        assert len(suggestions) <= 5
        assert any("handle" in s for s in suggestions)

        # Test partial match
        suggestions = history.get_pattern_suggestions("proc", limit=3)
        assert len(suggestions) <= 3


def test_search_rating_and_tags():
    """Test search rating and tagging functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = SearchConfig(cache_dir=Path(tmpdir))
        history = SearchHistory(cfg)

        # Add a search
        result = SearchResult(
            items=[],
            stats=SearchStats(
                files_scanned=5, files_matched=2, items=3, elapsed_ms=100.0, indexed_files=50
            ),
        )
        query = Query(pattern="important_function", output=OutputFormat.TEXT)
        history.add_search(query, result)

        # Rate the search
        assert history.rate_search("important_function", 5)
        assert not history.rate_search("important_function", 6)  # Invalid rating
        assert not history.rate_search("nonexistent", 5)  # Pattern not found

        # Add tags
        tags = {"important", "work", "api"}
        assert history.add_tags_to_search("important_function", tags)
        assert not history.add_tags_to_search("nonexistent", tags)

        # Search by tags
        tagged_searches = history.search_history_by_tags({"important"})
        assert len(tagged_searches) == 1
        assert tagged_searches[0].query_pattern == "important_function"

        tagged_searches = history.search_history_by_tags({"work", "api"})
        assert len(tagged_searches) == 1

        tagged_searches = history.search_history_by_tags({"nonexistent"})
        assert len(tagged_searches) == 0


def test_api_integration():
    """Test enhanced history features through the API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def process_data():\n    return data")

        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        engine = PySearch(cfg)

        # Perform some searches
        result1 = engine.search("process", output=OutputFormat.JSON)
        result2 = engine.search("data", output=OutputFormat.JSON)

        # Test analytics
        analytics = engine.get_search_analytics(days=1)
        assert analytics["total_searches"] >= 2

        # Test pattern suggestions
        suggestions = engine.get_pattern_suggestions("proc", limit=3)
        assert isinstance(suggestions, list)

        # Test session management
        session = engine.get_current_session()
        assert session is not None
        assert session.total_searches >= 2

        sessions = engine.get_search_sessions(limit=5)
        assert len(sessions) >= 1

        # Test bookmark folders
        assert engine.create_bookmark_folder("test_folder", "Test folder")
        folders = engine.get_bookmark_folders()
        assert "test_folder" in folders

        # Test bookmarks
        engine.add_bookmark(
            "test_bookmark", Query(pattern="process", output=OutputFormat.TEXT), result1
        )
        assert engine.add_bookmark_to_folder("test_bookmark", "test_folder")

        folder_bookmarks = engine.get_bookmarks_in_folder("test_folder")
        assert len(folder_bookmarks) == 1


if __name__ == "__main__":
    pytest.main([__file__])
