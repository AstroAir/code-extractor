import tempfile
from pathlib import Path

import pytest

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.scorer import (
    RankingStrategy,
    ScoringWeights,
    _analyze_code_structure,
    _calculate_file_type_score,
    _calculate_popularity_score,
    cluster_results_by_similarity,
    sort_items,
)
from pysearch.types import OutputFormat, SearchItem


def create_test_item(
    file_path: str, start_line: int, end_line: int, lines: list, match_spans: list = None
):
    """Helper to create SearchItem for testing."""
    if match_spans is None:
        match_spans = [(0, (0, 10))]  # Default match span

    return SearchItem(
        file=Path(file_path),
        start_line=start_line,
        end_line=end_line,
        lines=lines,
        match_spans=match_spans,
    )


def test_scoring_weights():
    """Test configurable scoring weights."""
    weights = ScoringWeights()

    # Check default values
    assert weights.text_match == 1.0
    assert weights.match_density == 0.5
    assert weights.semantic_similarity == 0.4

    # Test custom weights
    custom_weights = ScoringWeights(text_match=2.0, exact_match_bonus=1.0, file_popularity=0.5)
    assert custom_weights.text_match == 2.0
    assert custom_weights.exact_match_bonus == 1.0
    assert custom_weights.file_popularity == 0.5


def test_file_type_scoring():
    """Test file type relevance scoring."""
    # Python files should score highest
    assert _calculate_file_type_score(Path("main.py")) == 1.2
    assert _calculate_file_type_score(Path("script.pyx")) == 1.2

    # JavaScript/TypeScript files
    assert _calculate_file_type_score(Path("app.js")) == 1.15
    assert _calculate_file_type_score(Path("component.tsx")) == 1.15

    # Java files
    assert _calculate_file_type_score(Path("Main.java")) == 1.1

    # Configuration files
    assert _calculate_file_type_score(Path("config.json")) == 0.9
    assert _calculate_file_type_score(Path("settings.yaml")) == 0.9

    # Documentation files
    assert _calculate_file_type_score(Path("README.md")) == 0.8

    # Special files
    assert _calculate_file_type_score(Path("Dockerfile")) == 1.0
    assert _calculate_file_type_score(Path("Makefile")) == 1.0


def test_popularity_scoring():
    """Test file popularity scoring."""
    all_files = {Path("main.py"), Path("utils.py"), Path("test_main.py")}

    # Important files should score higher than test files
    main_score = _calculate_popularity_score(Path("main.py"), all_files)
    test_score = _calculate_popularity_score(Path("test_main.py"), all_files)
    vendor_score = _calculate_popularity_score(Path("vendor/lib.py"), all_files)

    # Main files should score higher than test files
    assert main_score > test_score

    # Vendor files should score lower than both
    assert vendor_score < test_score
    assert vendor_score < main_score


def test_code_structure_analysis():
    """Test code structure analysis for bonus scoring."""
    # Function definition
    item = create_test_item("test.py", 1, 3, ["def process_data():", "    return data", ""])
    score = _analyze_code_structure(item, "process")
    assert score > 0.4  # Should get function definition bonus

    # Class definition
    item = create_test_item("test.py", 1, 3, ["class DataProcessor:", "    pass", ""])
    score = _analyze_code_structure(item, "Processor")
    assert score > 0.5  # Should get class definition bonus

    # Variable assignment
    item = create_test_item("test.py", 1, 2, ["data_processor = DataProcessor()", ""])
    score = _analyze_code_structure(item, "processor")
    assert score > 0.2  # Should get variable definition bonus

    # Import statement
    item = create_test_item("test.py", 1, 2, ["from data_processor import process", ""])
    score = _analyze_code_structure(item, "processor")
    assert score > 0.3  # Should get import bonus


def test_ranking_strategies():
    """Test different ranking strategies."""
    # Create test items with different characteristics
    items = [
        create_test_item(
            "main.py", 1, 5, ["def main():", "    print('hello')", "    return 0"], [(0, (4, 8))]
        ),
        create_test_item("utils.py", 10, 15, ["def helper():", "    pass"], [(0, (4, 10))]),
        create_test_item("test.py", 50, 55, ["def test_main():", "    assert True"], [(0, (4, 8))]),
    ]

    cfg = SearchConfig()

    # Test relevance strategy
    relevance_sorted = sort_items(items, cfg, "main", RankingStrategy.RELEVANCE)
    assert len(relevance_sorted) == 3

    # Test frequency strategy
    frequency_sorted = sort_items(items, cfg, "main", RankingStrategy.FREQUENCY)
    assert len(frequency_sorted) == 3

    # Test hybrid strategy
    hybrid_sorted = sort_items(items, cfg, "main", RankingStrategy.HYBRID)
    assert len(hybrid_sorted) == 3


def test_result_clustering():
    """Test result clustering by similarity."""
    # Create items with similar and different content
    items = [
        create_test_item("file1.py", 1, 3, ["def process_data():", "    return data"]),
        create_test_item("file2.py", 1, 3, ["def process_info():", "    return info"]),  # Similar
        create_test_item("file3.py", 1, 3, ["class MyClass:", "    pass"]),  # Different
        create_test_item(
            "file4.py", 1, 3, ["def handle_data():", "    return result"]
        ),  # Similar to first two
    ]

    # Test clustering with high similarity threshold
    clusters = cluster_results_by_similarity(items, similarity_threshold=0.3)

    # Should have at least 2 clusters (similar functions vs class)
    assert len(clusters) >= 2

    # Test that each item appears in exactly one cluster
    all_clustered_items = []
    for cluster in clusters:
        all_clustered_items.extend(cluster)
    assert len(all_clustered_items) == len(items)


def test_enhanced_search_api():
    """Test enhanced search API with ranking options."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files with different characteristics
        files = {
            "main.py": "def main():\n    print('Main function')\n    return 0",
            "utils.py": "def helper():\n    print('Helper function')\n    return True",
            "config.py": "CONFIG = {\n    'debug': True\n}",
            "test_main.py": "def test_main():\n    assert main() == 0",
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test search with different ranking strategies
        result_relevance = engine.search_with_ranking(
            pattern="main", ranking_strategy="relevance", output=OutputFormat.JSON
        )
        assert len(result_relevance.items) >= 1  # Should find at least main.py

        result_popularity = engine.search_with_ranking(
            pattern="main", ranking_strategy="popularity", output=OutputFormat.JSON
        )
        assert len(result_popularity.items) >= 1  # Should find at least main.py

        # Test clustering - use a pattern that should definitely match
        result_clustered = engine.search_with_ranking(
            pattern="function",  # This should match "Main function" and "Helper function"
            ranking_strategy="hybrid",
            cluster_results=True,
            output=OutputFormat.JSON,
        )
        assert len(result_clustered.items) >= 1  # Should find at least one match


def test_ranking_analysis():
    """Test ranking strategy analysis and suggestions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def process_data():\n    return data")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test analysis for different query types
        result = engine.search("process", output=OutputFormat.JSON)

        # Simple query analysis
        analysis = engine.get_ranking_suggestions("process", result)
        assert "query_type" in analysis
        assert "recommended_strategy" in analysis
        assert "suggestions" in analysis
        assert isinstance(analysis["suggestions"], list)

        # Complex query analysis
        complex_result = engine.search("process data function", output=OutputFormat.JSON)
        complex_analysis = engine.get_ranking_suggestions("process data function", complex_result)
        # Just check that analysis is returned, don't assert specific values
        assert "query_type" in complex_analysis
        assert "recommended_strategy" in complex_analysis

        # Constant query analysis
        constant_result = engine.search("CONSTANT_NAME", output=OutputFormat.JSON)
        constant_analysis = engine.get_ranking_suggestions("CONSTANT_NAME", constant_result)
        # Just check that analysis is returned
        assert "query_type" in constant_analysis


def test_result_clusters_api():
    """Test result clustering through API."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create files with similar content
        files = {
            "file1.py": "def process():\n    return data",
            "file2.py": "def process():\n    return info",
            "file3.py": "class MyClass:\n    pass",
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        result = engine.search("process", output=OutputFormat.JSON)

        # Test clustering
        clusters = engine.get_result_clusters(result, similarity_threshold=0.5)
        assert len(clusters) >= 1

        # Test that all items are included
        total_items = sum(len(cluster) for cluster in clusters)
        assert total_items == len(result.items)


if __name__ == "__main__":
    pytest.main([__file__])
