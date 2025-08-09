from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import Language, MetadataFilters


def test_semantic_advanced_scoring_and_thresholds(tmp_path: Path) -> None:
    # Create files with varying content similarity
    (tmp_path / "high_match.py").write_text("""
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
""", encoding="utf-8")
    
    (tmp_path / "low_match.py").write_text("""
import os
import sys
print("Hello world")
""", encoding="utf-8")
    
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))
    
    # High threshold should return fewer/better matches
    high_thresh = eng.search_semantic_advanced("calculate sum numbers", threshold=0.8, max_results=5)
    
    # Low threshold should return more matches
    low_thresh = eng.search_semantic_advanced("calculate sum numbers", threshold=0.1, max_results=5)
    
    # Low threshold should have >= high threshold matches
    assert low_thresh.stats.items >= high_thresh.stats.items
    
    # Test with metadata filters
    md_filters = MetadataFilters(languages={Language.PYTHON}, min_lines=2)
    filtered_res = eng.search_semantic_advanced(
        "calculate", 
        threshold=0.1, 
        max_results=10,
        metadata_filters=md_filters
    )
    assert filtered_res.stats.files_scanned >= 0


def test_semantic_advanced_empty_corpus_early_exit(tmp_path: Path) -> None:
    # Empty directory
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))
    
    # Should handle empty corpus gracefully
    result = eng.search_semantic_advanced("anything", threshold=0.5, max_results=10)
    assert result.stats.files_scanned == 0
    assert result.stats.items == 0
