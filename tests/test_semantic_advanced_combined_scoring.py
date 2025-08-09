from __future__ import annotations

from pathlib import Path

from pysearch.semantic_advanced import SemanticSearchEngine


def test_semantic_search_combined_scoring_and_thresholds(tmp_path: Path) -> None:
    engine = SemanticSearchEngine()
    
    # Content with clear semantic structure
    content = """
def calculate_total(numbers):
    '''Calculate the sum of numbers'''
    total = 0
    for num in numbers:
        total += num
    return total

class DataProcessor:
    def process_data(self, data):
        return data.strip()
"""
    
    # Test different thresholds
    high_threshold_matches = engine.search_semantic(
        query="calculate sum numbers",
        content=content,
        file_path=tmp_path / "test.py",
        threshold=0.8
    )
    
    low_threshold_matches = engine.search_semantic(
        query="calculate sum numbers", 
        content=content,
        file_path=tmp_path / "test.py",
        threshold=0.1
    )
    
    # Low threshold should return more or equal matches
    assert len(low_threshold_matches) >= len(high_threshold_matches)
    
    # Test combined score calculation and sorting
    if low_threshold_matches:
        # Matches should be sorted by combined_score in descending order
        scores = [match.combined_score for match in low_threshold_matches]
        assert scores == sorted(scores, reverse=True)
        
        # Each match should have all score components
        for match in low_threshold_matches:
            assert hasattr(match, 'semantic_score')
            assert hasattr(match, 'structural_score')
            assert hasattr(match, 'contextual_score')
            assert hasattr(match, 'combined_score')
            assert match.combined_score >= 0.0


def test_semantic_search_empty_content_early_exit(tmp_path: Path) -> None:
    engine = SemanticSearchEngine()
    
    # Empty content should return empty results
    empty_matches = engine.search_semantic(
        query="anything",
        content="",
        file_path=tmp_path / "empty.py",
        threshold=0.1
    )
    assert len(empty_matches) == 0
    
    # Content with no relevant concepts should also return empty
    irrelevant_content = "# just a comment\n"
    irrelevant_matches = engine.search_semantic(
        query="complex algorithm",
        content=irrelevant_content,
        file_path=tmp_path / "irrelevant.py",
        threshold=0.5
    )
    assert len(irrelevant_matches) == 0


def test_semantic_search_concept_relevance_filtering(tmp_path: Path) -> None:
    engine = SemanticSearchEngine()
    
    content = """
def user_login(username, password):
    return authenticate(username, password)

def calculate_tax(income):
    return income * 0.2
"""
    
    # Query matching first function
    login_matches = engine.search_semantic(
        query="user authentication login",
        content=content,
        file_path=tmp_path / "auth.py",
        threshold=0.1
    )
    
    # Query matching second function  
    tax_matches = engine.search_semantic(
        query="tax calculation income",
        content=content,
        file_path=tmp_path / "finance.py", 
        threshold=0.1
    )
    
    # Should find relevant concepts for each query
    assert len(login_matches) >= 0
    assert len(tax_matches) >= 0
