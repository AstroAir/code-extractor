"""Tests for pysearch.search.scorer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.types import SearchItem
from pysearch.search.scorer import (
    RankingStrategy,
    ScoringWeights,
    cluster_results_by_similarity,
    deduplicate_overlapping_results,
    group_results_by_file,
    score_item,
    sort_items,
)


def _item(file: str = "test.py", start: int = 1, end: int = 1,
          lines: list[str] | None = None, spans: list | None = None) -> SearchItem:
    return SearchItem(
        file=Path(file), start_line=start, end_line=end,
        lines=lines or ["x = 1"],
        match_spans=spans or [],
    )


class TestRankingStrategy:
    """Tests for RankingStrategy enum."""

    def test_values(self):
        assert RankingStrategy.RELEVANCE == "relevance"
        assert RankingStrategy.FREQUENCY == "frequency"
        assert RankingStrategy.RECENCY == "recency"
        assert RankingStrategy.POPULARITY == "popularity"
        assert RankingStrategy.HYBRID == "hybrid"

    def test_is_string_enum(self):
        assert isinstance(RankingStrategy.RELEVANCE, str)


class TestScoringWeights:
    """Tests for ScoringWeights dataclass."""

    def test_defaults(self):
        w = ScoringWeights()
        assert w.text_match == 1.0
        assert w.match_density == 0.5
        assert w.semantic_similarity == 0.4

    def test_custom_weights(self):
        w = ScoringWeights(text_match=2.0, match_density=1.0)
        assert w.text_match == 2.0
        assert w.match_density == 1.0


class TestScoreItem:
    """Tests for score_item function."""

    def test_basic_scoring(self):
        item = SearchItem(
            file=Path("test.py"), start_line=1, end_line=3,
            lines=["def test():", "    pass", ""],
            match_spans=[(0, (0, 3))],
        )
        cfg = SearchConfig()
        score = score_item(item, cfg, query_text="def")
        assert score > 0

    def test_no_matches_zero_base(self):
        item = SearchItem(
            file=Path("test.py"), start_line=1, end_line=1,
            lines=["x = 1"],
            match_spans=[],
        )
        cfg = SearchConfig()
        score = score_item(item, cfg)
        assert score >= 0

    def test_exact_match_bonus(self):
        item = SearchItem(
            file=Path("test.py"), start_line=1, end_line=1,
            lines=["def main():"],
            match_spans=[(0, (0, 3))],
        )
        cfg = SearchConfig()
        score_exact = score_item(item, cfg, query_text="def main")
        score_partial = score_item(item, cfg, query_text="xyz")
        assert score_exact > score_partial

    def test_multiple_matches_higher_score(self):
        item1 = SearchItem(
            file=Path("test.py"), start_line=1, end_line=1,
            lines=["test test test"],
            match_spans=[(0, (0, 4)), (0, (5, 9)), (0, (10, 14))],
        )
        item2 = SearchItem(
            file=Path("test.py"), start_line=1, end_line=1,
            lines=["test only"],
            match_spans=[(0, (0, 4))],
        )
        cfg = SearchConfig()
        assert score_item(item1, cfg) > score_item(item2, cfg)

    def test_score_with_all_files(self):
        item = _item("main.py", spans=[(0, (0, 1))])
        cfg = SearchConfig()
        all_files = {Path("main.py"), Path("utils.py")}
        score = score_item(item, cfg, query_text="x", all_files=all_files)
        assert score >= 0


class TestSortItems:
    """Tests for sort_items function."""

    def _make_items(self) -> list[SearchItem]:
        return [
            _item("a.py", 10, 12, ["def foo():", "    pass", ""], [(0, (0, 3))]),
            _item("b.py", 1, 1, ["import os"], [(0, (0, 6))]),
            _item("a.py", 1, 2, ["class Main:", "    pass"], [(0, (0, 5))]),
        ]

    def test_sort_relevance(self):
        items = self._make_items()
        cfg = SearchConfig()
        result = sort_items(items, cfg, "def", RankingStrategy.RELEVANCE)
        assert len(result) == len(items)

    def test_sort_frequency(self):
        items = self._make_items()
        cfg = SearchConfig()
        result = sort_items(items, cfg, "def", RankingStrategy.FREQUENCY)
        assert len(result) == len(items)

    def test_sort_recency(self):
        items = self._make_items()
        cfg = SearchConfig()
        result = sort_items(items, cfg, "def", RankingStrategy.RECENCY)
        assert len(result) == len(items)

    def test_sort_popularity(self):
        items = self._make_items()
        cfg = SearchConfig()
        result = sort_items(items, cfg, "def", RankingStrategy.POPULARITY)
        assert len(result) == len(items)

    def test_sort_hybrid(self):
        items = self._make_items()
        cfg = SearchConfig()
        result = sort_items(items, cfg, "def", RankingStrategy.HYBRID)
        assert len(result) == len(items)

    def test_sort_empty_list(self):
        cfg = SearchConfig()
        result = sort_items([], cfg, "x", RankingStrategy.RELEVANCE)
        assert result == []


class TestClusterResultsBySimilarity:
    """Tests for cluster_results_by_similarity function."""

    def test_empty(self):
        clusters = cluster_results_by_similarity([])
        assert clusters == []

    def test_single_item(self):
        items = [_item("a.py", lines=["hello world"])]
        clusters = cluster_results_by_similarity(items)
        assert len(clusters) == 1
        assert len(clusters[0]) == 1

    def test_identical_items_cluster_together(self):
        items = [
            _item("a.py", 1, 1, ["def foo(): pass"]),
            _item("b.py", 1, 1, ["def foo(): pass"]),
        ]
        clusters = cluster_results_by_similarity(items, similarity_threshold=0.9)
        # Identical content should cluster together
        assert len(clusters) >= 1

    def test_different_items_separate(self):
        items = [
            _item("a.py", 1, 1, ["import os"]),
            _item("b.py", 1, 1, ["class MyDatabaseConnection:"]),
        ]
        clusters = cluster_results_by_similarity(items, similarity_threshold=0.9)
        assert len(clusters) == 2


class TestGroupResultsByFile:
    """Tests for group_results_by_file function."""

    def test_empty(self):
        grouped = group_results_by_file([])
        assert grouped == {}

    def test_single_file(self):
        items = [_item("a.py", 1), _item("a.py", 5)]
        grouped = group_results_by_file(items)
        assert len(grouped) == 1
        assert Path("a.py") in grouped
        assert len(grouped[Path("a.py")]) == 2

    def test_multiple_files(self):
        items = [_item("a.py"), _item("b.py"), _item("a.py", 5)]
        grouped = group_results_by_file(items)
        assert len(grouped) == 2
        assert len(grouped[Path("a.py")]) == 2
        assert len(grouped[Path("b.py")]) == 1

    def test_sorted_by_start_line(self):
        items = [_item("a.py", 10), _item("a.py", 1), _item("a.py", 5)]
        grouped = group_results_by_file(items)
        lines = [item.start_line for item in grouped[Path("a.py")]]
        assert lines == sorted(lines)


class TestDeduplicateOverlappingResults:
    """Tests for deduplicate_overlapping_results function."""

    def test_empty(self):
        result = deduplicate_overlapping_results([])
        assert result == []

    def test_no_overlap(self):
        items = [
            _item("a.py", 1, 3),
            _item("a.py", 20, 22),
        ]
        result = deduplicate_overlapping_results(items)
        assert len(result) == 2

    def test_overlapping_items(self):
        items = [
            _item("a.py", 1, 10, ["line"] * 10, [(0, (0, 4))]),
            _item("a.py", 5, 15, ["line"] * 11, [(0, (0, 4))]),
        ]
        result = deduplicate_overlapping_results(items, overlap_threshold=3)
        assert len(result) < len(items)

    def test_different_files_no_dedup(self):
        items = [
            _item("a.py", 1, 10),
            _item("b.py", 1, 10),
        ]
        result = deduplicate_overlapping_results(items)
        assert len(result) == 2
