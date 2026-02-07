"""Tests for pysearch.search.scorer module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.types import SearchItem
from pysearch.search.scorer import (
    RankingStrategy,
    ScoringWeights,
    score_item,
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
