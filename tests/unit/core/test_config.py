"""Tests for pysearch.core.config module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.config import RankStrategy, SearchConfig
from pysearch.core.types import Language, OutputFormat


class TestRankStrategy:
    """Tests for RankStrategy enum."""

    def test_default_value(self):
        assert RankStrategy.DEFAULT == "default"

    def test_is_string_enum(self):
        assert isinstance(RankStrategy.DEFAULT, str)


class TestSearchConfigDefaults:
    """Tests for SearchConfig default values."""

    def test_default_paths(self):
        cfg = SearchConfig()
        assert cfg.paths == ["."]

    def test_default_include_exclude(self):
        cfg = SearchConfig()
        assert cfg.include is None
        assert cfg.exclude is None

    def test_default_languages(self):
        cfg = SearchConfig()
        assert cfg.languages is None

    def test_default_file_size_limit(self):
        cfg = SearchConfig()
        assert cfg.file_size_limit == 2_000_000

    def test_default_context(self):
        cfg = SearchConfig()
        assert cfg.context == 2

    def test_default_output_format(self):
        cfg = SearchConfig()
        assert cfg.output_format == OutputFormat.TEXT

    def test_default_follow_symlinks(self):
        cfg = SearchConfig()
        assert cfg.follow_symlinks is False

    def test_default_content_toggles(self):
        cfg = SearchConfig()
        assert cfg.enable_docstrings is True
        assert cfg.enable_comments is True
        assert cfg.enable_strings is True

    def test_default_performance(self):
        cfg = SearchConfig()
        assert cfg.parallel is True
        assert cfg.workers == 0
        assert cfg.cache_dir is None

    def test_default_hash_and_prune(self):
        cfg = SearchConfig()
        assert cfg.strict_hash_check is False
        assert cfg.dir_prune_exclude is True

    def test_default_ranking(self):
        cfg = SearchConfig()
        assert cfg.rank_strategy == RankStrategy.DEFAULT
        assert cfg.ast_weight == 2.0
        assert cfg.text_weight == 1.0

    def test_default_graphrag(self):
        cfg = SearchConfig()
        assert cfg.enable_graphrag is False
        assert cfg.graphrag_max_hops == 2
        assert cfg.graphrag_min_confidence == 0.5
        assert cfg.graphrag_semantic_threshold == 0.7
        assert cfg.graphrag_context_window == 5

    def test_default_metadata_indexing(self):
        cfg = SearchConfig()
        assert cfg.enable_metadata_indexing is False

    def test_default_qdrant(self):
        cfg = SearchConfig()
        assert cfg.qdrant_enabled is False
        assert cfg.qdrant_host == "localhost"
        assert cfg.qdrant_port == 6333
        assert cfg.qdrant_collection_name == "pysearch_vectors"

    def test_custom_values(self):
        cfg = SearchConfig(
            paths=["src"],
            context=5,
            parallel=False,
            workers=8,
        )
        assert cfg.paths == ["src"]
        assert cfg.context == 5
        assert cfg.parallel is False
        assert cfg.workers == 8


class TestSearchConfigMethods:
    """Tests for SearchConfig methods."""

    def test_resolve_cache_dir_default(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        cache_dir = cfg.resolve_cache_dir()
        assert cache_dir == tmp_path / ".pysearch-cache"

    def test_resolve_cache_dir_custom(self, tmp_path: Path):
        custom_cache = tmp_path / "my-cache"
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=custom_cache)
        assert cfg.resolve_cache_dir() == custom_cache

    def test_get_include_patterns_custom(self):
        cfg = SearchConfig(include=["**/*.py", "**/*.js"])
        assert cfg.get_include_patterns() == ["**/*.py", "**/*.js"]

    def test_get_include_patterns_default_all_languages(self):
        cfg = SearchConfig()
        patterns = cfg.get_include_patterns()
        assert "**/*.py" in patterns
        assert "**/*.js" in patterns
        assert "**/*.ts" in patterns
        assert len(patterns) > 10

    def test_get_exclude_patterns_custom(self):
        cfg = SearchConfig(exclude=["**/vendor/**"])
        assert cfg.get_exclude_patterns() == ["**/vendor/**"]

    def test_get_exclude_patterns_default(self):
        cfg = SearchConfig()
        patterns = cfg.get_exclude_patterns()
        assert "**/.venv/**" in patterns
        assert "**/.git/**" in patterns
        assert "**/__pycache__/**" in patterns

    def test_get_qdrant_config_disabled(self):
        cfg = SearchConfig(qdrant_enabled=False)
        assert cfg.get_qdrant_config() is None

    def test_get_graphrag_query_defaults(self):
        cfg = SearchConfig(graphrag_max_hops=3, graphrag_min_confidence=0.8)
        defaults = cfg.get_graphrag_query_defaults()
        assert defaults["max_hops"] == 3
        assert defaults["min_confidence"] == 0.8
        assert "semantic_threshold" in defaults
        assert "context_window" in defaults

    def test_is_optional_features_enabled_none(self):
        cfg = SearchConfig()
        assert cfg.is_optional_features_enabled() is False

    def test_is_optional_features_enabled_graphrag(self):
        cfg = SearchConfig(enable_graphrag=True)
        assert cfg.is_optional_features_enabled() is True

    def test_is_optional_features_enabled_metadata(self):
        cfg = SearchConfig(enable_metadata_indexing=True)
        assert cfg.is_optional_features_enabled() is True

    def test_is_optional_features_enabled_qdrant(self):
        cfg = SearchConfig(qdrant_enabled=True)
        assert cfg.is_optional_features_enabled() is True

    def test_validate_optional_config_clean(self):
        cfg = SearchConfig()
        issues = cfg.validate_optional_config()
        assert isinstance(issues, list)

    def test_validate_optional_config_graphrag_without_metadata(self):
        cfg = SearchConfig(enable_graphrag=True, enable_metadata_indexing=False)
        issues = cfg.validate_optional_config()
        assert any("metadata indexing" in i.lower() for i in issues)

    def test_validate_optional_config_bad_qdrant(self):
        cfg = SearchConfig(qdrant_enabled=True, qdrant_vector_size=-1)
        issues = cfg.validate_optional_config()
        assert any("vector size" in i.lower() for i in issues)

    def test_validate_optional_config_bad_distance_metric(self):
        cfg = SearchConfig(qdrant_enabled=True, qdrant_distance_metric="Invalid")
        issues = cfg.validate_optional_config()
        assert any("distance metric" in i.lower() for i in issues)

    def test_validate_optional_config_bad_graphrag_hops(self):
        cfg = SearchConfig(graphrag_max_hops=0)
        issues = cfg.validate_optional_config()
        assert any("max hops" in i.lower() for i in issues)

    def test_validate_optional_config_bad_confidence(self):
        cfg = SearchConfig(graphrag_min_confidence=2.0)
        issues = cfg.validate_optional_config()
        assert any("confidence" in i.lower() for i in issues)

    def test_should_include_language_none(self):
        cfg = SearchConfig(languages=None)
        assert cfg.should_include_language(Language.PYTHON) is True
        assert cfg.should_include_language(Language.UNKNOWN) is False

    def test_should_include_language_specific(self):
        cfg = SearchConfig(languages={Language.PYTHON, Language.JAVASCRIPT})
        assert cfg.should_include_language(Language.PYTHON) is True
        assert cfg.should_include_language(Language.JAVASCRIPT) is True
        assert cfg.should_include_language(Language.GO) is False
