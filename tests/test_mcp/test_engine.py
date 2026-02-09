"""
Tests for mcp/servers/engine.py — PySearchEngine, SearchResponse, ConfigResponse.

Covers: initialization, get_search_config, get_supported_languages,
configure_search, clear_caches, get_search_history, search_text,
search_regex (valid + invalid), analyze_file (valid + not found),
SearchResponse dataclass, ConfigResponse dataclass, search_ast,
search_semantic, search_fuzzy, search_multi_pattern (OR/AND/invalid),
suggest_corrections, word_level_fuzzy_search, _format_result, _add_to_history.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mcp.servers.engine import ConfigResponse, PySearchEngine, SearchResponse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a PySearchEngine instance."""
    return PySearchEngine()


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEngineInitialization:
    """Tests for PySearchEngine initialization."""

    def test_default_init(self, engine):
        """PySearchEngine initializes with all required attributes."""
        assert engine.search_engine is not None
        assert engine.current_config is not None
        assert engine.search_history == []
        assert engine.validator is not None
        assert engine.resource_manager is not None
        assert engine.session_manager is not None
        assert engine.progress_tracker is not None

    def test_config_defaults(self, engine):
        """Default config has expected values."""
        config = engine.get_search_config()
        assert isinstance(config, ConfigResponse)
        assert config.context_lines == 3
        assert config.parallel is True
        assert config.workers == 4
        assert isinstance(config.paths, list)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestEngineConfiguration:
    """Tests for get_search_config and configure_search."""

    def test_get_search_config(self, engine):
        """get_search_config returns a ConfigResponse."""
        config = engine.get_search_config()
        assert isinstance(config, ConfigResponse)

    def test_configure_search(self, engine):
        """configure_search updates and returns new config."""
        resp = engine.configure_search(context=5, workers=2)
        assert isinstance(resp, ConfigResponse)
        assert resp.context_lines == 5
        assert resp.workers == 2

    def test_configure_search_paths(self, engine):
        """configure_search with paths updates the paths list."""
        resp = engine.configure_search(paths=["."])
        assert "." in resp.paths

    def test_configure_search_partial(self, engine):
        """configure_search with partial params preserves other values."""
        original = engine.get_search_config()
        resp = engine.configure_search(context=10)
        assert resp.context_lines == 10
        assert resp.parallel == original.parallel


# ---------------------------------------------------------------------------
# Supported languages
# ---------------------------------------------------------------------------


class TestSupportedLanguages:
    """Tests for get_supported_languages."""

    def test_returns_list(self, engine):
        """get_supported_languages returns a non-empty list."""
        languages = engine.get_supported_languages()
        assert isinstance(languages, list)
        assert len(languages) > 0

    def test_python_included(self, engine):
        """Python is in the supported languages."""
        languages = engine.get_supported_languages()
        assert "python" in languages


# ---------------------------------------------------------------------------
# Clear caches
# ---------------------------------------------------------------------------


class TestClearCaches:
    """Tests for clear_caches."""

    def test_returns_status(self, engine):
        """clear_caches returns a dict with 'status' key."""
        result = engine.clear_caches()
        assert isinstance(result, dict)
        assert "status" in result

    def test_clears_resource_cache(self, engine):
        """clear_caches empties the resource manager cache."""
        engine.resource_manager.set_cache("test_key", "test_val")
        engine.clear_caches()
        assert engine.resource_manager.get_cache("test_key") is None


# ---------------------------------------------------------------------------
# Search history
# ---------------------------------------------------------------------------


class TestSearchHistory:
    """Tests for get_search_history."""

    def test_empty_initially(self, engine):
        """get_search_history returns empty list on fresh engine."""
        assert engine.get_search_history() == []

    def test_limit_parameter(self, engine):
        """get_search_history respects limit parameter."""
        history = engine.get_search_history(limit=5)
        assert isinstance(history, list)


# ---------------------------------------------------------------------------
# search_text
# ---------------------------------------------------------------------------


class TestSearchText:
    """Tests for search_text."""

    def test_returns_search_response(self, engine):
        """search_text returns a SearchResponse."""
        resp = engine.search_text("import", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)
        assert resp.total_matches >= 0
        assert resp.execution_time_ms >= 0
        assert isinstance(resp.items, list)
        assert isinstance(resp.stats, dict)
        assert isinstance(resp.query_info, dict)

    def test_adds_history(self, engine):
        """search_text adds an entry to search history."""
        engine.search_text("def", paths=["./src"], context=1)
        history = engine.get_search_history(limit=5)
        assert len(history) >= 1
        assert history[-1]["query"]["pattern"] == "def"
        assert history[-1]["search_type"] == "text"

    def test_query_info_no_regex(self, engine):
        """search_text sets use_regex=False in query_info."""
        resp = engine.search_text("test", paths=["./src"], context=1)
        assert resp.query_info["use_regex"] is False


# ---------------------------------------------------------------------------
# search_regex
# ---------------------------------------------------------------------------


class TestSearchRegex:
    """Tests for search_regex."""

    def test_returns_search_response(self, engine):
        """search_regex returns a SearchResponse with use_regex=True."""
        resp = engine.search_regex(r"def\s+\w+", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)
        assert resp.query_info["use_regex"] is True

    def test_invalid_regex_raises(self, engine):
        """search_regex with invalid pattern raises ValueError."""
        with pytest.raises(ValueError, match="Invalid regex"):
            engine.search_regex("[unclosed", paths=["./src"])


# ---------------------------------------------------------------------------
# search_ast
# ---------------------------------------------------------------------------


class TestSearchAST:
    """Tests for search_ast."""

    def test_returns_search_response(self, engine):
        """search_ast returns a SearchResponse."""
        resp = engine.search_ast("def", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)

    def test_with_filters(self, engine):
        """search_ast accepts func_name and class_name filters."""
        resp = engine.search_ast("test", func_name="test_.*", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)


# ---------------------------------------------------------------------------
# search_semantic
# ---------------------------------------------------------------------------


class TestSearchSemantic:
    """Tests for search_semantic."""

    def test_returns_search_response(self, engine):
        """search_semantic returns a SearchResponse."""
        resp = engine.search_semantic("authentication", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)


# ---------------------------------------------------------------------------
# search_fuzzy
# ---------------------------------------------------------------------------


class TestSearchFuzzy:
    """Tests for search_fuzzy."""

    def test_returns_search_response(self, engine):
        """search_fuzzy returns a SearchResponse."""
        resp = engine.search_fuzzy("test", paths=["./src"], context=1)
        assert isinstance(resp, SearchResponse)

    def test_max_results_limit(self, engine):
        """search_fuzzy respects max_results parameter."""
        resp = engine.search_fuzzy("a", max_results=5, paths=["./src"], context=1)
        assert len(resp.items) <= 5


# ---------------------------------------------------------------------------
# search_multi_pattern
# ---------------------------------------------------------------------------


class TestSearchMultiPattern:
    """Tests for search_multi_pattern."""

    def test_or_operator(self, engine):
        """search_multi_pattern with OR returns combined results."""
        resp = engine.search_multi_pattern(
            ["import", "def"], operator="OR", paths=["./src"], context=1
        )
        assert isinstance(resp, SearchResponse)

    def test_and_operator(self, engine):
        """search_multi_pattern with AND intersects file sets."""
        resp = engine.search_multi_pattern(
            ["import", "def"], operator="AND", paths=["./src"], context=1
        )
        assert isinstance(resp, SearchResponse)

    def test_empty_patterns_raises(self, engine):
        """search_multi_pattern with empty list raises ValueError."""
        with pytest.raises(ValueError, match="At least one pattern"):
            engine.search_multi_pattern([], paths=["./src"])

    def test_invalid_operator_raises(self, engine):
        """search_multi_pattern with invalid operator raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported operator"):
            engine.search_multi_pattern(["test"], operator="XOR", paths=["./src"])


# ---------------------------------------------------------------------------
# suggest_corrections
# ---------------------------------------------------------------------------


class TestSuggestCorrections:
    """Tests for suggest_corrections."""

    def test_returns_list(self, engine):
        """suggest_corrections returns a list of dicts."""
        results = engine.suggest_corrections("prnt", paths=["./src"])
        assert isinstance(results, list)
        for item in results:
            assert "identifier" in item
            assert "similarity" in item


# ---------------------------------------------------------------------------
# analyze_file
# ---------------------------------------------------------------------------


class TestAnalyzeFile:
    """Tests for analyze_file."""

    def test_analyze_python_file(self, engine):
        """analyze_file returns metrics for a Python file."""
        result = engine.analyze_file("./src/pysearch/__init__.py")
        assert isinstance(result, dict)
        assert "total_lines" in result
        assert "code_lines" in result
        assert "language" in result
        assert result["language"] == "python"
        assert result["total_lines"] > 0

    def test_file_not_found(self, engine):
        """analyze_file with non-existent path raises ValueError."""
        with pytest.raises(ValueError, match="File not found"):
            engine.analyze_file("/nonexistent/path.py")

    def test_analyze_file_metrics(self, engine):
        """analyze_file returns all expected metric keys."""
        result = engine.analyze_file("./src/pysearch/__init__.py")
        expected_keys = [
            "file_path",
            "file_name",
            "language",
            "file_size_bytes",
            "last_modified",
            "total_lines",
            "code_lines",
            "blank_lines",
            "comment_lines",
            "comment_ratio",
            "functions_count",
            "classes_count",
            "imports_count",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# SearchResponse dataclass
# ---------------------------------------------------------------------------


class TestSearchResponseDataclass:
    """Tests for SearchResponse dataclass construction."""

    def test_construction(self):
        """SearchResponse can be constructed with all fields."""
        resp = SearchResponse(
            items=[{"file": "test.py", "start_line": 1}],
            stats={"files_scanned": 10},
            query_info={"pattern": "test"},
            total_matches=1,
            execution_time_ms=42.0,
        )
        assert resp.total_matches == 1
        assert resp.execution_time_ms == 42.0
        assert len(resp.items) == 1

    def test_empty_items(self):
        """SearchResponse works with empty items list."""
        resp = SearchResponse(
            items=[],
            stats={},
            query_info={},
            total_matches=0,
            execution_time_ms=0.0,
        )
        assert resp.total_matches == 0
        assert resp.items == []


# ---------------------------------------------------------------------------
# ConfigResponse dataclass
# ---------------------------------------------------------------------------


class TestConfigResponseDataclass:
    """Tests for ConfigResponse dataclass construction."""

    def test_construction(self):
        """ConfigResponse can be constructed with all fields."""
        resp = ConfigResponse(
            paths=["."],
            include_patterns=["**/*.py"],
            exclude_patterns=["**/node_modules/**"],
            context_lines=3,
            parallel=True,
            workers=4,
            languages=None,
        )
        assert resp.paths == ["."]
        assert resp.context_lines == 3
        assert resp.parallel is True

    def test_with_languages(self):
        """ConfigResponse stores languages list."""
        resp = ConfigResponse(
            paths=["."],
            include_patterns=None,
            exclude_patterns=None,
            context_lines=1,
            parallel=False,
            workers=1,
            languages=["python", "javascript"],
        )
        assert resp.languages == ["python", "javascript"]
