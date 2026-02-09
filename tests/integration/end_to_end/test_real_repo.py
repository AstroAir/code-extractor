"""End-to-end integration tests using the real sample_repo test fixture.

Validates all core pysearch features against a realistic multi-module
Python project (test_data/sample_repo/) instead of trivial inline snippets.

Test categories:
    - Text and regex search
    - AST structural search
    - Boolean search
    - Fuzzy search
    - Multi-language search and language detection
    - Dependency analysis
    - IDE integration (jump-to-def, find-references, completions, hover)
    - GraphRAG entity extraction
    - Output formats
    - Edge cases
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch import (
    ASTFilters,
    Language,
    MetadataFilters,
    OutputFormat,
    PySearch,
    Query,
    SearchConfig,
    SearchResult,
)
from pysearch.analysis.dependency_analysis import DependencyAnalyzer
from pysearch.analysis.language_detection import detect_language
from pysearch.integrations.ide_hooks import IDEIntegration, ide_query

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_REPO = Path(__file__).resolve().parents[3] / "test_data" / "sample_repo"


@pytest.fixture(scope="module")
def repo_path() -> Path:
    """Return the path to the sample repository."""
    assert SAMPLE_REPO.is_dir(), f"sample_repo not found at {SAMPLE_REPO}"
    return SAMPLE_REPO


@pytest.fixture(scope="module")
def py_engine(repo_path: Path) -> PySearch:
    """Create a PySearch engine configured for the sample repository (Python only)."""
    cfg = SearchConfig(
        paths=[str(repo_path)],
        include=["**/*.py"],
        exclude=["**/__pycache__/**", "**/.pysearch-cache/**"],
        context=2,
        parallel=False,
    )
    return PySearch(cfg)


@pytest.fixture(scope="module")
def all_lang_engine(repo_path: Path) -> PySearch:
    """Create a PySearch engine for all languages in the sample repository."""
    cfg = SearchConfig(
        paths=[str(repo_path)],
        include=["**/*.py", "**/*.js", "**/*.ts", "**/*.css", "**/*.md", "**/*.toml"],
        exclude=["**/__pycache__/**", "**/.pysearch-cache/**"],
        context=2,
        parallel=False,
    )
    return PySearch(cfg)


@pytest.fixture(scope="module")
def ide_integration(py_engine: PySearch) -> IDEIntegration:
    """Create an IDEIntegration instance backed by the sample repo engine."""
    return IDEIntegration(py_engine)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _files_in_results(result: SearchResult) -> set[str]:
    """Extract unique file names from search results."""
    return {item.file.name for item in result.items}


def _all_lines(result: SearchResult) -> str:
    """Concatenate all result lines into one string for easy assertion."""
    parts: list[str] = []
    for item in result.items:
        parts.extend(item.lines)
    return "\n".join(parts)


# ===========================================================================
# 1. Text Search
# ===========================================================================


class TestTextSearch:
    """Verify basic text and regex search against real project files."""

    def test_search_function_definition(self, py_engine: PySearch) -> None:
        """Find a known function definition."""
        result = py_engine.search("def create_app", output=OutputFormat.TEXT)
        assert result.items, "Expected to find 'def create_app' in sample_repo"
        assert any("create_app" in "\n".join(it.lines) for it in result.items)

    def test_search_class_definition(self, py_engine: PySearch) -> None:
        """Find a known class definition."""
        result = py_engine.search("class User", output=OutputFormat.TEXT)
        assert result.items, "Expected to find 'class User'"
        files = _files_in_results(result)
        assert "user.py" in files

    def test_search_regex_pattern(self, py_engine: PySearch) -> None:
        """Regex search for all async function definitions."""
        result = py_engine.search(r"async\s+def\s+\w+", regex=True, output=OutputFormat.TEXT)
        assert len(result.items) >= 3, "Expected at least 3 async functions in sample_repo"

    def test_search_with_context_lines(self, py_engine: PySearch) -> None:
        """Ensure context lines are included."""
        result = py_engine.search("def validate", context=3, output=OutputFormat.TEXT)
        assert result.items
        # With context=3, each result should have more lines than just the match
        for item in result.items:
            assert len(item.lines) >= 2, "Context lines should be included"

    def test_search_returns_stats(self, py_engine: PySearch) -> None:
        """Search stats should report realistic numbers."""
        result = py_engine.search("import", output=OutputFormat.TEXT)
        assert result.stats.files_scanned > 0
        assert result.stats.items > 0
        assert result.stats.elapsed_ms >= 0

    def test_search_no_match(self, py_engine: PySearch) -> None:
        """Search for something that doesn't exist."""
        result = py_engine.search("zzz_nonexistent_pattern_xyz_42", output=OutputFormat.TEXT)
        assert len(result.items) == 0

    def test_search_count_only(self, py_engine: PySearch) -> None:
        """Count-only search should return counts without full content."""
        count_result = py_engine.search_count_only("def ")
        assert count_result.total_matches > 0
        assert count_result.files_matched > 0

    def test_search_todo_comments(self, py_engine: PySearch) -> None:
        """Find TODO/FIXME comments in the sample repo."""
        result = py_engine.search(r"(TODO|FIXME)", regex=True, output=OutputFormat.TEXT)
        assert len(result.items) >= 2, "Expected at least 2 TODO/FIXME comments"

    def test_search_decorator_pattern(self, py_engine: PySearch) -> None:
        """Find all decorated functions via regex."""
        result = py_engine.search(r"@\w+", regex=True, output=OutputFormat.TEXT)
        assert len(result.items) >= 3, "Expected at least 3 decorators in sample_repo"

    def test_search_import_pattern(self, py_engine: PySearch) -> None:
        """Find all import statements."""
        result = py_engine.search(r"^from\s+src\.", regex=True, output=OutputFormat.TEXT)
        assert len(result.items) >= 5, "Expected cross-module imports in sample_repo"


# ===========================================================================
# 2. AST Search
# ===========================================================================


class TestASTSearch:
    """Verify AST-based structural search features."""

    def test_ast_find_functions_by_name(self, py_engine: PySearch) -> None:
        """Find a function by name using AST filters."""
        filters = ASTFilters(func_name="create_app")
        q = Query(
            pattern="def",
            use_regex=False,
            use_ast=True,
            context=1,
            output=OutputFormat.TEXT,
            filters=filters,
        )
        result = py_engine.run(q)
        assert result.items, "AST search should find 'create_app'"
        assert any("create_app" in "\n".join(it.lines) for it in result.items)

    def test_ast_find_classes(self, py_engine: PySearch) -> None:
        """Find classes using AST filters."""
        filters = ASTFilters(class_name=".*Service")
        q = Query(
            pattern="class",
            use_regex=False,
            use_ast=True,
            context=0,
            output=OutputFormat.TEXT,
            filters=filters,
        )
        result = py_engine.run(q)
        assert result.items, "AST search should find *Service classes"
        lines_text = _all_lines(result)
        assert (
            "AuthService" in lines_text
            or "UserService" in lines_text
            or "PostService" in lines_text
        )

    def test_ast_find_decorated_functions(self, py_engine: PySearch) -> None:
        """Find functions with a specific decorator."""
        filters = ASTFilters(decorator="cached")
        q = Query(
            pattern="def",
            use_regex=False,
            use_ast=True,
            context=1,
            output=OutputFormat.TEXT,
            filters=filters,
        )
        result = py_engine.run(q)
        assert result.items, "AST search should find @cached decorated functions"

    def test_ast_find_async_functions(self, py_engine: PySearch) -> None:
        """Find async functions via AST."""
        filters = ASTFilters(func_name=".*")
        q = Query(
            pattern="async def",
            use_regex=False,
            use_ast=True,
            context=0,
            output=OutputFormat.TEXT,
            filters=filters,
        )
        result = py_engine.run(q)
        assert len(result.items) >= 3, "Expected multiple async functions"


# ===========================================================================
# 3. Boolean Search
# ===========================================================================


class TestBooleanSearch:
    """Verify boolean (AND/OR/NOT) query logic."""

    def test_boolean_and(self, py_engine: PySearch) -> None:
        """AND query: both terms must appear in matched context."""
        q = Query(
            pattern="password AND hash",
            use_boolean=True,
            context=2,
            output=OutputFormat.TEXT,
        )
        result = py_engine.run(q)
        assert result.items, "Boolean AND should find lines with both 'password' and 'hash'"

    def test_boolean_or(self, py_engine: PySearch) -> None:
        """OR query: at least one term must appear."""
        q = Query(
            pattern="BaseModel OR dataclass",
            use_boolean=True,
            context=0,
            output=OutputFormat.TEXT,
        )
        result = py_engine.run(q)
        assert result.items, "Boolean OR should find results for BaseModel/dataclass"

    def test_boolean_not(self, py_engine: PySearch) -> None:
        """NOT query: exclude certain terms."""
        q = Query(
            pattern="async AND def NOT test",
            use_boolean=True,
            context=0,
            output=OutputFormat.TEXT,
        )
        result = py_engine.run(q)
        assert result.items, "Boolean NOT should still return some results"


# ===========================================================================
# 4. Multi-Language Search
# ===========================================================================


class TestMultiLanguageSearch:
    """Verify search across multiple programming languages."""

    def test_search_across_languages(self, all_lang_engine: PySearch) -> None:
        """Search a common term across Python, JS, TS files."""
        result = all_lang_engine.search("function", output=OutputFormat.TEXT)
        files = _files_in_results(result)
        # 'function' should appear in JS files and possibly TS/Python docstrings
        assert "app.js" in files, "Expected match in JavaScript file"

    def test_search_in_typescript(self, all_lang_engine: PySearch) -> None:
        """Search for TypeScript-specific patterns."""
        result = all_lang_engine.search("interface", output=OutputFormat.TEXT)
        files = _files_in_results(result)
        assert "types.ts" in files, "Expected match in TypeScript file"

    def test_search_in_css(self, all_lang_engine: PySearch) -> None:
        """Search for CSS variable definitions."""
        result = all_lang_engine.search("--color-primary", output=OutputFormat.TEXT)
        files = _files_in_results(result)
        assert "styles.css" in files, "Expected match in CSS file"

    def test_search_in_markdown(self, all_lang_engine: PySearch) -> None:
        """Search for content in README."""
        result = all_lang_engine.search("Quick Start", output=OutputFormat.TEXT)
        files = _files_in_results(result)
        assert "README.md" in files

    def test_language_detection_python(self, repo_path: Path) -> None:
        """Detect Python language from file extension."""
        assert detect_language(repo_path / "src" / "app.py") == Language.PYTHON

    def test_language_detection_javascript(self, repo_path: Path) -> None:
        """Detect JavaScript language."""
        assert detect_language(repo_path / "web" / "app.js") == Language.JAVASCRIPT

    def test_language_detection_typescript(self, repo_path: Path) -> None:
        """Detect TypeScript language."""
        assert detect_language(repo_path / "web" / "types.ts") == Language.TYPESCRIPT

    def test_language_detection_dockerfile(self, repo_path: Path) -> None:
        """Detect Dockerfile."""
        assert detect_language(repo_path / "Dockerfile") == Language.DOCKERFILE

    def test_language_detection_makefile(self, repo_path: Path) -> None:
        """Detect Makefile."""
        assert detect_language(repo_path / "Makefile") == Language.MAKEFILE


# ===========================================================================
# 5. Dependency Analysis
# ===========================================================================


class TestDependencyAnalysis:
    """Verify dependency analysis against the real project structure."""

    @pytest.fixture
    def analyzer(self) -> DependencyAnalyzer:
        return DependencyAnalyzer()

    def test_build_dependency_graph(self, analyzer: DependencyAnalyzer, repo_path: Path) -> None:
        """Build a dependency graph from the sample repo."""
        graph = analyzer.analyze_directory(repo_path / "src")
        assert len(graph.nodes) >= 5, f"Expected at least 5 modules, got {len(graph.nodes)}"
        assert len(graph.edges) >= 3, f"Expected at least 3 edges, got {len(graph.edges)}"

    def test_dependency_metrics(self, analyzer: DependencyAnalyzer, repo_path: Path) -> None:
        """Calculate dependency metrics."""
        analyzer.analyze_directory(repo_path / "src")
        metrics = analyzer.calculate_metrics()
        assert metrics.total_modules >= 5
        assert metrics.total_dependencies >= 3

    def test_cross_module_imports_detected(
        self, analyzer: DependencyAnalyzer, repo_path: Path
    ) -> None:
        """Verify that cross-module imports (e.g. services -> models) are detected."""
        graph = analyzer.analyze_directory(repo_path / "src")
        # There should be edges connecting different packages
        edge_strs = [f"{e}" for e in graph.edges]
        assert len(edge_strs) > 0, "Expected cross-module dependency edges"

    def test_engine_dependency_analysis(self, py_engine: PySearch, repo_path: Path) -> None:
        """Test dependency analysis via the PySearch engine API."""
        graph = py_engine.analyze_dependencies(directory=repo_path / "src")
        assert graph is not None
        assert hasattr(graph, "nodes")


# ===========================================================================
# 6. IDE Integration
# ===========================================================================


class TestIDEIntegration:
    """Verify IDE hook features against real code."""

    def test_jump_to_definition_function(self, ide_integration: IDEIntegration) -> None:
        """Jump to definition of a known function."""
        loc = ide_integration.jump_to_definition("src/app.py", 1, "create_app")
        assert loc is not None, "Should find definition of 'create_app'"
        assert loc.symbol_name == "create_app"
        assert loc.symbol_type == "function"
        assert loc.line >= 1

    def test_jump_to_definition_class(self, ide_integration: IDEIntegration) -> None:
        """Jump to definition of a known class."""
        loc = ide_integration.jump_to_definition("src/models/user.py", 1, "User")
        assert loc is not None, "Should find definition of 'User'"
        assert loc.symbol_name == "User"
        assert loc.symbol_type == "class"

    def test_find_references(self, ide_integration: IDEIntegration) -> None:
        """Find all references to a symbol across the codebase."""
        refs = ide_integration.find_references("src/config.py", 1, "Settings")
        assert len(refs) >= 2, "Expected multiple references to 'Settings'"

    def test_find_references_with_definition(self, ide_integration: IDEIntegration) -> None:
        """Find references including the definition itself."""
        refs = ide_integration.find_references(
            "src/models/base.py", 1, "BaseModel", include_definition=True
        )
        has_def = any(r.is_definition for r in refs)
        assert has_def, "Should include the definition location"

    def test_provide_completion(self, ide_integration: IDEIntegration) -> None:
        """Get completions for a prefix."""
        completions = ide_integration.provide_completion("test.py", 1, 0, prefix="create")
        assert len(completions) >= 1, "Expected completions starting with 'create'"
        labels = [c.label for c in completions]
        assert any("create" in label for label in labels)

    def test_provide_hover(self, ide_integration: IDEIntegration) -> None:
        """Get hover info for a symbol."""
        hover = ide_integration.provide_hover("test.py", 1, 0, symbol="validate")
        assert hover is not None, "Expected hover info for 'validate'"
        assert hover.symbol_name == "validate"

    def test_ide_query_json(self, py_engine: PySearch) -> None:
        """Test the ide_query convenience function."""
        q = Query(pattern="class.*Service", use_regex=True, output=OutputFormat.JSON, context=1)
        result = ide_query(py_engine, q)
        assert isinstance(result, dict)
        assert "items" in result
        assert "stats" in result
        assert len(result["items"]) >= 1


# ===========================================================================
# 7. GraphRAG Entity Extraction
# ===========================================================================


class TestGraphRAG:
    """Verify GraphRAG entity extraction on real code."""

    @pytest.mark.asyncio
    async def test_entity_extraction(self, repo_path: Path) -> None:
        """Extract entities from a real Python file."""
        from pysearch.analysis.graphrag import EntityExtractor

        extractor = EntityExtractor()
        file_path = repo_path / "src" / "models" / "user.py"
        entities = await extractor.extract_from_file(file_path)

        entity_names = {e.name for e in entities}
        # Should find the User class, UserRole enum, UserProfile class
        assert (
            "User" in entity_names or "UserRole" in entity_names
        ), f"Expected to find User or UserRole in entities, got: {entity_names}"

    @pytest.mark.asyncio
    async def test_relationship_mapping(self, repo_path: Path) -> None:
        """Extract relationships between entities."""
        from pysearch.analysis.graphrag import EntityExtractor, RelationshipMapper

        extractor = EntityExtractor()
        file_path = repo_path / "src" / "models" / "user.py"
        entities = await extractor.extract_from_file(file_path)

        mapper = RelationshipMapper()
        file_contents = {file_path: file_path.read_text(encoding="utf-8")}
        relationships = await mapper.map_relationships(entities, file_contents)
        # There should be at least some relationships (e.g. User contains methods)
        assert isinstance(relationships, list)

    @pytest.mark.asyncio
    async def test_knowledge_graph_construction(self, repo_path: Path) -> None:
        """Build a knowledge graph from multiple files."""
        from pysearch import KnowledgeGraph
        from pysearch.analysis.graphrag import EntityExtractor

        extractor = EntityExtractor()
        kg = KnowledgeGraph()

        py_files = list((repo_path / "src").rglob("*.py"))
        assert len(py_files) >= 5, "Expected at least 5 Python files"

        for py_file in py_files[:5]:  # Limit to 5 files for speed
            entities = await extractor.extract_from_file(py_file)
            for entity in entities:
                kg.add_entity(entity)

        assert (
            len(kg.entities) >= 3
        ), f"Expected at least 3 entities in knowledge graph, got {len(kg.entities)}"


# ===========================================================================
# 8. Output Formats
# ===========================================================================


class TestOutputFormats:
    """Verify different output format modes."""

    def test_text_output(self, py_engine: PySearch) -> None:
        """TEXT output should return usable results."""
        result = py_engine.search("class", output=OutputFormat.TEXT)
        assert result.items
        for item in result.items:
            assert item.file is not None
            assert item.start_line >= 1
            assert isinstance(item.lines, list)

    def test_json_output(self, py_engine: PySearch) -> None:
        """JSON output should return usable results."""
        result = py_engine.search("class", output=OutputFormat.JSON)
        assert result.items
        for item in result.items:
            assert item.file is not None
            assert item.start_line >= 1

    def test_highlight_output(self, py_engine: PySearch) -> None:
        """HIGHLIGHT output should return usable results."""
        result = py_engine.search("class", output=OutputFormat.HIGHLIGHT)
        assert result.items


# ===========================================================================
# 9. Semantic Search
# ===========================================================================


class TestSemanticSearch:
    """Verify lightweight semantic search on real code."""

    def test_semantic_search_database(self, py_engine: PySearch) -> None:
        """Semantic search for 'database connection' should find db-related code."""
        result = py_engine.search_semantic("database connection", threshold=0.05, max_results=20)
        assert isinstance(result, SearchResult)
        if result.items:
            files = _files_in_results(result)
            # connection.py or migrations.py should be among results
            assert any(
                f in files for f in ("connection.py", "migrations.py", "config.py")
            ), f"Expected db-related files, got: {files}"

    def test_semantic_search_authentication(self, py_engine: PySearch) -> None:
        """Semantic search for 'user authentication' should find auth-related code."""
        result = py_engine.search_semantic("user authentication", threshold=0.05, max_results=20)
        assert isinstance(result, SearchResult)


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge case handling on real project structure."""

    def test_search_empty_pattern(self, py_engine: PySearch) -> None:
        """Searching with an empty pattern should not crash."""
        result = py_engine.search("", output=OutputFormat.TEXT)
        assert isinstance(result, SearchResult)

    def test_search_special_regex_chars(self, py_engine: PySearch) -> None:
        """Special regex characters in non-regex mode should be safe."""
        result = py_engine.search("dict[str, Any]", output=OutputFormat.TEXT)
        assert isinstance(result, SearchResult)

    def test_search_unicode_content(self, py_engine: PySearch) -> None:
        """Searching for ASCII in files with potential unicode should work."""
        result = py_engine.search("utf-8", output=OutputFormat.TEXT)
        assert isinstance(result, SearchResult)

    def test_engine_clear_caches(self, py_engine: PySearch) -> None:
        """Cache clearing should not break subsequent searches."""
        py_engine.clear_caches()
        result = py_engine.search("def", output=OutputFormat.TEXT)
        assert result.items, "Search should work after cache clear"

    def test_multiple_searches_consistency(self, py_engine: PySearch) -> None:
        """Running the same search twice should give consistent results."""
        r1 = py_engine.search("class User", output=OutputFormat.TEXT)
        py_engine.clear_caches()
        r2 = py_engine.search("class User", output=OutputFormat.TEXT)
        assert len(r1.items) == len(r2.items), "Repeated search should be consistent"

    def test_max_per_file(self, py_engine: PySearch) -> None:
        """Per-file result limit should be respected."""
        q = Query(
            pattern="def",
            use_regex=False,
            context=0,
            output=OutputFormat.TEXT,
            max_per_file=1,
        )
        result = py_engine.run(q)
        # Each file should contribute at most 1 result
        file_counts: dict[str, int] = {}
        for item in result.items:
            name = str(item.file)
            file_counts[name] = file_counts.get(name, 0) + 1
        for fname, count in file_counts.items():
            assert count <= 1, f"File {fname} has {count} results, expected at most 1"


# ===========================================================================
# 11. Fuzzy Search
# ===========================================================================


class TestFuzzySearch:
    """Verify fuzzy matching algorithms on real code tokens."""

    def test_fuzzy_match_levenshtein(self) -> None:
        """Levenshtein fuzzy matching should find close matches."""
        from pysearch.search.fuzzy import FuzzyAlgorithm, fuzzy_match

        text = "class UserService:\n    def create_user(self):\n        pass"
        matches = fuzzy_match(
            text,
            "UserServce",
            max_distance=2,
            min_similarity=0.7,
            algorithm=FuzzyAlgorithm.LEVENSHTEIN,
        )
        matched_words = [m.matched_text for m in matches]
        assert (
            "UserService" in matched_words
        ), f"Expected 'UserService' in matches, got {matched_words}"

    def test_fuzzy_match_jaro_winkler(self) -> None:
        """Jaro-Winkler should handle common prefix similarity."""
        from pysearch.search.fuzzy import FuzzyAlgorithm, fuzzy_match

        text = "def validate_email(email):\n    pass"
        matches = fuzzy_match(
            text,
            "validate",
            max_distance=3,
            min_similarity=0.6,
            algorithm=FuzzyAlgorithm.JARO_WINKLER,
        )
        matched_words = [m.matched_text for m in matches]
        assert any("validate" in w for w in matched_words)

    def test_fuzzy_search_advanced_multi_algo(self) -> None:
        """Advanced fuzzy search using multiple algorithms."""
        from pysearch.search.fuzzy import FuzzyAlgorithm, fuzzy_search_advanced

        text = "class DatabasePool:\n    async def connect(self):\n        pass"
        matches = fuzzy_search_advanced(
            text,
            "DatabsePool",
            algorithms=[FuzzyAlgorithm.LEVENSHTEIN, FuzzyAlgorithm.JARO_WINKLER],
            max_distance=2,
            min_similarity=0.7,
            combine_results=True,
        )
        matched_words = [m.matched_text for m in matches]
        assert "DatabasePool" in matched_words

    def test_suggest_corrections(self) -> None:
        """Suggest corrections for a misspelled identifier."""
        from pysearch.search.fuzzy import suggest_corrections

        dictionary = ["UserService", "PostService", "AuthService", "DatabasePool", "Settings"]
        suggestions = suggest_corrections("UserServce", dictionary, max_suggestions=3)
        assert len(suggestions) >= 1
        assert suggestions[0][0] == "UserService"
        assert suggestions[0][1] > 0.8

    def test_fuzzy_match_on_real_file(self, repo_path: Path) -> None:
        """Run fuzzy search on real file content from sample_repo."""
        from pysearch.search.fuzzy import FuzzyAlgorithm, fuzzy_match

        code = (repo_path / "src" / "services" / "auth.py").read_text(encoding="utf-8")
        matches = fuzzy_match(
            code,
            "AuthServce",
            max_distance=2,
            min_similarity=0.7,
            algorithm=FuzzyAlgorithm.LEVENSHTEIN,
        )
        matched_words = [m.matched_text for m in matches]
        assert "AuthService" in matched_words

    def test_calculate_similarity(self) -> None:
        """Similarity calculation across different algorithms."""
        from pysearch.search.fuzzy import FuzzyAlgorithm, calculate_similarity

        sim_lev = calculate_similarity("create_user", "create_user", FuzzyAlgorithm.LEVENSHTEIN)
        assert sim_lev == 1.0

        sim_typo = calculate_similarity("create_user", "crete_user", FuzzyAlgorithm.LEVENSHTEIN)
        assert 0.8 <= sim_typo < 1.0

        sim_jw = calculate_similarity("create_user", "create_usr", FuzzyAlgorithm.JARO_WINKLER)
        assert sim_jw > 0.8


# ===========================================================================
# 12. Search History
# ===========================================================================


class TestSearchHistory:
    """Verify search history tracking on real searches."""

    def test_history_records_search(self, py_engine: PySearch) -> None:
        """Search history should record queries."""
        py_engine.history._history.clear()
        py_engine.clear_caches()
        py_engine.search("class User", output=OutputFormat.TEXT)
        assert len(py_engine.history._history) >= 1
        last = py_engine.history._history[-1]
        assert last.query_pattern == "class User"
        assert last.items_count >= 1

    def test_history_categorization(self, py_engine: PySearch) -> None:
        """Searches should be auto-categorized."""
        from pysearch.core.history import SearchCategory

        py_engine.history._history.clear()
        py_engine.clear_caches()

        py_engine.search("def create_app", output=OutputFormat.TEXT)
        last = py_engine.history._history[-1]
        assert last.category == SearchCategory.FUNCTION

        py_engine.clear_caches()
        py_engine.search("class User", output=OutputFormat.TEXT)
        last = py_engine.history._history[-1]
        assert last.category == SearchCategory.CLASS

        py_engine.clear_caches()
        py_engine.search("import asyncio", output=OutputFormat.TEXT)
        last = py_engine.history._history[-1]
        assert last.category == SearchCategory.IMPORT

    def test_history_success_score(self, py_engine: PySearch) -> None:
        """History entries should have a success score."""
        py_engine.history._history.clear()
        py_engine.clear_caches()
        py_engine.search("def validate", output=OutputFormat.TEXT)
        last = py_engine.history._history[-1]
        assert 0.0 <= last.success_score <= 1.0
        assert last.elapsed_ms >= 0

    def test_history_tracks_languages(self, py_engine: PySearch) -> None:
        """History should track which languages appeared in results."""
        py_engine.history._history.clear()
        py_engine.clear_caches()
        py_engine.search("def", output=OutputFormat.TEXT)
        last = py_engine.history._history[-1]
        if last.languages:
            assert "python" in last.languages

    def test_history_empty_results(self, py_engine: PySearch) -> None:
        """History should record even searches with no results."""
        py_engine.history._history.clear()
        py_engine.clear_caches()
        py_engine.search("zzz_never_matches_xyz_999", output=OutputFormat.TEXT)
        assert len(py_engine.history._history) >= 1
        last = py_engine.history._history[-1]
        assert last.items_count == 0


# ===========================================================================
# 13. Metadata Filters
# ===========================================================================


class TestMetadataFilters:
    """Verify metadata-based filtering on real project files."""

    def test_filter_by_language(self, repo_path: Path) -> None:
        """Filter search results to specific languages."""
        cfg = SearchConfig(
            paths=[str(repo_path)],
            include=["**/*.py", "**/*.js", "**/*.ts"],
            exclude=["**/__pycache__/**"],
            context=0,
            parallel=False,
        )
        engine = PySearch(cfg)
        q = Query(
            pattern="function",
            use_regex=False,
            context=0,
            output=OutputFormat.TEXT,
            metadata_filters=MetadataFilters(languages={Language.JAVASCRIPT}),
        )
        result = engine.run(q)
        # All results should be from JS files only
        for item in result.items:
            assert item.file.suffix in (".js",), f"Expected .js file, got {item.file}"

    def test_filter_by_min_size(self, repo_path: Path) -> None:
        """Filter out small files by minimum size."""
        cfg = SearchConfig(
            paths=[str(repo_path)],
            include=["**/*.py"],
            exclude=["**/__pycache__/**"],
            context=0,
            parallel=False,
        )
        engine = PySearch(cfg)
        q = Query(
            pattern="def",
            use_regex=False,
            context=0,
            output=OutputFormat.TEXT,
            metadata_filters=MetadataFilters(min_size=500),
        )
        result = engine.run(q)
        # Should only get results from files >= 500 bytes
        if result.items:
            for item in result.items:
                assert item.file.stat().st_size >= 500

    def test_filter_by_max_lines(self, repo_path: Path) -> None:
        """Filter files by maximum line count."""
        cfg = SearchConfig(
            paths=[str(repo_path)],
            include=["**/*.py"],
            exclude=["**/__pycache__/**"],
            context=0,
            parallel=False,
        )
        engine = PySearch(cfg)
        q = Query(
            pattern="class",
            use_regex=False,
            context=0,
            output=OutputFormat.TEXT,
            metadata_filters=MetadataFilters(max_lines=30),
        )
        result = engine.run(q)
        # Results should only be from small files
        for item in result.items:
            content = item.file.read_text(encoding="utf-8")
            line_count = content.count("\n") + 1
            assert line_count <= 30, f"File {item.file} has {line_count} lines, expected <= 30"


# ===========================================================================
# 14. IDE Integration Extended
# ===========================================================================


class TestIDEIntegrationExtended:
    """Additional IDE features: document symbols, workspace symbols, diagnostics."""

    def test_document_symbols(self, ide_integration: IDEIntegration, repo_path: Path) -> None:
        """List all symbols in a file."""
        file_path = str(repo_path / "src" / "models" / "user.py")
        symbols = ide_integration.get_document_symbols(file_path)
        assert len(symbols) >= 3, f"Expected at least 3 symbols, got {len(symbols)}"

        symbol_names = {s.name for s in symbols}
        # Should find classes and functions
        assert (
            "User" in symbol_names or "UserRole" in symbol_names
        ), f"Expected User or UserRole in symbols, got: {symbol_names}"

        kinds = {s.kind for s in symbols}
        assert "class" in kinds, "Expected 'class' kind in document symbols"

    def test_document_symbols_with_functions(
        self, ide_integration: IDEIntegration, repo_path: Path
    ) -> None:
        """Document symbols should include function definitions."""
        file_path = str(repo_path / "src" / "utils" / "helpers.py")
        symbols = ide_integration.get_document_symbols(file_path)
        func_symbols = [s for s in symbols if s.kind == "function"]
        assert (
            len(func_symbols) >= 3
        ), f"Expected at least 3 functions in helpers.py, got {len(func_symbols)}"

    def test_document_symbols_constants(
        self, ide_integration: IDEIntegration, repo_path: Path
    ) -> None:
        """Document symbols should detect UPPER_CASE constants."""
        file_path = str(repo_path / "src" / "config.py")
        symbols = ide_integration.get_document_symbols(file_path)
        var_symbols = [s for s in symbols if s.kind == "variable"]
        assert len(var_symbols) >= 1, "Expected at least 1 constant in config.py"

    def test_workspace_symbols_search(self, ide_integration: IDEIntegration) -> None:
        """Search for symbols across the workspace."""
        symbols = ide_integration.get_workspace_symbols("Service")
        assert len(symbols) >= 2, f"Expected at least 2 *Service symbols, got {len(symbols)}"
        names = {s.name for s in symbols}
        assert any("Service" in n for n in names)

    def test_workspace_symbols_short_query(self, ide_integration: IDEIntegration) -> None:
        """Short queries (< 2 chars) should return empty."""
        symbols = ide_integration.get_workspace_symbols("a")
        assert symbols == []

    def test_diagnostics_finds_todos(
        self, ide_integration: IDEIntegration, repo_path: Path
    ) -> None:
        """Diagnostics should detect TODO/FIXME markers."""
        file_path = str(repo_path / "src" / "app.py")
        diagnostics = ide_integration.get_diagnostics(file_path)
        codes = [d.code for d in diagnostics]
        assert (
            "TODO" in codes or "FIXME" in codes
        ), f"Expected TODO or FIXME diagnostics in app.py, got codes: {codes}"

    def test_diagnostics_severity(self, ide_integration: IDEIntegration, repo_path: Path) -> None:
        """FIXME should have 'warning' severity, TODO should have 'info'."""
        file_path = str(repo_path / "src" / "app.py")
        diagnostics = ide_integration.get_diagnostics(file_path)
        for diag in diagnostics:
            if diag.code == "TODO":
                assert diag.severity == "info"
            elif diag.code in ("FIXME", "HACK", "XXX"):
                assert diag.severity == "warning"

    def test_diagnostics_nonexistent_file(self, ide_integration: IDEIntegration) -> None:
        """Diagnostics for a nonexistent file should return empty."""
        diagnostics = ide_integration.get_diagnostics("/nonexistent/path/to/file.py")
        assert diagnostics == []

    def test_document_symbols_nonexistent_file(self, ide_integration: IDEIntegration) -> None:
        """Document symbols for a nonexistent file should return empty."""
        symbols = ide_integration.get_document_symbols("/nonexistent/path/to/file.py")
        assert symbols == []


# ===========================================================================
# 15. IDE Hooks Registry
# ===========================================================================


class TestIDEHooksRegistry:
    """Verify the IDEHooks hook registration and trigger system."""

    def test_register_and_trigger_search_hook(self, py_engine: PySearch) -> None:
        """Register a search hook and trigger it."""
        from pysearch.integrations.ide_hooks import IDEHooks

        hooks = IDEHooks()
        hook_id = hooks.register_search_handler(
            lambda query=None: py_engine.search("def validate", output=OutputFormat.TEXT)
        )
        result = hooks.trigger_hook(hook_id)
        assert result is not None
        assert hasattr(result, "items")
        assert len(result.items) >= 1

    def test_register_and_unregister_hook(self) -> None:
        """Hook registration and unregistration."""
        from pysearch.integrations.ide_hooks import IDEHooks

        hooks = IDEHooks()
        hook_id = hooks.register_jump_to_definition(lambda **kw: None)
        assert hooks.unregister_hook(hook_id) is True
        assert hooks.unregister_hook(hook_id) is False  # already removed

    def test_trigger_missing_hook(self) -> None:
        """Triggering a non-existent hook should return None."""
        from pysearch.integrations.ide_hooks import IDEHooks

        hooks = IDEHooks()
        result = hooks.trigger_hook("nonexistent_hook_id")
        assert result is None

    def test_list_hooks(self) -> None:
        """List all registered hooks."""
        from pysearch.integrations.ide_hooks import IDEHooks

        hooks = IDEHooks()
        hooks.register_search_handler(lambda **kw: None)
        hooks.register_completion_handler(lambda **kw: None)
        hooks.register_hover_handler(lambda **kw: None)

        hook_list = hooks.list_hooks()
        assert len(hook_list) == 3
        types = {h["type"] for h in hook_list}
        assert "search" in types
        assert "completion" in types
        assert "hover" in types

    def test_hook_error_handling(self) -> None:
        """Hook that raises an exception should not crash."""
        from pysearch.integrations.ide_hooks import IDEHooks

        hooks = IDEHooks()
        hook_id = hooks.register_search_handler(
            lambda **kw: (_ for _ in ()).throw(ValueError("test error"))
        )
        result = hooks.trigger_hook(hook_id)
        assert result is None


# ===========================================================================
# 16. Advanced Dependency Analysis
# ===========================================================================


class TestAdvancedDependencyAnalysis:
    """Advanced dependency analysis features on real project."""

    def test_circular_dependency_detection(self, repo_path: Path) -> None:
        """Detect circular dependencies (user_service <-> post_service)."""
        from pysearch.analysis.dependency_analysis import (
            CircularDependencyDetector,
            DependencyAnalyzer,
        )

        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(repo_path / "src")
        detector = CircularDependencyDetector(graph)
        cycles = detector.find_cycles()
        # user_service and post_service import each other
        assert isinstance(cycles, list)
        # The sample repo has intentional circular deps between user_service and post_service

    def test_graph_export_to_dict(self, repo_path: Path) -> None:
        """Export dependency graph to dictionary format."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(repo_path / "src")
        data = graph.to_dict()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) >= 5
        assert len(data["edges"]) >= 3
        # Each edge has required fields
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert "weight" in edge

    def test_transitive_dependencies(self, repo_path: Path) -> None:
        """Compute transitive dependencies for a module."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(repo_path / "src")

        # Find a node that has dependencies
        nodes_with_deps = [n for n in graph.nodes if graph.get_dependencies(n)]
        assert len(nodes_with_deps) >= 1

        trans_deps = graph.get_transitive_dependencies(nodes_with_deps[0])
        assert isinstance(trans_deps, set)

    def test_dependency_graph_has_path(self, repo_path: Path) -> None:
        """Check path existence between modules."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(repo_path / "src")

        # A module has a path to itself
        for node in list(graph.nodes)[:1]:
            assert graph.has_path(node, node) is True

    def test_coupling_metrics(self, repo_path: Path) -> None:
        """Calculate coupling metrics for the project."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        analyzer.analyze_directory(repo_path / "src")
        metrics = analyzer.calculate_metrics()

        assert metrics.total_modules >= 5
        assert metrics.total_dependencies >= 3
        assert metrics.average_dependencies_per_module >= 0
        assert isinstance(metrics.coupling_metrics, dict)
        assert isinstance(metrics.dead_modules, list)

    def test_get_dependents(self, repo_path: Path) -> None:
        """Get modules that depend on a given module."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        graph = analyzer.analyze_directory(repo_path / "src")

        # Find a target that has dependents
        for node in graph.nodes:
            dependents = graph.get_dependents(node)
            if dependents:
                assert isinstance(dependents, list)
                break

    def test_analyze_single_file(self, repo_path: Path) -> None:
        """Analyze imports from a single Python file."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(repo_path / "src" / "services" / "auth.py")
        assert len(imports) >= 1, "auth.py should have at least 1 import"
        # All imports should be Python
        for imp in imports:
            assert imp.language == Language.PYTHON

    def test_analyze_javascript_imports(self, repo_path: Path) -> None:
        """Analyze JavaScript import/export statements."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(repo_path / "web" / "app.js")
        # app.js doesn't have import statements from external modules via 'import from'
        # but we validate the analyzer doesn't crash
        assert isinstance(imports, list)

    def test_analyze_typescript_imports(self, repo_path: Path) -> None:
        """Analyze TypeScript import statements."""
        from pysearch.analysis.dependency_analysis import DependencyAnalyzer

        analyzer = DependencyAnalyzer()
        imports = analyzer.analyze_file(repo_path / "web" / "types.ts")
        assert isinstance(imports, list)


# ===========================================================================
# 17. Multi-Repository Search
# ===========================================================================


class TestMultiRepoSearch:
    """Verify multi-repository search across the sample_repo."""

    def test_add_and_list_repositories(self, repo_path: Path) -> None:
        """Add a repo to multi-repo engine and list it."""
        from pysearch.integrations.multi_repo import MultiRepoSearchEngine

        engine = MultiRepoSearchEngine()
        added = engine.add_repository("sample", repo_path)
        assert added is True

        names = engine.repository_manager.list_repositories()
        assert "sample" in names

    def test_remove_repository(self, repo_path: Path) -> None:
        """Remove a repository from the engine."""
        from pysearch.integrations.multi_repo import MultiRepoSearchEngine

        engine = MultiRepoSearchEngine()
        engine.add_repository("to_remove", repo_path)
        removed = engine.remove_repository("to_remove")
        assert removed is True
        removed_again = engine.remove_repository("to_remove")
        assert removed_again is False

    def test_add_multiple_repos(self, repo_path: Path, tmp_path: Path) -> None:
        """Add multiple repos and verify listing."""
        from pysearch.integrations.multi_repo import MultiRepoSearchEngine

        engine = MultiRepoSearchEngine()
        engine.add_repository("repo_a", repo_path)
        # Create a minimal second repo
        second = tmp_path / "repo_b"
        second.mkdir()
        (second / "hello.py").write_text("def hello(): pass\n")
        engine.add_repository("repo_b", second)

        names = engine.repository_manager.list_repositories()
        assert "repo_a" in names
        assert "repo_b" in names

    def test_add_repo_with_priority(self, repo_path: Path) -> None:
        """Add repository with priority setting."""
        from pysearch.integrations.multi_repo import MultiRepoSearchEngine

        engine = MultiRepoSearchEngine()
        added = engine.add_repository("high_priority", repo_path, priority="high")
        assert added is True


# ===========================================================================
# 18. Indexer Integration
# ===========================================================================


class TestIndexerIntegration:
    """Verify indexer behaviour through the PySearch engine."""

    def test_indexer_scans_files(self, py_engine: PySearch) -> None:
        """Indexer should discover files in the sample repo."""
        paths = list(py_engine.indexer.iter_all_paths())
        assert len(paths) >= 10, f"Expected at least 10 indexed files, got {len(paths)}"

    def test_indexer_count(self, py_engine: PySearch) -> None:
        """Count indexed files."""
        count = py_engine.indexer.count_indexed()
        assert count >= 10

    def test_search_uses_indexed_paths(self, py_engine: PySearch) -> None:
        """Search stats should report correct file counts."""
        result = py_engine.search("def", output=OutputFormat.TEXT)
        assert result.stats.files_scanned >= 10
        assert result.stats.indexed_files >= 10

    def test_search_result_file_paths_are_absolute(self, py_engine: PySearch) -> None:
        """All result file paths should be absolute Path objects."""
        result = py_engine.search("class", output=OutputFormat.TEXT)
        for item in result.items:
            assert item.file.is_absolute(), f"Expected absolute path, got: {item.file}"

    def test_search_result_line_numbers_valid(self, py_engine: PySearch) -> None:
        """Line numbers in results should be positive integers."""
        result = py_engine.search("import", output=OutputFormat.TEXT)
        for item in result.items:
            assert item.start_line >= 1
            assert item.end_line >= item.start_line
            assert len(item.lines) > 0


# ===========================================================================
# 19. Error Handling
# ===========================================================================


class TestErrorHandling:
    """Verify error types and handling behaviour."""

    def test_error_types_importable(self) -> None:
        """All custom error types should be importable."""
        from pysearch import EncodingError, FileAccessError, ParsingError, SearchError

        assert issubclass(SearchError, Exception)
        assert issubclass(FileAccessError, Exception)
        assert issubclass(EncodingError, Exception)
        assert issubclass(ParsingError, Exception)

    def test_error_collector(self, py_engine: PySearch) -> None:
        """Error collector should be accessible and clearable."""
        py_engine.error_collector.clear()
        assert len(py_engine.error_collector.errors) == 0

    def test_search_non_existent_path(self) -> None:
        """Searching a non-existent path should not raise."""
        cfg = SearchConfig(
            paths=["/nonexistent/path/xyz_42"],
            include=["**/*.py"],
            context=0,
            parallel=False,
        )
        engine = PySearch(cfg)
        result = engine.search("def", output=OutputFormat.TEXT)
        assert isinstance(result, SearchResult)
        assert len(result.items) == 0


# ===========================================================================
# 20. Match Spans & Highlighting
# ===========================================================================


class TestMatchSpans:
    """Verify match span precision on real code."""

    def test_match_spans_present(self, py_engine: PySearch) -> None:
        """Regex search should produce match_spans."""
        q = Query(
            pattern=r"class\s+\w+",
            use_regex=True,
            context=0,
            output=OutputFormat.TEXT,
        )
        result = py_engine.run(q)
        assert result.items
        # At least some items should have match_spans
        items_with_spans = [it for it in result.items if it.match_spans]
        assert len(items_with_spans) >= 1, "Expected at least 1 item with match_spans"

    def test_match_spans_columns_valid(self, py_engine: PySearch) -> None:
        """Match span column values should be non-negative."""
        q = Query(
            pattern=r"def\s+\w+",
            use_regex=True,
            context=0,
            output=OutputFormat.TEXT,
        )
        result = py_engine.run(q)
        for item in result.items:
            for line_idx, (start_col, end_col) in item.match_spans:
                assert line_idx >= 0
                assert start_col >= 0
                assert end_col >= start_col


# ===========================================================================
# 21. Language Detection Extended
# ===========================================================================


class TestLanguageDetectionExtended:
    """Extended language detection tests."""

    def test_detect_css(self, repo_path: Path) -> None:
        assert detect_language(repo_path / "web" / "styles.css") == Language.CSS

    def test_detect_toml(self, repo_path: Path) -> None:
        assert detect_language(repo_path / "pyproject.toml") == Language.TOML

    def test_detect_markdown(self, repo_path: Path) -> None:
        assert detect_language(repo_path / "README.md") == Language.MARKDOWN

    def test_detect_unknown_extension(self, tmp_path: Path) -> None:
        """Unknown file extension should return UNKNOWN."""
        f = tmp_path / "data.xyz123"
        f.write_text("some content")
        assert detect_language(f) == Language.UNKNOWN

    def test_get_supported_languages(self) -> None:
        """Utility should return a list of supported languages."""
        from pysearch import get_supported_languages

        langs = get_supported_languages()
        assert isinstance(langs, list | set)
        assert len(langs) >= 10
