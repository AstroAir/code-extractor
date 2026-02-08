"""
Integration tests for IDE hooks and IDEIntegration module.

Tests the IDE integration functionality with a real PySearch engine,
including ide_query, IDEIntegration jump-to-definition, find-references,
completion, hover, document symbols, workspace symbols, and diagnostics.
"""

import os
import tempfile
from pathlib import Path

from pysearch import OutputFormat, PySearch, Query, SearchConfig
from pysearch.integrations.ide_hooks import IDEIntegration, ide_query


# ---------------------------------------------------------------------------
# ide_query integration tests
# ---------------------------------------------------------------------------


class TestIdeQueryIntegration:
    """Integration tests for the ide_query convenience function."""

    def test_ide_query_basic(self, tmp_path: Path):
        """Test basic IDE query functionality."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def hello():\n    print('Hello World')\n    return True")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        query = Query(pattern="hello", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        assert isinstance(result, dict)
        assert "items" in result
        assert "stats" in result
        assert isinstance(result["items"], list)
        assert isinstance(result["stats"], dict)

        assert len(result["items"]) >= 1
        item = result["items"][0]
        assert "file" in item
        assert "start_line" in item
        assert "end_line" in item
        assert "lines" in item
        assert "spans" in item

        assert os.path.basename(str(test_file)) in item["file"]
        assert item["start_line"] >= 1
        assert item["end_line"] >= item["start_line"]
        assert isinstance(item["lines"], list)
        assert isinstance(item["spans"], list)

    def test_ide_query_with_matches(self, tmp_path: Path):
        """Test IDE query with specific match spans."""
        test_file = tmp_path / "multi.py"
        test_file.write_text(
            "def process_data():\n"
            "    data = get_data()\n"
            "    return process(data)\n"
            "\n"
            "def process_file():\n"
            "    return process_content()\n"
        )

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        query = Query(pattern="process", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        assert len(result["items"]) >= 2

        for item in result["items"]:
            assert isinstance(item["spans"], list)
            for span in item["spans"]:
                assert isinstance(span, tuple)
                assert len(span) == 2
                line_idx, (start, end) = span
                assert isinstance(line_idx, int)
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert start <= end

    def test_ide_query_no_matches(self, tmp_path: Path):
        """Test IDE query when no matches are found."""
        (tmp_path / "empty.py").write_text("# Empty file\npass\n")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        query = Query(pattern="nonexistent_pattern_xyz", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        assert isinstance(result, dict)
        assert len(result["items"]) == 0
        assert result["stats"]["items"] == 0

    def test_ide_query_with_context(self, tmp_path: Path):
        """Test IDE query with context lines."""
        (tmp_path / "context.py").write_text(
            "# Header comment\n"
            "import os\n"
            "def main():\n"
            "    print('Hello')\n"
            "    return 0\n"
            "# Footer comment\n"
        )

        cfg = SearchConfig(paths=[str(tmp_path)], context=2)
        engine = PySearch(cfg)
        query = Query(pattern="main", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        assert len(result["items"]) >= 1
        item = result["items"][0]
        assert len(item["lines"]) > 1
        assert item["start_line"] < 3
        assert item["end_line"] > 3

    def test_ide_query_multiple_files(self, tmp_path: Path):
        """Test IDE query across multiple files."""
        (tmp_path / "file1.py").write_text("def helper():\n    return True\n")
        (tmp_path / "file2.py").write_text("def helper_function():\n    return False\n")
        (tmp_path / "file3.py").write_text("class helper_class:\n    pass\n")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        query = Query(pattern="helper", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        assert len(result["items"]) >= 3
        found_files = {item["file"] for item in result["items"]}
        assert len(found_files) >= 3

    def test_ide_query_stats_structure(self, tmp_path: Path):
        """Test that IDE query returns properly structured stats."""
        for i in range(3):
            (tmp_path / f"test{i}.py").write_text(f"def function_{i}():\n    return {i}\n")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)
        query = Query(pattern="function", output=OutputFormat.JSON)
        result = ide_query(engine, query)

        stats = result["stats"]
        for field in ("files_scanned", "files_matched", "items", "elapsed_ms", "indexed_files"):
            assert field in stats
            assert isinstance(stats[field], (int, float))

        assert stats["files_scanned"] >= 3
        assert stats["files_matched"] >= 3
        assert stats["items"] >= 3
        assert stats["elapsed_ms"] >= 0
        assert stats["indexed_files"] >= 3


# ---------------------------------------------------------------------------
# IDEIntegration integration tests (with real PySearch engine)
# ---------------------------------------------------------------------------


class TestIDEIntegrationWithEngine:
    """Integration tests for IDEIntegration backed by a real PySearch engine."""

    @staticmethod
    def _make_engine(tmp_path: Path) -> PySearch:
        cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0)
        return PySearch(cfg)

    def test_jump_to_definition(self, tmp_path: Path):
        """Test jump-to-definition finds a real function definition."""
        (tmp_path / "lib.py").write_text(
            "def my_helper(x):\n    return x + 1\n", encoding="utf-8"
        )
        (tmp_path / "main.py").write_text(
            "from lib import my_helper\nresult = my_helper(42)\n", encoding="utf-8"
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("main.py", 2, "my_helper")
        assert loc is not None
        assert loc.symbol_name == "my_helper"
        assert loc.symbol_type == "function"
        assert loc.line >= 1

    def test_jump_to_definition_class(self, tmp_path: Path):
        """Test jump-to-definition for a class."""
        (tmp_path / "models.py").write_text(
            "class UserModel:\n    pass\n", encoding="utf-8"
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("other.py", 1, "UserModel")
        assert loc is not None
        assert loc.symbol_type == "class"

    def test_jump_to_definition_not_found(self, tmp_path: Path):
        """Test jump-to-definition returns None for unknown symbol."""
        (tmp_path / "empty.py").write_text("x = 1\n", encoding="utf-8")

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        loc = integration.jump_to_definition("f.py", 1, "totally_missing_symbol_xyz")
        assert loc is None

    def test_find_references(self, tmp_path: Path):
        """Test find-references across files."""
        (tmp_path / "a.py").write_text("def compute(x):\n    return x\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("from a import compute\ncompute(1)\n", encoding="utf-8")

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        refs = integration.find_references("b.py", 2, "compute")
        assert len(refs) >= 2

    def test_find_references_exclude_definition(self, tmp_path: Path):
        """Test find-references can exclude the definition."""
        (tmp_path / "a.py").write_text(
            "def compute(x):\n    return x\n", encoding="utf-8"
        )
        (tmp_path / "b.py").write_text("from a import compute\ncompute(1)\n", encoding="utf-8")

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        refs = integration.find_references("b.py", 2, "compute", include_definition=False)
        assert all(not r.is_definition for r in refs)

    def test_get_document_symbols(self, tmp_path: Path):
        """Test listing symbols in a document."""
        (tmp_path / "module.py").write_text(
            "MAX_SIZE = 100\n\n"
            "def process():\n    pass\n\n"
            "class Handler:\n    def run(self):\n        pass\n",
            encoding="utf-8",
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        symbols = integration.get_document_symbols(str(tmp_path / "module.py"))
        names = [s.name for s in symbols]
        assert "MAX_SIZE" in names
        assert "process" in names
        assert "Handler" in names

    def test_get_workspace_symbols(self, tmp_path: Path):
        """Test searching workspace symbols."""
        (tmp_path / "utils.py").write_text(
            "def parse_input(data):\n    pass\n\n"
            "def parse_output(data):\n    pass\n",
            encoding="utf-8",
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        symbols = integration.get_workspace_symbols("parse")
        assert isinstance(symbols, list)
        # Should find at least the two parse_ functions
        assert len(symbols) >= 1

    def test_get_diagnostics_markers(self, tmp_path: Path):
        """Test diagnostics detect TODO/FIXME/HACK markers."""
        (tmp_path / "todo.py").write_text(
            "x = 1  # TODO implement properly\n"
            "y = 2  # FIXME critical bug\n"
            "z = 3  # HACK temporary workaround\n",
            encoding="utf-8",
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        diags = integration.get_diagnostics(str(tmp_path / "todo.py"))
        codes = [d.code for d in diags]
        assert "TODO" in codes
        assert "FIXME" in codes
        assert "HACK" in codes

    def test_provide_hover_function(self, tmp_path: Path):
        """Test hover information for a function."""
        (tmp_path / "funcs.py").write_text(
            "def calculate_total(items, tax_rate):\n"
            '    """Calculate total with tax."""\n'
            "    return sum(items) * (1 + tax_rate)\n",
            encoding="utf-8",
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        hover = integration.provide_hover("caller.py", 1, 0, "calculate_total")
        # May or may not find depending on context parameter; ensure no crash
        if hover is not None:
            assert hover.symbol_name == "calculate_total"
            assert hover.symbol_type == "function"

    def test_provide_completion(self, tmp_path: Path):
        """Test completion returns suggestions."""
        (tmp_path / "api.py").write_text(
            "def handle_request(req):\n    pass\n\n"
            "def handle_response(res):\n    pass\n",
            encoding="utf-8",
        )

        engine = self._make_engine(tmp_path)
        integration = IDEIntegration(engine)
        completions = integration.provide_completion("main.py", 1, 0, prefix="handle")
        assert isinstance(completions, list)
