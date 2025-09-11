import tempfile
import time
from pathlib import Path

import pytest

from pysearch import PySearch
from pysearch import SearchConfig
from pysearch.utils.metadata_filters import (
    create_date_filter,
    create_metadata_filters,
    create_size_filter,
)
from pysearch import Language, OutputFormat, Query


def test_size_filter_parsing():
    """Test size filter parsing."""
    # Test basic bytes
    min_bytes, max_bytes = create_size_filter("100", "1000")
    assert min_bytes == 100
    assert max_bytes == 1000

    # Test KB/MB/GB
    min_bytes, max_bytes = create_size_filter("1KB", "5MB")
    assert min_bytes == 1024
    assert max_bytes == 5 * 1024 * 1024

    # Test with spaces and case variations
    min_bytes, max_bytes = create_size_filter("2 MB", "1 GB")
    assert min_bytes == 2 * 1024 * 1024
    assert max_bytes == 1024 * 1024 * 1024

    # Test None values
    min_bytes, max_bytes = create_size_filter(None, "1MB")
    assert min_bytes is None
    assert max_bytes == 1024 * 1024


def test_date_filter_parsing():
    """Test date filter parsing."""
    # Test relative dates
    mod_after, mod_before, create_after, create_before = create_date_filter(
        modified_after="1d", modified_before="1w"
    )

    now = time.time()
    assert mod_after is not None and mod_after < now
    assert mod_before is not None and mod_before < now
    assert mod_after > mod_before  # 1 day ago is more recent than 1 week ago

    # Test ISO dates
    mod_after, mod_before, create_after, create_before = create_date_filter(
        modified_after="2023-01-01", modified_before="2023-12-31"
    )

    assert mod_after is not None
    assert mod_before is not None
    assert mod_after < mod_before


def test_metadata_filters_creation():
    """Test metadata filters creation."""
    filters = create_metadata_filters(
        min_size="1KB",
        max_size="1MB",
        modified_after="1d",
        min_lines=10,
        max_lines=1000,
        author_pattern="john.*",
        languages={Language.PYTHON, Language.JAVASCRIPT},
    )

    assert filters.min_size == 1024
    assert filters.max_size == 1024 * 1024
    assert filters.modified_after is not None
    assert filters.min_lines == 10
    assert filters.max_lines == 1000
    assert filters.author_pattern == "john.*"
    assert filters.languages == {Language.PYTHON, Language.JAVASCRIPT}


def test_size_based_filtering():
    """Test filtering files by size."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create files of different sizes
        small_file = tmp_path / "small.py"
        small_file.write_text("# Small file\nprint('hello')")  # ~30 bytes

        large_file = tmp_path / "large.py"
        large_content = "# Large file\n" + "print('hello world')\n" * 100  # ~1800 bytes
        large_file.write_text(large_content)

        # Test minimum size filter
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Search with minimum size filter (should only find large file)
        metadata_filters = create_metadata_filters(min_size="100")
        query = Query(pattern="hello", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find the large file
        assert len(result.items) == 1
        assert result.items[0].file.name == "large.py"

        # Test maximum size filter (should only find small file)
        metadata_filters = create_metadata_filters(max_size="100")
        query = Query(pattern="hello", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find the small file
        assert len(result.items) == 1
        assert result.items[0].file.name == "small.py"


def test_line_count_filtering():
    """Test filtering files by line count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create files with different line counts
        short_file = tmp_path / "short.py"
        short_file.write_text("print('hello')\nprint('world')")  # 2 lines

        long_file = tmp_path / "long.py"
        long_content = "\n".join([f"print('line {i}')" for i in range(50)])  # 50 lines
        long_file.write_text(long_content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test minimum lines filter
        metadata_filters = create_metadata_filters(min_lines=10)
        query = Query(pattern="print", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find the long file
        assert len(result.items) >= 1
        found_files = {item.file.name for item in result.items}
        assert "long.py" in found_files
        assert "short.py" not in found_files

        # Test maximum lines filter
        metadata_filters = create_metadata_filters(max_lines=10)
        query = Query(pattern="print", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find the short file
        assert len(result.items) >= 1
        found_files = {item.file.name for item in result.items}
        assert "short.py" in found_files
        assert "long.py" not in found_files


def test_language_filtering():
    """Test filtering files by programming language."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create files in different languages
        python_file = tmp_path / "script.py"
        python_file.write_text("def hello():\n    print('Hello from Python')")

        js_file = tmp_path / "script.js"
        js_file.write_text("function hello() {\n    console.log('Hello from JS');\n}")

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test Python-only filter
        metadata_filters = create_metadata_filters(languages={Language.PYTHON})
        query = Query(pattern="hello", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find Python file
        assert len(result.items) == 1
        assert result.items[0].file.name == "script.py"

        # Test JavaScript-only filter
        metadata_filters = create_metadata_filters(languages={Language.JAVASCRIPT})
        query = Query(pattern="hello", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find JavaScript file
        assert len(result.items) == 1
        assert result.items[0].file.name == "script.js"


def test_combined_filters():
    """Test combining multiple metadata filters."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files
        files = {
            "small.py": "print('small')",  # Small Python file
            "large.py": "\n".join([f"print('line {i}')" for i in range(100)]),  # Large Python file
            "small.js": "console.log('small');",  # Small JS file
            "large.js": "\n".join(
                [f"console.log('line {i}');" for i in range(100)]
            ),  # Large JS file
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content)

        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = PySearch(cfg)

        # Test combined filters: Python files with more than 10 lines
        metadata_filters = create_metadata_filters(languages={Language.PYTHON}, min_lines=10)
        query = Query(pattern="print", metadata_filters=metadata_filters, output=OutputFormat.JSON)
        result = engine.run(query)

        # Should only find large.py
        assert len(result.items) >= 1
        found_files = {item.file.name for item in result.items}
        assert "large.py" in found_files
        assert "small.py" not in found_files
        assert "small.js" not in found_files
        assert "large.js" not in found_files


if __name__ == "__main__":
    pytest.main([__file__])
