"""
Shared test fixtures and utilities for PySearch tests.

This module provides common fixtures, test data, and helper functions
to reduce code duplication and improve test consistency across the test suite.
"""

import tempfile
from pathlib import Path

import pytest

from pysearch import OutputFormat, PySearch, Query, SearchConfig

# Test data constants
SAMPLE_PYTHON_CODE = """
def foo():
    '''A simple function'''
    pass

class Bar:
    '''A simple class'''
    def baz(self):
        return "ok"
    
    def method_with_args(self, x: int, y: str = "default"):
        '''Method with typed arguments'''
        return f"{x}: {y}"

async def async_function():
    '''An async function'''
    await some_async_call()
    return True

@decorator
def decorated_function():
    '''A decorated function'''
    return "decorated"
"""

SAMPLE_JAVASCRIPT_CODE = """
function hello() {
    console.log("Hello World");
}

class MyClass {
    constructor(name) {
        this.name = name;
    }
    
    greet() {
        return `Hello, ${this.name}!`;
    }
}

const arrow = () => {
    return "arrow function";
};
"""

SAMPLE_CONFIG_FILE = """
# Configuration file
CONFIG = {
    'debug': True,
    'version': '1.0.0',
    'features': ['search', 'index', 'cache']
}

DATABASE_URL = "sqlite:///test.db"
"""


@pytest.fixture
def temp_project_dir():
    """Create a temporary directory for test projects."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_python_project(temp_project_dir):
    """Create a sample Python project with multiple files."""
    project_dir = temp_project_dir / "sample_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Main module
    (project_dir / "main.py").write_text(
        "def main():\n" "    '''Main function'''\n" "    print('Hello World')\n" "    return 0\n"
    )

    # Utils module
    (project_dir / "utils.py").write_text(SAMPLE_PYTHON_CODE)

    # Config module
    (project_dir / "config.py").write_text(SAMPLE_CONFIG_FILE)

    # Subpackage
    subpkg = project_dir / "subpackage"
    subpkg.mkdir()
    (subpkg / "__init__.py").write_text("")
    (subpkg / "module.py").write_text(
        "class SubModule:\n" "    def process(self):\n" "        return 'processed'\n"
    )

    return project_dir


@pytest.fixture
def mixed_language_project(temp_project_dir):
    """Create a project with multiple programming languages."""
    project_dir = temp_project_dir / "mixed_project"
    project_dir.mkdir(parents=True, exist_ok=True)

    # Python files
    (project_dir / "app.py").write_text(SAMPLE_PYTHON_CODE)

    # JavaScript files
    (project_dir / "script.js").write_text(SAMPLE_JAVASCRIPT_CODE)

    # Other files
    (project_dir / "README.md").write_text("# Test Project\n\nThis is a test project.")
    (project_dir / "requirements.txt").write_text("pytest\nblack\nmypy\n")

    return project_dir


@pytest.fixture
def basic_search_config(sample_python_project):
    """Create a basic SearchConfig for testing."""
    return SearchConfig(
        paths=[str(sample_python_project)],
        include=["**/*.py"],
        exclude=[],
        context=1,
        parallel=False,  # Disable parallel processing for deterministic tests
    )


@pytest.fixture
def pysearch_engine(basic_search_config):
    """Create a PySearch engine instance for testing."""
    return PySearch(basic_search_config)


@pytest.fixture
def comprehensive_search_config(mixed_language_project):
    """Create a comprehensive SearchConfig for multi-language testing."""
    return SearchConfig(
        paths=[str(mixed_language_project)],
        include=["**/*.py", "**/*.js", "**/*.md"],
        exclude=["**/node_modules/**", "**/__pycache__/**"],
        context=2,
        parallel=False,
    )


class TestDataHelper:
    """Helper class for creating test data and assertions."""

    @staticmethod
    def create_file_with_content(path: Path, content: str, encoding: str = "utf-8"):
        """Create a file with specified content."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)

    @staticmethod
    def assert_search_results(results, expected_count: int = None, contains_text: str = None):
        """Assert search results meet expectations."""
        if expected_count is not None:
            assert (
                len(results.items) >= expected_count
            ), f"Expected at least {expected_count} results"

        if contains_text:
            found = any(contains_text in "".join(item.lines) for item in results.items)
            assert found, f"Expected to find '{contains_text}' in search results"

    @staticmethod
    def create_query(pattern: str, **kwargs) -> Query:
        """Create a Query object with common defaults."""
        defaults = {"use_regex": False, "context": 1, "output_format": OutputFormat.TEXT}
        defaults.update(kwargs)
        return Query(pattern=pattern, **defaults)


@pytest.fixture
def test_helper():
    """Provide the TestDataHelper for tests."""
    return TestDataHelper()


# Markers for different test categories
pytest_plugins = []


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow tests that may take longer to run")
    config.addinivalue_line("markers", "benchmark: Performance benchmark tests")
    config.addinivalue_line("markers", "api: API-related tests")
    config.addinivalue_line("markers", "cli: CLI-related tests")
    config.addinivalue_line("markers", "indexer: Indexer-related tests")
    config.addinivalue_line("markers", "matcher: Matcher-related tests")
    config.addinivalue_line("markers", "cache: Cache-related tests")


# Common test data patterns
TEST_PATTERNS = {
    "function_def": r"def\s+\w+\s*\(",
    "class_def": r"class\s+\w+\s*[:\(]",
    "import_stmt": r"(?:from\s+\w+\s+)?import\s+\w+",
    "async_def": r"async\s+def\s+\w+\s*\(",
    "decorator": r"@\w+",
}

# File extensions for different languages
LANGUAGE_EXTENSIONS = {
    "python": [".py", ".pyx", ".pyi"],
    "javascript": [".js", ".jsx", ".ts", ".tsx"],
    "markdown": [".md", ".markdown"],
    "json": [".json"],
    "yaml": [".yml", ".yaml"],
    "text": [".txt"],
}
