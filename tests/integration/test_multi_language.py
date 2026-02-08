import tempfile
from pathlib import Path

import pytest

from pysearch import Language, OutputFormat, PySearch, SearchConfig
from pysearch.analysis.language_detection import detect_language


def test_language_detection():
    """Test basic language detection functionality."""
    # Test Python detection
    assert detect_language(Path("test.py")) == Language.PYTHON
    assert detect_language(Path("script.py"), "def main():\n    pass") == Language.PYTHON

    # Test JavaScript detection
    assert detect_language(Path("app.js")) == Language.JAVASCRIPT
    assert (
        detect_language(Path("script.js"), "function main() { return 42; }") == Language.JAVASCRIPT
    )

    # Test TypeScript detection
    assert detect_language(Path("app.ts")) == Language.TYPESCRIPT
    assert (
        detect_language(Path("types.ts"), "interface User { name: string; }") == Language.TYPESCRIPT
    )

    # Test special filenames
    assert detect_language(Path("Dockerfile")) == Language.DOCKERFILE
    assert detect_language(Path("Makefile")) == Language.MAKEFILE

    # Test shebang detection
    assert (
        detect_language(Path("script"), "#!/usr/bin/env python\nprint('hello')") == Language.PYTHON
    )
    assert detect_language(Path("script"), "#!/bin/bash\necho hello") == Language.SHELL


def test_multi_language_search():
    """Test searching across multiple programming languages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files in different languages
        files = {
            "main.py": "def hello():\n    print('Hello from Python')",
            "app.js": "function hello() {\n    console.log('Hello from JavaScript');\n}",
            "types.ts": "interface Greeting {\n    message: string;\n}\nfunction hello(): Greeting {\n    return { message: 'Hello from TypeScript' };\n}",
            "Main.java": 'public class Main {\n    public static void hello() {\n        System.out.println("Hello from Java");\n    }\n}',
            "main.go": 'package main\n\nimport "fmt"\n\nfunc hello() {\n    fmt.Println("Hello from Go")\n}',
            "README.md": "# Test Project\n\nThis is a test project with multiple languages.",
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content, encoding="utf-8")

        # Test searching all languages
        cfg = SearchConfig(
            paths=[str(tmp_path)],
            languages=None,  # All languages
        )
        engine = PySearch(cfg)

        # Search for "hello" function across all files
        result = engine.search("hello", regex=False, output=OutputFormat.JSON)

        # Should find matches in all code files
        assert len(result.items) >= 4  # At least Python, JS, TS, Java, Go

        # Verify we found matches in different languages
        found_files = {item.file.name for item in result.items}
        assert "main.py" in found_files
        assert "app.js" in found_files
        assert "types.ts" in found_files
        assert "Main.java" in found_files
        assert "main.go" in found_files


def test_language_specific_search():
    """Test searching only specific languages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create test files
        files = {
            "script.py": "def process_data():\n    return 'python'",
            "script.js": "function processData() {\n    return 'javascript';\n}",
            "script.go": 'func processData() string {\n    return "go"\n}',
        }

        for filename, content in files.items():
            file_path = tmp_path / filename
            file_path.write_text(content, encoding="utf-8")

        # Test Python-only search
        cfg = SearchConfig(
            paths=[str(tmp_path)],
            languages={Language.PYTHON},
        )
        engine = PySearch(cfg)

        result = engine.search("process", regex=False, output=OutputFormat.JSON)

        # Should only find Python file
        assert len(result.items) == 1
        assert result.items[0].file.name == "script.py"

        # Test JavaScript + Go search
        cfg.languages = {Language.JAVASCRIPT, Language.GO}
        engine = PySearch(cfg)

        result = engine.search("process", regex=False, output=OutputFormat.JSON)

        # Should find JS and Go files
        assert len(result.items) == 2
        found_files = {item.file.name for item in result.items}
        assert "script.js" in found_files
        assert "script.go" in found_files
        assert "script.py" not in found_files


def test_auto_include_patterns():
    """Test automatic include pattern generation."""
    # Test all languages
    cfg = SearchConfig(languages=None)
    patterns = cfg.get_include_patterns()

    # Should include common file types
    assert "**/*.py" in patterns
    assert "**/*.js" in patterns
    assert "**/*.java" in patterns
    assert "**/Dockerfile" in patterns

    # Test specific languages
    cfg = SearchConfig(languages={Language.PYTHON, Language.JAVASCRIPT})
    patterns = cfg.get_include_patterns()

    # Should include Python and JS patterns
    assert "**/*.py" in patterns
    assert "**/*.js" in patterns
    # Should not include Java patterns (not explicitly, but might be in special files)

    # Test single language
    cfg = SearchConfig(languages={Language.RUST})
    patterns = cfg.get_include_patterns()

    assert "**/*.rs" in patterns


if __name__ == "__main__":
    pytest.main([__file__])
