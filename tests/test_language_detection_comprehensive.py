"""
Comprehensive tests for language detection edge cases and utility functions.

This module tests comprehensive language detection scenarios, utility functions,
and integration between different detection strategies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Set

import pytest

from pysearch.analysis.language_detection import (
    detect_language,
    detect_language_by_content,
    get_language_extensions,
    get_supported_languages,
    is_text_file,
    EXTENSION_MAP,
    FILENAME_PATTERNS,
)
from pysearch import Language


class TestComprehensiveLanguageDetection:
    """Test comprehensive language detection using multiple strategies."""

    def test_filename_priority_over_extension(self) -> None:
        """Test that filename patterns have priority over extensions."""
        # Dockerfile has no extension but should be detected by filename
        path = Path("Dockerfile")
        detected = detect_language(path)
        assert detected == Language.DOCKERFILE
        
        # Even with content, filename should take priority
        content = "FROM ubuntu:20.04\nRUN apt-get update"
        detected = detect_language(path, content)
        assert detected == Language.DOCKERFILE

    def test_extension_detection_with_content_verification(self) -> None:
        """Test extension detection with content verification."""
        path = Path("script.py")
        content = "#!/usr/bin/env python\nimport os\nprint('hello')"
        
        detected = detect_language(path, content)
        assert detected == Language.PYTHON

    def test_shebang_override_extension(self) -> None:
        """Test shebang overriding extension detection."""
        # File with .py extension but shell shebang
        path = Path("script.py")
        content = "#!/bin/bash\necho 'This is actually a shell script'"
        
        detected = detect_language(path, content)
        # Shebang should override extension
        assert detected == Language.SHELL

    def test_ambiguous_extension_resolution(self) -> None:
        """Test resolution of ambiguous file extensions."""
        # .h files could be C or C++
        path = Path("header.h")
        
        # C-style content
        c_content = "#include <stdio.h>\nint main() { printf('hello'); }"
        detected = detect_language(path, c_content)
        assert detected == Language.C
        
        # C++-style content
        cpp_content = "#include <iostream>\nint main() { std::cout << 'hello'; }"
        detected = detect_language(path, cpp_content)
        # Should still be C since .h maps to C by default, but content detection might help
        assert detected in [Language.C, Language.CPP]

    def test_matlab_vs_objective_c(self) -> None:
        """Test disambiguation between MATLAB and Objective-C (.m files)."""
        path = Path("script.m")
        
        # MATLAB content
        matlab_content = "function result = myFunction(x)\n    result = x * 2;\nend"
        detected = detect_language(path, matlab_content)
        assert detected == Language.MATLAB
        
        # Without content, should default to MATLAB (as per EXTENSION_MAP)
        detected = detect_language(path)
        assert detected == Language.MATLAB

    def test_fallback_to_content_detection(self) -> None:
        """Test fallback to content detection for unknown extensions."""
        path = Path("script.unknown")
        
        # Python content in unknown extension file
        content = "import sys\ndef main():\n    print('hello')\nif __name__ == '__main__':\n    main()"
        detected = detect_language(path, content)
        assert detected == Language.PYTHON
        
        # JavaScript content in unknown extension file
        content = "function main() {\n    console.log('hello');\n}\nmodule.exports = main;"
        detected = detect_language(path, content)
        assert detected == Language.JAVASCRIPT

    def test_no_detection_possible(self) -> None:
        """Test cases where no language can be detected."""
        # Unknown extension, no content
        path = Path("data.binary")
        detected = detect_language(path)
        assert detected == Language.UNKNOWN
        
        # Unknown extension, non-code content
        content = "This is just plain text with no programming patterns."
        detected = detect_language(path, content)
        assert detected == Language.UNKNOWN

    def test_empty_file_detection(self) -> None:
        """Test detection of empty files."""
        path = Path("empty.py")
        content = ""
        
        detected = detect_language(path, content)
        # Should detect by extension since content is empty
        assert detected == Language.PYTHON

    def test_whitespace_only_file(self) -> None:
        """Test detection of files with only whitespace."""
        path = Path("whitespace.js")
        content = "   \n\n\t\t\n   "
        
        detected = detect_language(path, content)
        # Should detect by extension
        assert detected == Language.JAVASCRIPT


class TestLanguageUtilityFunctions:
    """Test language detection utility functions."""

    def test_get_language_extensions(self) -> None:
        """Test getting extensions for specific languages."""
        # Test Python extensions
        py_extensions = get_language_extensions(Language.PYTHON)
        expected_py = [".py", ".pyw", ".pyi", ".pyx"]
        assert all(ext in py_extensions for ext in expected_py)
        
        # Test JavaScript extensions
        js_extensions = get_language_extensions(Language.JAVASCRIPT)
        expected_js = [".js", ".jsx", ".mjs"]
        assert all(ext in js_extensions for ext in expected_js)
        
        # Test TypeScript extensions
        ts_extensions = get_language_extensions(Language.TYPESCRIPT)
        expected_ts = [".ts", ".tsx", ".d.ts"]
        assert all(ext in ts_extensions for ext in expected_ts)

    def test_get_language_extensions_unknown(self) -> None:
        """Test getting extensions for unknown language."""
        extensions = get_language_extensions(Language.UNKNOWN)
        assert extensions == []

    def test_get_supported_languages(self) -> None:
        """Test getting list of supported languages."""
        languages = get_supported_languages()
        
        # Should include all Language enum values
        assert Language.PYTHON in languages
        assert Language.JAVASCRIPT in languages
        assert Language.TYPESCRIPT in languages
        assert Language.JAVA in languages
        assert Language.GO in languages
        assert Language.RUST in languages
        assert Language.UNKNOWN in languages
        
        # Should be a list of Language enum values
        assert all(isinstance(lang, Language) for lang in languages)

    def test_is_text_file_common_extensions(self) -> None:
        """Test text file detection for common extensions."""
        text_files = [
            "script.py",
            "app.js",
            "component.tsx",
            "Main.java",
            "main.go",
            "lib.rs",
            "index.php",
            "script.rb",
            "style.css",
            "page.html",
            "data.json",
            "config.yaml",
            "settings.toml",
            "README.md",
            "query.sql",
            "script.sh",
            "config.ps1",
        ]
        
        for filename in text_files:
            path = Path(filename)
            assert is_text_file(path), f"{filename} should be detected as text file"

    def test_is_text_file_special_filenames(self) -> None:
        """Test text file detection for special filenames."""
        special_files = [
            "Dockerfile",
            "Makefile",
            "Rakefile",
            "Gemfile",
            "Vagrantfile",
        ]
        
        for filename in special_files:
            path = Path(filename)
            assert is_text_file(path), f"{filename} should be detected as text file"

    def test_is_text_file_non_text_extensions(self) -> None:
        """Test text file detection for non-text extensions."""
        non_text_files = [
            "image.jpg",
            "video.mp4",
            "archive.zip",
            "binary.exe",
            "library.so",
            "data.bin",
            "font.ttf",
        ]
        
        for filename in non_text_files:
            path = Path(filename)
            assert not is_text_file(path), f"{filename} should not be detected as text file"

    def test_is_text_file_no_extension(self) -> None:
        """Test text file detection for files without extensions."""
        # Files without extensions that are in FILENAME_PATTERNS should be text
        special_files = ["Dockerfile", "Makefile", "Rakefile"]
        for filename in special_files:
            path = Path(filename)
            assert is_text_file(path)
        
        # Files without extensions not in patterns should not be text
        unknown_files = ["binary", "data", "unknown"]
        for filename in unknown_files:
            path = Path(filename)
            assert not is_text_file(path)


class TestLanguageDetectionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_long_filename(self) -> None:
        """Test detection with very long filename."""
        long_name = "a" * 200 + ".py"
        path = Path(long_name)
        detected = detect_language(path)
        assert detected == Language.PYTHON

    def test_filename_with_special_characters(self) -> None:
        """Test detection with special characters in filename."""
        special_names = [
            "test-file.py",
            "test_file.js",
            "test.file.ts",
            "test@file.java",
            "test file.go",  # Space in filename
            "тест.py",       # Unicode characters
        ]
        
        for filename in special_names:
            path = Path(filename)
            detected = detect_language(path)
            # Should detect based on extension regardless of special chars
            if filename.endswith(".py"):
                assert detected == Language.PYTHON
            elif filename.endswith(".js"):
                assert detected == Language.JAVASCRIPT

    def test_case_sensitivity_in_filenames(self) -> None:
        """Test case sensitivity in filename detection."""
        # Dockerfile variants
        assert detect_language(Path("Dockerfile")) == Language.DOCKERFILE
        assert detect_language(Path("dockerfile")) == Language.DOCKERFILE
        assert detect_language(Path("DOCKERFILE")) == Language.UNKNOWN  # Not in patterns
        
        # Makefile variants
        assert detect_language(Path("Makefile")) == Language.MAKEFILE
        assert detect_language(Path("makefile")) == Language.MAKEFILE
        assert detect_language(Path("MAKEFILE")) == Language.UNKNOWN  # Not in patterns

    def test_multiple_dots_in_filename(self) -> None:
        """Test files with multiple dots."""
        test_cases = [
            ("config.dev.json", Language.JSON),
            ("test.spec.js", Language.JAVASCRIPT),
            ("types.d.ts", Language.TYPESCRIPT),
            ("backup.old.py", Language.PYTHON),  # .py is the final extension
            ("archive.tar.gz", Language.UNKNOWN),
        ]
        
        for filename, expected in test_cases:
            path = Path(filename)
            detected = detect_language(path)
            assert detected == expected, f"Failed for {filename}: expected {expected}, got {detected}"

    def test_content_with_mixed_patterns(self) -> None:
        """Test content that contains patterns from multiple languages."""
        mixed_content = """
        // This looks like JavaScript
        function test() {
            return true;
        }
        
        # But this looks like Python
        def another_test():
            return False
            
        // And this could be Java
        public class Test {
            public static void main(String[] args) {
            }
        }
        """
        
        # Test with different candidate sets
        js_candidates = {Language.JAVASCRIPT}
        detected = detect_language_by_content(mixed_content, js_candidates)
        assert detected == Language.JAVASCRIPT
        
        py_candidates = {Language.PYTHON}
        detected = detect_language_by_content(mixed_content, py_candidates)
        assert detected == Language.PYTHON
        
        java_candidates = {Language.JAVA}
        detected = detect_language_by_content(mixed_content, java_candidates)
        assert detected == Language.JAVA
        
        # With all candidates, should pick the one with highest score
        all_candidates = {Language.JAVASCRIPT, Language.PYTHON, Language.JAVA}
        detected = detect_language_by_content(mixed_content, all_candidates)
        assert detected in all_candidates
