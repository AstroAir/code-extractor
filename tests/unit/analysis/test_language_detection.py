"""Tests for pysearch.analysis.language_detection module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.analysis.language_detection import (
    CONTENT_PATTERNS,
    EXTENSION_MAP,
    FILENAME_PATTERNS,
    SHEBANG_PATTERNS,
    detect_language,
    detect_language_by_content,
    detect_language_by_extension,
    detect_language_by_filename,
    detect_language_by_shebang,
    get_language_extensions,
    get_supported_languages,
    is_text_file,
)
from pysearch.core.types import Language


class TestDetectLanguageByExtension:
    """Tests for detect_language_by_extension."""

    def test_python(self):
        assert detect_language_by_extension(Path("main.py")) == Language.PYTHON

    def test_javascript(self):
        assert detect_language_by_extension(Path("app.js")) == Language.JAVASCRIPT

    def test_typescript(self):
        assert detect_language_by_extension(Path("app.ts")) == Language.TYPESCRIPT

    def test_java(self):
        assert detect_language_by_extension(Path("Main.java")) == Language.JAVA

    def test_go(self):
        assert detect_language_by_extension(Path("main.go")) == Language.GO

    def test_rust(self):
        assert detect_language_by_extension(Path("main.rs")) == Language.RUST

    def test_unknown(self):
        assert detect_language_by_extension(Path("file.xyz")) == Language.UNKNOWN

    def test_case_insensitive(self):
        assert detect_language_by_extension(Path("file.PY")) == Language.PYTHON


class TestDetectLanguageByFilename:
    """Tests for detect_language_by_filename."""

    def test_dockerfile(self):
        assert detect_language_by_filename(Path("Dockerfile")) == Language.DOCKERFILE

    def test_makefile(self):
        assert detect_language_by_filename(Path("Makefile")) == Language.MAKEFILE

    def test_rakefile(self):
        assert detect_language_by_filename(Path("Rakefile")) == Language.RUBY

    def test_unknown_filename(self):
        assert detect_language_by_filename(Path("random.txt")) == Language.UNKNOWN


class TestDetectLanguageByShebang:
    """Tests for detect_language_by_shebang."""

    def test_python_shebang(self):
        assert detect_language_by_shebang("#!/usr/bin/env python\nprint()") == Language.PYTHON

    def test_bash_shebang(self):
        assert detect_language_by_shebang("#!/bin/bash\necho hello") == Language.SHELL

    def test_node_shebang(self):
        assert detect_language_by_shebang("#!/usr/bin/env node\nconsole.log()") == Language.JAVASCRIPT

    def test_no_shebang(self):
        assert detect_language_by_shebang("just some text") == Language.UNKNOWN

    def test_empty_content(self):
        assert detect_language_by_shebang("") == Language.UNKNOWN


class TestDetectLanguageByContent:
    """Tests for detect_language_by_content."""

    def test_python_content(self):
        code = "import os\ndef main():\n    pass"
        result = detect_language_by_content(code, {Language.PYTHON, Language.JAVASCRIPT})
        assert result == Language.PYTHON

    def test_javascript_content(self):
        code = "const x = 1;\nfunction foo() {}\nmodule.exports = foo;"
        result = detect_language_by_content(code, {Language.PYTHON, Language.JAVASCRIPT})
        assert result == Language.JAVASCRIPT

    def test_empty_content(self):
        assert detect_language_by_content("", {Language.PYTHON}) == Language.UNKNOWN

    def test_empty_candidates(self):
        assert detect_language_by_content("import os", set()) == Language.UNKNOWN

    def test_no_match(self):
        result = detect_language_by_content("random text", {Language.GO})
        assert result == Language.UNKNOWN


class TestDetectLanguage:
    """Tests for detect_language (comprehensive)."""

    def test_by_extension(self):
        assert detect_language(Path("main.py")) == Language.PYTHON

    def test_by_filename(self):
        assert detect_language(Path("Dockerfile")) == Language.DOCKERFILE

    def test_with_content_shebang(self):
        result = detect_language(Path("script"), "#!/usr/bin/env python\nprint()")
        assert result == Language.PYTHON

    def test_unknown_no_content(self):
        assert detect_language(Path("file.xyz")) == Language.UNKNOWN

    def test_ambiguous_h_extension_with_content(self):
        c_code = "#include <stdio.h>\nint main() { return 0; }"
        result = detect_language(Path("header.h"), c_code)
        # .h defaults to C
        assert result in (Language.C, Language.CPP)


class TestGetLanguageExtensions:
    """Tests for get_language_extensions."""

    def test_python_extensions(self):
        exts = get_language_extensions(Language.PYTHON)
        assert ".py" in exts
        assert ".pyi" in exts

    def test_unknown_has_no_extensions(self):
        exts = get_language_extensions(Language.UNKNOWN)
        assert exts == []


class TestGetSupportedLanguages:
    """Tests for get_supported_languages."""

    def test_returns_list(self):
        langs = get_supported_languages()
        assert isinstance(langs, list)
        assert Language.PYTHON in langs
        assert Language.JAVASCRIPT in langs


class TestIsTextFile:
    """Tests for is_text_file."""

    def test_python_is_text(self):
        assert is_text_file(Path("main.py")) is True

    def test_json_is_text(self):
        assert is_text_file(Path("data.json")) is True

    def test_binary_is_not_text(self):
        assert is_text_file(Path("image.png")) is False

    def test_dockerfile_is_text(self):
        assert is_text_file(Path("Dockerfile")) is True
