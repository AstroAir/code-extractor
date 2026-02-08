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


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------
class TestConstants:
    """Ensure mapping constants are well-formed."""

    def test_extension_map_not_empty(self):
        assert len(EXTENSION_MAP) > 0

    def test_filename_patterns_not_empty(self):
        assert len(FILENAME_PATTERNS) > 0

    def test_shebang_patterns_not_empty(self):
        assert len(SHEBANG_PATTERNS) > 0

    def test_content_patterns_not_empty(self):
        assert len(CONTENT_PATTERNS) > 0

    def test_extension_map_values_are_language(self):
        for ext, lang in EXTENSION_MAP.items():
            assert isinstance(lang, Language), f"{ext} mapped to non-Language {lang}"


# ---------------------------------------------------------------------------
# detect_language_by_extension
# ---------------------------------------------------------------------------
class TestDetectLanguageByExtension:
    """Tests for detect_language_by_extension."""

    @pytest.mark.parametrize(
        "filename, expected",
        [
            ("main.py", Language.PYTHON),
            ("app.js", Language.JAVASCRIPT),
            ("app.ts", Language.TYPESCRIPT),
            ("Main.java", Language.JAVA),
            ("main.go", Language.GO),
            ("main.rs", Language.RUST),
            ("style.css", Language.CSS),
            ("page.html", Language.HTML),
            ("data.json", Language.JSON),
            ("config.yaml", Language.YAML),
            ("config.yml", Language.YAML),
            ("notes.md", Language.MARKDOWN),
            ("script.sh", Language.SHELL),
            ("app.rb", Language.RUBY),
            ("app.php", Language.PHP),
            ("Example.cs", Language.CSHARP),
            ("Example.kt", Language.KOTLIN),
            ("query.sql", Language.SQL),
            ("config.toml", Language.TOML),
        ],
    )
    def test_known_extensions(self, filename: str, expected: Language):
        assert detect_language_by_extension(Path(filename)) == expected

    def test_unknown(self):
        assert detect_language_by_extension(Path("file.xyz")) == Language.UNKNOWN

    def test_case_insensitive(self):
        assert detect_language_by_extension(Path("file.PY")) == Language.PYTHON


# ---------------------------------------------------------------------------
# detect_language_by_filename
# ---------------------------------------------------------------------------
class TestDetectLanguageByFilename:
    """Tests for detect_language_by_filename."""

    def test_dockerfile(self):
        assert detect_language_by_filename(Path("Dockerfile")) == Language.DOCKERFILE

    def test_makefile(self):
        assert detect_language_by_filename(Path("Makefile")) == Language.MAKEFILE

    def test_gnumakefile(self):
        assert detect_language_by_filename(Path("GNUmakefile")) == Language.MAKEFILE

    def test_rakefile(self):
        assert detect_language_by_filename(Path("Rakefile")) == Language.RUBY

    def test_gemfile(self):
        assert detect_language_by_filename(Path("Gemfile")) == Language.RUBY

    def test_unknown_filename(self):
        assert detect_language_by_filename(Path("random.txt")) == Language.UNKNOWN


# ---------------------------------------------------------------------------
# detect_language_by_shebang
# ---------------------------------------------------------------------------
class TestDetectLanguageByShebang:
    """Tests for detect_language_by_shebang."""

    def test_python_shebang(self):
        assert detect_language_by_shebang("#!/usr/bin/env python\nprint()") == Language.PYTHON

    def test_python_direct(self):
        assert detect_language_by_shebang("#!/usr/bin/python\nprint()") == Language.PYTHON

    def test_bash_shebang(self):
        assert detect_language_by_shebang("#!/bin/bash\necho hello") == Language.SHELL

    def test_sh_shebang(self):
        assert detect_language_by_shebang("#!/bin/sh\necho hello") == Language.SHELL

    def test_node_shebang(self):
        assert detect_language_by_shebang("#!/usr/bin/env node\nconsole.log()") == Language.JAVASCRIPT

    def test_ruby_shebang(self):
        assert detect_language_by_shebang("#!/usr/bin/env ruby\nputs 'hi'") == Language.RUBY

    def test_no_shebang(self):
        assert detect_language_by_shebang("just some text") == Language.UNKNOWN

    def test_empty_content(self):
        assert detect_language_by_shebang("") == Language.UNKNOWN


# ---------------------------------------------------------------------------
# detect_language_by_content
# ---------------------------------------------------------------------------
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

    def test_go_content(self):
        code = "package main\n\nfunc main() {\n}"
        result = detect_language_by_content(code, {Language.GO, Language.PYTHON})
        assert result == Language.GO

    def test_rust_content(self):
        code = 'fn main() {\n    let mut x = 5;\n    println!("{}", x);\n}'
        result = detect_language_by_content(code, {Language.RUST, Language.PYTHON})
        assert result == Language.RUST

    def test_empty_content(self):
        assert detect_language_by_content("", {Language.PYTHON}) == Language.UNKNOWN

    def test_empty_candidates(self):
        assert detect_language_by_content("import os", set()) == Language.UNKNOWN

    def test_no_match(self):
        result = detect_language_by_content("random text with no patterns", {Language.GO})
        assert result == Language.UNKNOWN


# ---------------------------------------------------------------------------
# detect_language (comprehensive)
# ---------------------------------------------------------------------------
class TestDetectLanguage:
    """Tests for detect_language (comprehensive)."""

    def test_by_extension(self):
        assert detect_language(Path("main.py")) == Language.PYTHON

    def test_by_filename(self):
        assert detect_language(Path("Dockerfile")) == Language.DOCKERFILE

    def test_filename_takes_priority_over_extension(self):
        # Makefile has no extension but matches filename pattern
        assert detect_language(Path("Makefile")) == Language.MAKEFILE

    def test_with_content_shebang(self):
        result = detect_language(Path("script"), "#!/usr/bin/env python\nprint()")
        assert result == Language.PYTHON

    def test_content_fallback_for_unknown_extension(self):
        python_code = "import os\ndef main():\n    pass"
        result = detect_language(Path("unknown_file"), python_code)
        assert result == Language.PYTHON

    def test_unknown_no_content(self):
        assert detect_language(Path("file.xyz")) == Language.UNKNOWN

    def test_ambiguous_h_extension_with_content(self):
        c_code = "#include <stdio.h>\nint main() { return 0; }"
        result = detect_language(Path("header.h"), c_code)
        assert result in (Language.C, Language.CPP)

    def test_extension_with_no_content(self):
        assert detect_language(Path("app.js")) == Language.JAVASCRIPT


# ---------------------------------------------------------------------------
# get_language_extensions
# ---------------------------------------------------------------------------
class TestGetLanguageExtensions:
    """Tests for get_language_extensions."""

    def test_python_extensions(self):
        exts = get_language_extensions(Language.PYTHON)
        assert ".py" in exts
        assert ".pyi" in exts
        assert ".pyx" in exts

    def test_javascript_extensions(self):
        exts = get_language_extensions(Language.JAVASCRIPT)
        assert ".js" in exts
        assert ".jsx" in exts

    def test_unknown_has_no_extensions(self):
        exts = get_language_extensions(Language.UNKNOWN)
        assert exts == []


# ---------------------------------------------------------------------------
# get_supported_languages
# ---------------------------------------------------------------------------
class TestGetSupportedLanguages:
    """Tests for get_supported_languages."""

    def test_returns_list(self):
        langs = get_supported_languages()
        assert isinstance(langs, list)
        assert Language.PYTHON in langs
        assert Language.JAVASCRIPT in langs

    def test_includes_common_languages(self):
        langs = get_supported_languages()
        for lang in [Language.JAVA, Language.GO, Language.RUST, Language.CSHARP]:
            assert lang in langs


# ---------------------------------------------------------------------------
# is_text_file
# ---------------------------------------------------------------------------
class TestIsTextFile:
    """Tests for is_text_file."""

    @pytest.mark.parametrize(
        "filename",
        ["main.py", "data.json", "page.html", "style.css", "config.yaml",
         "script.sh", "notes.md", "data.xml", "query.sql", "app.ts"],
    )
    def test_text_files(self, filename: str):
        assert is_text_file(Path(filename)) is True

    @pytest.mark.parametrize(
        "filename",
        ["image.png", "video.mp4", "archive.zip", "binary.exe"],
    )
    def test_binary_files(self, filename: str):
        assert is_text_file(Path(filename)) is False

    def test_dockerfile_is_text(self):
        assert is_text_file(Path("Dockerfile")) is True

    def test_makefile_is_text(self):
        assert is_text_file(Path("Makefile")) is True
