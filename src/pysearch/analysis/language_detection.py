from __future__ import annotations

import re
from pathlib import Path

from ..core.types import Language

# File extension to language mapping
EXTENSION_MAP: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".pyi": Language.PYTHON,
    ".pyx": Language.PYTHON,
    # JavaScript/TypeScript
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".d.ts": Language.TYPESCRIPT,
    # Java/JVM Languages
    ".java": Language.JAVA,
    ".kt": Language.KOTLIN,
    ".kts": Language.KOTLIN,
    ".scala": Language.SCALA,
    ".sc": Language.SCALA,
    # C/C++
    ".c": Language.C,
    ".h": Language.C,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".cc": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    ".hh": Language.CPP,
    # C#
    ".cs": Language.CSHARP,
    ".csx": Language.CSHARP,
    # Go
    ".go": Language.GO,
    # Rust
    ".rs": Language.RUST,
    # PHP
    ".php": Language.PHP,
    ".php3": Language.PHP,
    ".php4": Language.PHP,
    ".php5": Language.PHP,
    ".phtml": Language.PHP,
    # Ruby
    ".rb": Language.RUBY,
    ".rbw": Language.RUBY,
    ".rake": Language.RUBY,
    ".gemspec": Language.RUBY,
    # Swift
    ".swift": Language.SWIFT,
    # R
    ".r": Language.R,
    ".R": Language.R,
    # MATLAB
    ".m": Language.MATLAB,
    ".mat": Language.MATLAB,
    # Shell
    ".sh": Language.SHELL,
    ".bash": Language.SHELL,
    ".zsh": Language.SHELL,
    ".fish": Language.SHELL,
    ".ksh": Language.SHELL,
    # PowerShell
    ".ps1": Language.POWERSHELL,
    ".psm1": Language.POWERSHELL,
    ".psd1": Language.POWERSHELL,
    # SQL
    ".sql": Language.SQL,
    ".mysql": Language.SQL,
    ".pgsql": Language.SQL,
    ".plsql": Language.SQL,
    # Web
    ".html": Language.HTML,
    ".htm": Language.HTML,
    ".xhtml": Language.HTML,
    ".css": Language.CSS,
    ".scss": Language.CSS,
    ".sass": Language.CSS,
    ".less": Language.CSS,
    # Data formats
    ".xml": Language.XML,
    ".xsd": Language.XML,
    ".xsl": Language.XML,
    ".json": Language.JSON,
    ".yaml": Language.YAML,
    ".yml": Language.YAML,
    ".toml": Language.TOML,
    # Documentation
    ".md": Language.MARKDOWN,
    ".markdown": Language.MARKDOWN,
    ".mdown": Language.MARKDOWN,
    ".mkd": Language.MARKDOWN,
    ".rst": Language.MARKDOWN,
    # Lua
    ".lua": Language.LUA,
    ".luau": Language.LUA,
    # Perl
    ".pl": Language.PERL,
    ".pm": Language.PERL,
    ".pod": Language.PERL,
    ".t": Language.PERL,
    # Dart
    ".dart": Language.DART,
    # Elixir
    ".ex": Language.ELIXIR,
    ".exs": Language.ELIXIR,
    ".heex": Language.ELIXIR,
    # Haskell
    ".hs": Language.HASKELL,
    ".lhs": Language.HASKELL,
    # Julia
    ".jl": Language.JULIA,
    # Groovy
    ".groovy": Language.GROOVY,
    ".gvy": Language.GROOVY,
    ".gy": Language.GROOVY,
    ".gsh": Language.GROOVY,
    # Objective-C
    ".mm": Language.OBJECTIVE_C,
    # Zig
    ".zig": Language.ZIG,
}

# Special filename patterns
FILENAME_PATTERNS: dict[str, Language] = {
    "Dockerfile": Language.DOCKERFILE,
    "dockerfile": Language.DOCKERFILE,
    "Dockerfile.dev": Language.DOCKERFILE,
    "Dockerfile.prod": Language.DOCKERFILE,
    "Makefile": Language.MAKEFILE,
    "makefile": Language.MAKEFILE,
    "GNUmakefile": Language.MAKEFILE,
    "Rakefile": Language.RUBY,
    "Gemfile": Language.RUBY,
    "Podfile": Language.RUBY,
    "Vagrantfile": Language.RUBY,
    "CMakeLists.txt": Language.MAKEFILE,
    "Justfile": Language.MAKEFILE,
    "justfile": Language.MAKEFILE,
    "SConstruct": Language.PYTHON,
    "SConscript": Language.PYTHON,
    "BUILD": Language.PYTHON,
    "BUILD.bazel": Language.PYTHON,
    "WORKSPACE": Language.PYTHON,
    "Jenkinsfile": Language.GROOVY,
    "build.gradle": Language.GROOVY,
    "build.gradle.kts": Language.KOTLIN,
    "settings.gradle": Language.GROOVY,
    "settings.gradle.kts": Language.KOTLIN,
}

# Shebang patterns for script detection
SHEBANG_PATTERNS: dict[str, Language] = {
    r"#!/usr/bin/env python": Language.PYTHON,
    r"#!/usr/bin/python": Language.PYTHON,
    r"#!/usr/bin/env python3": Language.PYTHON,
    r"#!/usr/bin/python3": Language.PYTHON,
    r"#!/usr/bin/env node": Language.JAVASCRIPT,
    r"#!/usr/bin/node": Language.JAVASCRIPT,
    r"#!/bin/bash": Language.SHELL,
    r"#!/bin/sh": Language.SHELL,
    r"#!/usr/bin/env bash": Language.SHELL,
    r"#!/usr/bin/env sh": Language.SHELL,
    r"#!/usr/bin/env zsh": Language.SHELL,
    r"#!/bin/zsh": Language.SHELL,
    r"#!/usr/bin/env ruby": Language.RUBY,
    r"#!/usr/bin/ruby": Language.RUBY,
    r"#!/usr/bin/env php": Language.PHP,
    r"#!/usr/bin/php": Language.PHP,
    r"#!/usr/bin/env perl": Language.PERL,
    r"#!/usr/bin/perl": Language.PERL,
    r"#!/usr/bin/env lua": Language.LUA,
    r"#!/usr/bin/lua": Language.LUA,
    r"#!/usr/bin/env elixir": Language.ELIXIR,
    r"#!/usr/bin/env julia": Language.JULIA,
    r"#!/usr/bin/env dart": Language.DART,
    r"#!/usr/bin/env groovy": Language.GROOVY,
}

# Content-based detection patterns
CONTENT_PATTERNS: dict[Language, list[str]] = {
    Language.PYTHON: [
        r"^import\s+\w+",
        r"^from\s+\w+\s+import",
        r"def\s+\w+\s*\(",
        r"class\s+\w+\s*\(",
        r"if\s+__name__\s*==\s*['\"]__main__['\"]",
    ],
    Language.JAVASCRIPT: [
        r"function\s+\w+\s*\(",
        r"const\s+\w+\s*=",
        r"let\s+\w+\s*=",
        r"var\s+\w+\s*=",
        r"require\s*\(",
        r"module\.exports",
        r"export\s+(default\s+)?",
    ],
    Language.TYPESCRIPT: [
        r"interface\s+\w+",
        r"type\s+\w+\s*=",
        r":\s*(string|number|boolean|any)",
        r"export\s+interface",
        r"import\s+.*\s+from\s+['\"]",
    ],
    Language.JAVA: [
        r"public\s+class\s+\w+",
        r"package\s+[\w\.]+;",
        r"import\s+[\w\.]+;",
        r"public\s+static\s+void\s+main",
        r"@\w+",  # annotations
    ],
    Language.GO: [
        r"package\s+\w+",
        r"import\s+\(",
        r"func\s+\w+\s*\(",
        r"type\s+\w+\s+struct",
        r"go\s+\w+\(",
    ],
    Language.RUST: [
        r"fn\s+\w+\s*\(",
        r"struct\s+\w+",
        r"impl\s+\w+",
        r"use\s+\w+::",
        r"let\s+mut\s+\w+",
        r"#\[derive\(",
    ],
    Language.PHP: [
        r"<\?php",
        r"function\s+\w+\s*\(",
        r"class\s+\w+",
        r"\$\w+\s*=",
        r"echo\s+",
    ],
    Language.RUBY: [
        r"def\s+\w+",
        r"class\s+\w+",
        r"module\s+\w+",
        r"require\s+['\"]",
        r"puts\s+",
        r"end\s*$",
    ],
    Language.C: [
        r"#include\s*[<\"]",
        r"int\s+main\s*\(",
        r"void\s+\w+\s*\(",
        r"typedef\s+",
        r"struct\s+\w+\s*\{",
        r"printf\s*\(",
    ],
    Language.CPP: [
        r"#include\s*[<\"]",
        r"using\s+namespace",
        r"std::",
        r"class\s+\w+\s*[:{]",
        r"template\s*<",
        r"cout\s*<<",
        r"nullptr",
        r"auto\s+\w+\s*=",
    ],
    Language.CSHARP: [
        r"using\s+[A-Z][\w.]*;",
        r"namespace\s+[\w.]+",
        r"public\s+(class|interface|enum|struct)\s+\w+",
        r"private\s+(void|int|string|bool)",
        r"\[\w+\]",
        r"Console\.Write",
    ],
    Language.KOTLIN: [
        r"fun\s+\w+\s*\(",
        r"val\s+\w+",
        r"var\s+\w+",
        r"class\s+\w+",
        r"object\s+\w+",
        r"data\s+class",
        r"import\s+[\w.]+",
        r"package\s+[\w.]+",
    ],
    Language.SWIFT: [
        r"func\s+\w+\s*\(",
        r"let\s+\w+",
        r"var\s+\w+",
        r"class\s+\w+",
        r"struct\s+\w+",
        r"protocol\s+\w+",
        r"import\s+\w+",
        r"guard\s+let",
    ],
    Language.SCALA: [
        r"object\s+\w+\s*\{",
        r"trait\s+\w+",
        r"case\s+class",
        r"val\s+\w+\s*:",
        r"var\s+\w+\s*:",
        r"def\s+\w+\s*\([^)]*\)\s*:",
        r"sealed\s+(?:trait|class|abstract)",
        r"implicit\s+",
    ],
    Language.SHELL: [
        r"#!/bin/(ba)?sh",
        r"\w+\s*\(\)\s*\{",
        r"function\s+\w+",
        r"if\s+\[\s+",
        r"echo\s+",
        r"export\s+\w+=",
        r"source\s+",
        r"\$\{\w+\}",
    ],
    Language.POWERSHELL: [
        r"function\s+\w+",
        r"\$\w+\s*=",
        r"Write-Host",
        r"Get-\w+",
        r"Set-\w+",
        r"Import-Module",
        r"param\s*\(",
        r"\[CmdletBinding\(\)\]",
    ],
    Language.SQL: [
        r"SELECT\s+",
        r"FROM\s+",
        r"WHERE\s+",
        r"CREATE\s+(TABLE|VIEW|INDEX|DATABASE)",
        r"INSERT\s+INTO",
        r"ALTER\s+TABLE",
        r"DROP\s+TABLE",
        r"JOIN\s+",
    ],
    Language.R: [
        r"<-\s*function\s*\(",
        r"library\s*\(",
        r"require\s*\(",
        r"data\.frame\s*\(",
        r"ggplot\s*\(",
        r"<-\s*",
    ],
    Language.LUA: [
        r"function\s+\w+\s*\(",
        r"local\s+function",
        r"local\s+\w+\s*=",
        r"require\s*[\(\"]",
        r"end\s*$",
        r"then\s*$",
    ],
    Language.PERL: [
        r"use\s+strict",
        r"use\s+warnings",
        r"sub\s+\w+",
        r"my\s+\$\w+",
        r"print\s+",
        r"use\s+[A-Z]\w+",
    ],
    Language.DART: [
        r"import\s+['\"]package:",
        r"void\s+main\s*\(",
        r"class\s+\w+",
        r"final\s+\w+",
        r"Widget\s+build",
        r"@override",
    ],
    Language.ELIXIR: [
        r"defmodule\s+\w+",
        r"defp\s+\w+",
        r"\|>\s*",
        r"alias\s+\w+",
        r"@moduledoc",
        r"@doc\s+",
        r"def\s+\w+.*\bdo\b",
    ],
    Language.HASKELL: [
        r"module\s+\w+",
        r"import\s+(qualified\s+)?\w+",
        r"::\s*\w+\s*->\s*\w+",
        r"data\s+\w+",
        r"where\s*$",
        r"do\s*$",
    ],
    Language.JULIA: [
        r"function\s+\w+\s*\(",
        r"module\s+\w+",
        r"using\s+\w+",
        r"import\s+\w+",
        r"struct\s+\w+",
        r"end\s*$",
    ],
    Language.GROOVY: [
        r"def\s+\w+",
        r"class\s+\w+",
        r"import\s+[\w.]+",
        r"println\s+",
        r"@\w+",
    ],
    Language.OBJECTIVE_C: [
        r"#import\s*[<\"]",
        r"@interface\s+\w+",
        r"@implementation\s+\w+",
        r"@property",
        r"\[\w+\s+\w+\]",
        r"@end",
    ],
    Language.ZIG: [
        r"const\s+std\s*=",
        r"pub\s+fn\s+\w+",
        r"fn\s+\w+\s*\(",
        r"@import\s*\(",
        r"try\s+",
        r"comptime",
    ],
    Language.MATLAB: [
        r"function\s+.*=\s*\w+\s*\(",
        r"classdef\s+\w+",
        r"disp\s*\(",
        r"fprintf\s*\(",
        r"%\s+",
    ],
}


def detect_language_by_extension(path: Path) -> Language:
    """Detect language based on file extension."""
    suffix = path.suffix.lower()
    return EXTENSION_MAP.get(suffix, Language.UNKNOWN)


def detect_language_by_filename(path: Path) -> Language:
    """Detect language based on filename patterns."""
    name = path.name
    return FILENAME_PATTERNS.get(name, Language.UNKNOWN)


def detect_language_by_shebang(content: str) -> Language:
    """Detect language based on shebang line."""
    if not content:
        return Language.UNKNOWN

    first_line = content.split("\n", 1)[0].strip()
    if not first_line.startswith("#!"):
        return Language.UNKNOWN

    for pattern, language in SHEBANG_PATTERNS.items():
        if re.search(pattern, first_line):
            return language

    return Language.UNKNOWN


def detect_language_by_content(content: str, candidates: set[Language]) -> Language:
    """Detect language based on content patterns."""
    if not content or not candidates:
        return Language.UNKNOWN

    # Score each candidate language
    scores: dict[Language, int] = {lang: 0 for lang in candidates}

    for language in candidates:
        patterns = CONTENT_PATTERNS.get(language, [])
        for pattern in patterns:
            if re.search(pattern, content, re.MULTILINE):
                scores[language] += 1

    # Return language with highest score
    if scores:
        best_lang = max(scores.items(), key=lambda x: x[1])
        if best_lang[1] > 0:
            return best_lang[0]

    return Language.UNKNOWN


def detect_language(path: Path, content: str | None = None) -> Language:
    """
    Comprehensive language detection using multiple strategies.

    Args:
        path: File path
        content: Optional file content for content-based detection

    Returns:
        Detected language
    """
    # Strategy 1: Filename patterns (highest priority)
    lang = detect_language_by_filename(path)
    if lang != Language.UNKNOWN:
        return lang

    # Strategy 2: File extension
    lang = detect_language_by_extension(path)
    if lang != Language.UNKNOWN:
        # If we have content, verify with shebang or content patterns
        if content:
            shebang_lang = detect_language_by_shebang(content)
            if shebang_lang != Language.UNKNOWN:
                return shebang_lang

            # For ambiguous extensions, use content detection
            # Could be C/C++/Objective-C/MATLAB
            if path.suffix.lower() in [".h", ".m", ".mm"]:
                content_lang = detect_language_by_content(
                    content,
                    {Language.C, Language.CPP, Language.OBJECTIVE_C, Language.MATLAB},
                )
                if content_lang != Language.UNKNOWN:
                    return content_lang

        return lang

    # Strategy 3: Shebang detection
    if content:
        lang = detect_language_by_shebang(content)
        if lang != Language.UNKNOWN:
            return lang

        # Strategy 4: Content-based detection (fallback)
        # Try common languages
        common_languages = {
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.JAVA,
            Language.GO,
            Language.RUST,
            Language.PHP,
            Language.RUBY,
            Language.C,
            Language.CPP,
            Language.CSHARP,
            Language.KOTLIN,
            Language.SWIFT,
            Language.SCALA,
            Language.SHELL,
            Language.LUA,
            Language.PERL,
            Language.DART,
            Language.ELIXIR,
            Language.HASKELL,
        }
        lang = detect_language_by_content(content, common_languages)
        if lang != Language.UNKNOWN:
            return lang

    return Language.UNKNOWN


def get_language_extensions(language: Language) -> list[str]:
    """Get all file extensions associated with a language."""
    return [ext for ext, lang in EXTENSION_MAP.items() if lang == language]


def get_supported_languages() -> list[Language]:
    """Get list of all supported languages."""
    return list(Language)


def is_text_file(path: Path) -> bool:
    """Check if a file is likely a text file based on extension."""
    text_extensions = {
        # Code files
        ".py",
        ".pyi",
        ".pyx",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".mjs",
        ".java",
        ".c",
        ".cpp",
        ".cc",
        ".cxx",
        ".h",
        ".hpp",
        ".hxx",
        ".hh",
        ".cs",
        ".csx",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".rake",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sc",
        ".r",
        ".m",
        ".mm",
        ".sh",
        ".bash",
        ".zsh",
        ".fish",
        ".ps1",
        ".psm1",
        ".sql",
        ".lua",
        ".pl",
        ".pm",
        ".dart",
        ".ex",
        ".exs",
        ".hs",
        ".jl",
        ".groovy",
        ".zig",
        # Web files
        ".html",
        ".htm",
        ".css",
        ".scss",
        ".sass",
        ".less",
        # Data files
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".csv",
        ".tsv",
        # Documentation
        ".md",
        ".rst",
        ".txt",
        ".log",
        # Config files
        ".conf",
        ".cfg",
        ".ini",
        ".properties",
        ".env",
    }

    return path.suffix.lower() in text_extensions or path.name in FILENAME_PATTERNS
