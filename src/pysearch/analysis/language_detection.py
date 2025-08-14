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
}

# Special filename patterns
FILENAME_PATTERNS: dict[str, Language] = {
    "Dockerfile": Language.DOCKERFILE,
    "dockerfile": Language.DOCKERFILE,
    "Makefile": Language.MAKEFILE,
    "makefile": Language.MAKEFILE,
    "GNUmakefile": Language.MAKEFILE,
    "Rakefile": Language.RUBY,
    "Gemfile": Language.RUBY,
    "Podfile": Language.RUBY,
    "Vagrantfile": Language.RUBY,
}

# Shebang patterns for script detection
SHEBANG_PATTERNS: dict[str, Language] = {
    r"#!/usr/bin/env python": Language.PYTHON,
    r"#!/usr/bin/python": Language.PYTHON,
    r"#!/usr/bin/env node": Language.JAVASCRIPT,
    r"#!/usr/bin/node": Language.JAVASCRIPT,
    r"#!/bin/bash": Language.SHELL,
    r"#!/bin/sh": Language.SHELL,
    r"#!/usr/bin/env bash": Language.SHELL,
    r"#!/usr/bin/env sh": Language.SHELL,
    r"#!/usr/bin/env ruby": Language.RUBY,
    r"#!/usr/bin/ruby": Language.RUBY,
    r"#!/usr/bin/env php": Language.PHP,
    r"#!/usr/bin/php": Language.PHP,
    r"#!/usr/bin/env perl": Language.UNKNOWN,  # Perl not in our enum yet
    r"#!/usr/bin/perl": Language.UNKNOWN,
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
            if path.suffix.lower() in [".h", ".m"]:
                content_lang = detect_language_by_content(
                    content, {Language.C, Language.CPP, Language.MATLAB}
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
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".scala",
        ".r",
        ".m",
        ".sh",
        ".ps1",
        ".sql",
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
    }

    return path.suffix.lower() in text_extensions or path.name in FILENAME_PATTERNS
