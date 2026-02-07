"""
Language definitions for pysearch.

This module contains language-related constants and enumerations used across
the pysearch system. It's kept minimal to avoid circular imports.
"""

from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    """Supported programming languages for syntax-aware processing."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    C = "c"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    PHP = "php"
    RUBY = "ruby"
    KOTLIN = "kotlin"
    SWIFT = "swift"
    SCALA = "scala"
    R = "r"
    MATLAB = "matlab"
    SHELL = "shell"
    POWERSHELL = "powershell"
    SQL = "sql"
    HTML = "html"
    CSS = "css"
    XML = "xml"
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    MARKDOWN = "markdown"
    DOCKERFILE = "dockerfile"
    MAKEFILE = "makefile"
    UNKNOWN = "unknown"
