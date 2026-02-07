"""
Analysis utilities for metadata indexing.

This module provides helper functions for analyzing code files and entities,
including complexity calculation, import extraction, and dependency analysis.

Functions:
    calculate_file_complexity: Calculate complexity score for a file
    calculate_entity_complexity: Calculate complexity score for an entity
    extract_imports: Extract import statements from content
    extract_dependencies: Extract dependency information from content
    create_entity_text: Create text representation of entity for semantic embedding
"""

from __future__ import annotations

from ...core.types import CodeEntity, Language


def calculate_file_complexity(content: str, language: Language) -> float:
    """Calculate a simple complexity score for a file."""
    lines = content.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    # Basic complexity metrics
    complexity = 0.0
    complexity += len(non_empty_lines) * 0.1  # Line count
    complexity += content.count("if ") * 0.5  # Conditional statements
    complexity += content.count("for ") * 0.5  # Loops
    complexity += content.count("while ") * 0.5  # Loops
    complexity += content.count("try:") * 0.3  # Exception handling
    complexity += content.count("def ") * 0.2  # Function definitions
    complexity += content.count("class ") * 0.4  # Class definitions

    return min(complexity, 100.0)  # Cap at 100


def calculate_entity_complexity(entity: CodeEntity, content: str) -> float:
    """Calculate complexity score for an entity."""
    if not entity.signature:
        return 0.0

    # Extract entity content
    lines = content.split("\n")
    start_idx = max(0, entity.start_line - 1)
    end_idx = min(len(lines), entity.end_line)
    entity_content = "\n".join(lines[start_idx:end_idx])

    # Calculate complexity
    complexity = 0.0
    complexity += len(entity_content.split("\n")) * 0.1
    complexity += entity_content.count("if ") * 0.5
    complexity += entity_content.count("for ") * 0.5
    complexity += entity_content.count("while ") * 0.5
    complexity += entity_content.count("try:") * 0.3

    return min(complexity, 50.0)  # Cap at 50 for entities


def extract_imports(content: str, language: Language) -> list[str]:
    """Extract import statements from content."""
    imports = []
    lines = content.split("\n")

    for line in lines:
        stripped = line.strip()
        if language == Language.PYTHON:
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            if stripped.startswith("import ") or (
                stripped.startswith("const ") and "require(" in stripped
            ):
                imports.append(stripped)
        elif language in [Language.JAVA, Language.KOTLIN, Language.SCALA]:
            if stripped.startswith("import "):
                imports.append(stripped)
        elif language == Language.GO:
            if stripped.startswith("import ") or stripped.startswith('"'):
                imports.append(stripped)
        elif language == Language.RUST:
            if stripped.startswith("use ") or stripped.startswith("extern crate "):
                imports.append(stripped)
        elif language == Language.CSHARP:
            if stripped.startswith("using "):
                imports.append(stripped)
        elif language == Language.CPP or language == Language.C:
            if stripped.startswith("#include "):
                imports.append(stripped)
        elif language == Language.PHP:
            if (
                stripped.startswith("use ")
                or stripped.startswith("require")
                or stripped.startswith("include")
            ):
                imports.append(stripped)
        elif language == Language.RUBY:
            if stripped.startswith("require ") or stripped.startswith("require_relative "):
                imports.append(stripped)
        elif language == Language.SWIFT:
            if stripped.startswith("import "):
                imports.append(stripped)
        elif language == Language.R:
            if stripped.startswith("library(") or stripped.startswith("require("):
                imports.append(stripped)

    return imports[:30]  # Limit to first 30 imports


def extract_dependencies(content: str, language: Language) -> list[str]:
    """Extract dependency information from content.

    Uses import statements and common library usage patterns to identify
    external dependencies referenced in the code.
    """
    import re

    dependencies: set[str] = set()

    if language == Language.PYTHON:
        # Extract top-level module names from import statements
        for match in re.finditer(r"^\s*(?:from|import)\s+([\w.]+)", content, re.MULTILINE):
            top_module = match.group(1).split(".")[0]
            dependencies.add(top_module)
    elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
        # Extract package names from import/require
        for match in re.finditer(r"(?:from|require\()\s*['\"]([^'\"./][^'\"]*)['\"]", content):
            pkg = match.group(1).split("/")[0]
            if pkg.startswith("@"):
                # Scoped package: @scope/name
                parts = match.group(1).split("/")
                pkg = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
            dependencies.add(pkg)
    elif language in [Language.JAVA, Language.KOTLIN, Language.SCALA]:
        for match in re.finditer(r"^\s*import\s+([\w.]+)", content, re.MULTILINE):
            # Use the top two segments as the dependency identifier
            parts = match.group(1).split(".")
            if len(parts) >= 2:
                dependencies.add(f"{parts[0]}.{parts[1]}")
    elif language == Language.GO:
        for match in re.finditer(r'"([^"]+)"', content):
            dep = match.group(1)
            if "/" in dep:
                dependencies.add(dep)
    elif language == Language.RUST:
        for match in re.finditer(r"^\s*(?:use|extern crate)\s+([\w:]+)", content, re.MULTILINE):
            crate = match.group(1).split("::")[0]
            dependencies.add(crate)
    elif language == Language.CSHARP:
        for match in re.finditer(r"^\s*using\s+([\w.]+)", content, re.MULTILINE):
            namespace = match.group(1).split(".")[0]
            dependencies.add(namespace)
    elif language in [Language.C, Language.CPP]:
        for match in re.finditer(
            r'#include\s*[<"]([^>"]+)[>"]', content
        ):
            header = match.group(1).split("/")[0].replace(".h", "")
            dependencies.add(header)
    elif language == Language.RUBY:
        for match in re.finditer(
            r"^\s*(?:require|require_relative|gem)\s+['\"]([^'\"]+)['\"]",
            content,
            re.MULTILINE,
        ):
            dependencies.add(match.group(1).split("/")[0])
    elif language == Language.PHP:
        for match in re.finditer(
            r"^\s*use\s+([\w\\]+)", content, re.MULTILINE
        ):
            namespace = match.group(1).split("\\")[0]
            dependencies.add(namespace)

    return sorted(dependencies)


def create_entity_text(entity: CodeEntity) -> str:
    """Create text representation of entity for semantic embedding."""
    parts = [entity.name]

    if entity.signature:
        parts.append(entity.signature)

    if entity.docstring:
        parts.append(entity.docstring)

    # Add some context from properties
    if entity.properties:
        for key, value in entity.properties.items():
            if isinstance(value, str):
                parts.append(f"{key}: {value}")
            elif isinstance(value, list):
                parts.append(f"{key}: {value}")

    return " ".join(parts)
