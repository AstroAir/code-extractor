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

from typing import List

from ...core.types import CodeEntity, Language


def calculate_file_complexity(content: str, language: Language) -> float:
    """Calculate a simple complexity score for a file."""
    lines = content.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]

    # Basic complexity metrics
    complexity = 0.0
    complexity += len(non_empty_lines) * 0.1  # Line count
    complexity += content.count('if ') * 0.5  # Conditional statements
    complexity += content.count('for ') * 0.5  # Loops
    complexity += content.count('while ') * 0.5  # Loops
    complexity += content.count('try:') * 0.3  # Exception handling
    complexity += content.count('def ') * 0.2  # Function definitions
    complexity += content.count('class ') * 0.4  # Class definitions

    return min(complexity, 100.0)  # Cap at 100


def calculate_entity_complexity(entity: CodeEntity, content: str) -> float:
    """Calculate complexity score for an entity."""
    if not entity.signature:
        return 0.0

    # Extract entity content
    lines = content.split('\n')
    start_idx = max(0, entity.start_line - 1)
    end_idx = min(len(lines), entity.end_line)
    entity_content = '\n'.join(lines[start_idx:end_idx])

    # Calculate complexity
    complexity = 0.0
    complexity += len(entity_content.split('\n')) * 0.1
    complexity += entity_content.count('if ') * 0.5
    complexity += entity_content.count('for ') * 0.5
    complexity += entity_content.count('while ') * 0.5
    complexity += entity_content.count('try:') * 0.3

    return min(complexity, 50.0)  # Cap at 50 for entities


def extract_imports(content: str, language: Language) -> List[str]:
    """Extract import statements from content."""
    imports = []
    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        if language == Language.PYTHON:
            if line.startswith('import ') or line.startswith('from '):
                imports.append(line)
        elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            if 'import ' in line and 'from ' in line:
                imports.append(line)
        elif language == Language.JAVA:
            if line.startswith('import '):
                imports.append(line)

    return imports[:20]  # Limit to first 20 imports


def extract_dependencies(content: str, language: Language) -> List[str]:
    """Extract dependency information from content."""
    # This is a simplified implementation
    # In practice, you'd want more sophisticated dependency analysis
    dependencies = []

    if language == Language.PYTHON:
        # Look for common library usage patterns
        common_libs = ['requests', 'numpy',
                       'pandas', 'flask', 'django', 'fastapi']
        for lib in common_libs:
            if lib in content:
                dependencies.append(lib)

    return dependencies


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

    return ' '.join(parts)
