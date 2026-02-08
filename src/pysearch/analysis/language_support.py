"""
Enhanced multi-language support with tree-sitter integration.

This module extends pysearch's language support with sophisticated tree-sitter
based parsing and analysis for multiple programming languages, inspired by
Continue's approach but with broader language coverage.

Classes:
    LanguageProcessor: Abstract base for language-specific processing
    TreeSitterProcessor: Tree-sitter based language processor
    LanguageRegistry: Registry of all supported language processors
    EnhancedLanguageDetector: Advanced language detection

Features:
    - Tree-sitter parsing for 20+ programming languages
    - Language-specific code chunking strategies
    - Advanced entity extraction (functions, classes, variables, etc.)
    - Dependency analysis and import resolution
    - Code structure analysis and metrics
    - Language-specific semantic patterns

Supported Languages:
    - Python, JavaScript, TypeScript, Java, C/C++, C#
    - Go, Rust, PHP, Ruby, Kotlin, Swift, Scala
    - Shell, PowerShell, SQL, HTML, CSS, XML
    - JSON, YAML, TOML, Markdown

Example:
    Basic language processing:
        >>> from pysearch.enhanced_language_support import LanguageRegistry
        >>> registry = LanguageRegistry()
        >>> processor = registry.get_processor("python")
        >>> entities = processor.extract_entities(python_code)

    Advanced chunking:
        >>> chunks = []
        >>> async for chunk in processor.chunk_code(code, max_size=1000):
        ...     chunks.append(chunk)
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import CodeEntity, EntityType, Language
from ..utils.logging_config import get_logger

logger = get_logger()

# Tree-sitter availability check
try:
    import tree_sitter
    import tree_sitter_c
    import tree_sitter_cpp
    import tree_sitter_go
    import tree_sitter_java
    import tree_sitter_javascript
    import tree_sitter_python
    import tree_sitter_rust

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "Tree-sitter libraries not available. Install with: "
        "pip install tree-sitter tree-sitter-python tree-sitter-javascript ..."
    )

try:
    import tree_sitter_typescript

    TREE_SITTER_TS_AVAILABLE = True
except ImportError:
    TREE_SITTER_TS_AVAILABLE = False

try:
    import tree_sitter_c_sharp

    TREE_SITTER_CSHARP_AVAILABLE = True
except ImportError:
    TREE_SITTER_CSHARP_AVAILABLE = False

try:
    import tree_sitter_php

    TREE_SITTER_PHP_AVAILABLE = True
except ImportError:
    TREE_SITTER_PHP_AVAILABLE = False

try:
    import tree_sitter_ruby

    TREE_SITTER_RUBY_AVAILABLE = True
except ImportError:
    TREE_SITTER_RUBY_AVAILABLE = False


@dataclass
class CodeChunk:
    """Represents a chunk of code with metadata."""

    content: str
    start_line: int
    end_line: int
    language: Language
    chunk_type: str = "code"  # "code", "comment", "docstring", "import"
    entity_name: str | None = None
    entity_type: EntityType | None = None
    complexity_score: float = 0.0
    dependencies: list[str] = field(default_factory=list)


@dataclass
class LanguageConfig:
    """Configuration for language-specific processing."""

    max_chunk_size: int = 1000
    respect_boundaries: bool = True  # Respect function/class boundaries
    include_comments: bool = True
    include_docstrings: bool = True
    include_imports: bool = True
    min_chunk_size: int = 50


class LanguageProcessor(ABC):
    """Abstract base class for language-specific processing."""

    def __init__(self, language: Language, config: LanguageConfig):
        self.language = language
        self.config = config

    @abstractmethod
    async def chunk_code(
        self,
        content: str,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Language-aware code chunking."""
        # This is an abstract async generator method
        if False:  # pragma: no cover
            yield

    @abstractmethod
    def extract_entities(self, content: str) -> list[CodeEntity]:
        """Extract code entities (functions, classes, etc.)."""
        pass

    @abstractmethod
    def analyze_dependencies(self, content: str) -> list[str]:
        """Analyze code dependencies and imports."""
        pass

    @abstractmethod
    def calculate_complexity(self, content: str) -> float:
        """Calculate code complexity score."""
        pass


class TreeSitterProcessor(LanguageProcessor):
    """Tree-sitter based language processor."""

    def __init__(self, language: Language, config: LanguageConfig):
        super().__init__(language, config)
        self.parser = None
        self.query_cache: dict[str, Any] = {}

        if TREE_SITTER_AVAILABLE:
            self._initialize_parser()

    def _initialize_parser(self) -> None:
        """Initialize tree-sitter parser for the language."""
        if not TREE_SITTER_AVAILABLE:
            return

        try:
            language_map: dict[Language, Any] = {
                Language.PYTHON: tree_sitter_python.language(),
                Language.JAVASCRIPT: tree_sitter_javascript.language(),
                Language.JAVA: tree_sitter_java.language(),
                Language.C: tree_sitter_c.language(),
                Language.CPP: tree_sitter_cpp.language(),
                Language.GO: tree_sitter_go.language(),
                Language.RUST: tree_sitter_rust.language(),
            }

            # tree-sitter-typescript exposes language_typescript() / language_tsx()
            # instead of a single language() in some versions
            if TREE_SITTER_TS_AVAILABLE:
                if hasattr(tree_sitter_typescript, "language_typescript"):
                    language_map[Language.TYPESCRIPT] = tree_sitter_typescript.language_typescript()
                elif hasattr(tree_sitter_typescript, "language"):
                    language_map[Language.TYPESCRIPT] = tree_sitter_typescript.language()

            if TREE_SITTER_CSHARP_AVAILABLE:
                if hasattr(tree_sitter_c_sharp, "language"):
                    language_map[Language.CSHARP] = tree_sitter_c_sharp.language()

            if TREE_SITTER_PHP_AVAILABLE:
                if hasattr(tree_sitter_php, "language_php"):
                    language_map[Language.PHP] = tree_sitter_php.language_php()
                elif hasattr(tree_sitter_php, "language"):
                    language_map[Language.PHP] = tree_sitter_php.language()

            if TREE_SITTER_RUBY_AVAILABLE:
                if hasattr(tree_sitter_ruby, "language"):
                    language_map[Language.RUBY] = tree_sitter_ruby.language()

            if self.language in language_map:
                self.parser = tree_sitter.Parser()
                if self.parser is not None:
                    self.parser.set_language(language_map[self.language])
            else:
                logger.warning(f"Tree-sitter not available for {self.language}")
        except Exception as e:
            logger.error(f"Error initializing tree-sitter parser for {self.language}: {e}")

    async def chunk_code(
        self,
        content: str,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Language-aware code chunking using tree-sitter."""
        if not self.parser:
            # Fallback to basic chunking
            async for chunk in self._basic_chunk(content, max_chunk_size):
                yield chunk
            return

        try:
            tree = self.parser.parse(content.encode("utf-8"))

            # Get language-specific chunking strategy
            if self.language == Language.PYTHON:
                async for chunk in self._chunk_python(content, tree, max_chunk_size):
                    yield chunk
            elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                async for chunk in self._chunk_javascript(content, tree, max_chunk_size):
                    yield chunk
            elif self.language == Language.JAVA:
                async for chunk in self._chunk_java(content, tree, max_chunk_size):
                    yield chunk
            elif self.language in [Language.C, Language.CPP]:
                async for chunk in self._chunk_c_cpp(content, tree, max_chunk_size):
                    yield chunk
            elif self.language == Language.GO:
                async for chunk in self._chunk_go(content, tree, max_chunk_size):
                    yield chunk
            elif self.language == Language.RUST:
                async for chunk in self._chunk_rust(content, tree, max_chunk_size):
                    yield chunk
            else:
                # Generic tree-sitter chunking
                async for chunk in self._chunk_generic(content, tree, max_chunk_size):
                    yield chunk

        except Exception as e:
            logger.error(f"Error chunking {self.language} code: {e}")
            # Fallback to basic chunking
            async for chunk in self._basic_chunk(content, max_chunk_size):
                yield chunk

    async def _basic_chunk(
        self,
        content: str,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Basic line-based chunking fallback."""
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_size = 0
        start_line = 1

        for i, line in enumerate(lines, 1):
            line_size = len(line) + 1  # +1 for newline

            if current_size + line_size > max_chunk_size and current_chunk:
                # Yield current chunk
                chunk_content = "\n".join(current_chunk)
                yield CodeChunk(
                    content=chunk_content,
                    start_line=start_line,
                    end_line=i - 1,
                    language=self.language,
                    chunk_type="code",
                )

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
                start_line = i
            else:
                current_chunk.append(line)
                current_size += line_size

        # Yield final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            yield CodeChunk(
                content=chunk_content,
                start_line=start_line,
                end_line=len(lines),
                language=self.language,
                chunk_type="code",
            )

    async def _chunk_python(
        self,
        content: str,
        tree: Any,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Python-specific chunking strategy."""
        # Query for top-level definitions
        query = tree_sitter.Query(
            tree.language,
            """
            (function_definition) @function
            (class_definition) @class
            (import_statement) @import
            (import_from_statement) @import
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        # Process each top-level entity
        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                # Entity fits in one chunk
                entity_type = (
                    EntityType.FUNCTION
                    if capture_name == "function"
                    else (
                        EntityType.CLASS
                        if capture_name == "class"
                        else (
                            EntityType.IMPORT
                            if capture_name == "import"
                            else EntityType.UNKNOWN_ENTITY
                        )
                    )
                )

                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type,
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                # Large entity - chunk it further
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_large_entity(
        self,
        content: str,
        start_line: int,
        max_chunk_size: int,
        entity_type: str,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Chunk large entities that exceed max_chunk_size."""
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_size = 0
        chunk_start = start_line

        for i, line in enumerate(lines):
            line_size = len(line) + 1

            if current_size + line_size > max_chunk_size and current_chunk:
                # Yield current chunk
                chunk_content = "\n".join(current_chunk)
                yield CodeChunk(
                    content=chunk_content,
                    start_line=chunk_start,
                    end_line=start_line + i - 1,
                    language=self.language,
                    chunk_type=f"{entity_type}_part",
                )

                # Start new chunk
                current_chunk = [line]
                current_size = line_size
                chunk_start = start_line + i
            else:
                current_chunk.append(line)
                current_size += line_size

        # Yield final chunk
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            yield CodeChunk(
                content=chunk_content,
                start_line=chunk_start,
                end_line=start_line + len(lines) - 1,
                language=self.language,
                chunk_type=f"{entity_type}_part",
            )

    def _calculate_node_complexity(self, node: Any) -> float:
        """Calculate complexity score for a tree-sitter node."""
        if not node:
            return 0.0

        # Simple complexity based on node count and depth
        node_count = self._count_nodes(node)
        max_depth = self._calculate_depth(node)

        # Normalize to 0-1 range
        complexity = min(1.0, (node_count * 0.01) + (max_depth * 0.1))
        return complexity

    def _count_nodes(self, node: Any) -> int:
        """Count total nodes in tree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def _calculate_depth(self, node: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of tree."""
        if not node.children:
            return current_depth

        max_child_depth = 0
        for child in node.children:
            child_depth = self._calculate_depth(child, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth

    async def _chunk_javascript(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """JavaScript/TypeScript-specific chunking using tree-sitter."""
        query = tree_sitter.Query(
            tree.language,
            """
            (function_declaration) @function
            (class_declaration) @class
            (export_statement) @export
            (import_statement) @import
            (lexical_declaration) @variable
            (expression_statement (assignment_expression) @assignment)
            (arrow_function) @arrow
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "class": EntityType.CLASS,
                    "import": EntityType.IMPORT,
                    "export": EntityType.UNKNOWN_ENTITY,
                    "variable": EntityType.VARIABLE,
                    "assignment": EntityType.VARIABLE,
                    "arrow": EntityType.FUNCTION,
                }
                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_java(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """Java-specific chunking using tree-sitter."""
        query = tree_sitter.Query(
            tree.language,
            """
            (class_declaration) @class
            (method_declaration) @method
            (constructor_declaration) @constructor
            (import_declaration) @import
            (interface_declaration) @interface
            (field_declaration) @field
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                entity_type_map = {
                    "class": EntityType.CLASS,
                    "method": EntityType.METHOD,
                    "constructor": EntityType.METHOD,
                    "import": EntityType.IMPORT,
                    "interface": EntityType.INTERFACE,
                    "field": EntityType.VARIABLE,
                }
                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_c_cpp(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """C/C++-specific chunking using tree-sitter."""
        query = tree_sitter.Query(
            tree.language,
            """
            (function_definition) @function
            (struct_specifier) @struct
            (class_specifier) @class
            (preproc_include) @include
            (declaration) @declaration
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "struct": EntityType.STRUCT,
                    "class": EntityType.CLASS,
                    "include": EntityType.IMPORT,
                    "declaration": EntityType.VARIABLE,
                }
                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_go(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """Go-specific chunking using tree-sitter."""
        query = tree_sitter.Query(
            tree.language,
            """
            (function_declaration) @function
            (method_declaration) @method
            (type_declaration) @type
            (import_declaration) @import
            (var_declaration) @variable
            (const_declaration) @constant
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "method": EntityType.METHOD,
                    "type": EntityType.CLASS,
                    "import": EntityType.IMPORT,
                    "variable": EntityType.VARIABLE,
                    "constant": EntityType.CONSTANT,
                }
                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_rust(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """Rust-specific chunking using tree-sitter."""
        query = tree_sitter.Query(
            tree.language,
            """
            (function_item) @function
            (struct_item) @struct
            (impl_item) @impl
            (enum_item) @enum
            (trait_item) @trait
            (use_declaration) @import
            (mod_item) @module
            (const_item) @constant
        """,
        )

        captures = query.captures(tree.root_node)
        lines = content.split("\n")

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            entity_content = "\n".join(lines[start_line - 1 : end_line])

            if len(entity_content) <= max_chunk_size:
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "struct": EntityType.STRUCT,
                    "impl": EntityType.CLASS,
                    "enum": EntityType.ENUM,
                    "trait": EntityType.INTERFACE,
                    "import": EntityType.IMPORT,
                    "module": EntityType.MODULE,
                    "constant": EntityType.CONSTANT,
                }
                yield CodeChunk(
                    content=entity_content,
                    start_line=start_line,
                    end_line=end_line,
                    language=self.language,
                    chunk_type=capture_name,
                    entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                    complexity_score=self._calculate_node_complexity(node),
                )
            else:
                async for chunk in self._chunk_large_entity(
                    entity_content, start_line, max_chunk_size, capture_name
                ):
                    yield chunk

    async def _chunk_generic(
        self, content: str, tree: Any, max_chunk_size: int
    ) -> AsyncGenerator[CodeChunk, None]:
        """Generic tree-sitter based chunking for unsupported languages."""
        # Use top-level children as chunk boundaries
        lines = content.split("\n")
        root = tree.root_node

        current_chunk: list[str] = []
        current_size = 0
        chunk_start = 1

        for child in root.children:
            child_start = child.start_point[0] + 1
            child_end = child.end_point[0] + 1
            child_content = "\n".join(lines[child_start - 1 : child_end])
            child_size = len(child_content) + 1

            if current_size + child_size > max_chunk_size and current_chunk:
                yield CodeChunk(
                    content="\n".join(current_chunk),
                    start_line=chunk_start,
                    end_line=child_start - 1,
                    language=self.language,
                    chunk_type="code",
                )
                current_chunk = []
                current_size = 0
                chunk_start = child_start

            if child_size > max_chunk_size:
                # Child itself exceeds limit; emit accumulated then chunk child
                if current_chunk:
                    yield CodeChunk(
                        content="\n".join(current_chunk),
                        start_line=chunk_start,
                        end_line=child_start - 1,
                        language=self.language,
                        chunk_type="code",
                    )
                    current_chunk = []
                    current_size = 0

                async for sub_chunk in self._chunk_large_entity(
                    child_content, child_start, max_chunk_size, "code"
                ):
                    yield sub_chunk
                chunk_start = child_end + 1
            else:
                current_chunk.append(child_content)
                current_size += child_size

        if current_chunk:
            yield CodeChunk(
                content="\n".join(current_chunk),
                start_line=chunk_start,
                end_line=len(lines),
                language=self.language,
                chunk_type="code",
            )

    def extract_entities(self, content: str) -> list[CodeEntity]:
        """Extract code entities using tree-sitter."""
        if not self.parser:
            return []

        try:
            tree = self.parser.parse(content.encode("utf-8"))
            entities = []

            if self.language == Language.PYTHON:
                entities.extend(self._extract_python_entities(content, tree))
            elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                entities.extend(self._extract_javascript_entities(content, tree))
            elif self.language == Language.JAVA:
                entities.extend(self._extract_java_entities(content, tree))
            elif self.language in [Language.C, Language.CPP]:
                entities.extend(self._extract_c_cpp_entities(content, tree))
            elif self.language == Language.GO:
                entities.extend(self._extract_go_entities(content, tree))
            elif self.language == Language.RUST:
                entities.extend(self._extract_rust_entities(content, tree))
            elif self.language == Language.CSHARP:
                entities.extend(self._extract_csharp_entities(content, tree))
            elif self.language == Language.PHP:
                entities.extend(self._extract_php_entities(content, tree))
            elif self.language == Language.RUBY:
                entities.extend(self._extract_ruby_entities(content, tree))

            return entities
        except Exception as e:
            logger.error(f"Error extracting entities for {self.language}: {e}")
            return []

    def _extract_python_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract Python entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        # Query for Python entities
        query = tree_sitter.Query(
            tree.language,
            """
            (function_definition name: (identifier) @func_name) @function
            (class_definition name: (identifier) @class_name) @class
            (assignment left: (identifier) @var_name) @variable
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "class", "variable"]:
                # Find the name node
                name_node = None
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break

                if name_node:
                    entity_name = name_node.text.decode("utf-8")
                    entity_type = (
                        EntityType.FUNCTION
                        if capture_name == "function"
                        else EntityType.CLASS if capture_name == "class" else EntityType.VARIABLE
                    )

                    entities.append(
                        CodeEntity(
                            id=f"{entity_name}_{start_line}_{end_line}",
                            name=entity_name,
                            entity_type=entity_type,
                            file_path=Path(""),  # Will be set by caller
                            start_line=start_line,
                            end_line=end_line,
                            language=self.language,
                            signature=self._extract_signature(node, lines),
                            docstring=self._extract_docstring(node, lines),
                        )
                    )

        return entities

    def _extract_javascript_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract JavaScript/TypeScript entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (function_declaration name: (identifier) @func_name) @function
            (class_declaration name: (identifier) @class_name) @class
            (lexical_declaration (variable_declarator name: (identifier) @var_name)) @variable
            (import_statement) @import
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "class", "variable", "import"]:
                # Find the name node
                name_node = None
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break
                    # For lexical_declaration, name is deeper
                    if child.type == "variable_declarator":
                        for sub_child in child.children:
                            if sub_child.type == "identifier":
                                name_node = sub_child
                                break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type = (
                    EntityType.FUNCTION
                    if capture_name == "function"
                    else EntityType.CLASS
                    if capture_name == "class"
                    else EntityType.IMPORT
                    if capture_name == "import"
                    else EntityType.VARIABLE
                )

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type,
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature(node, lines),
                    )
                )

        return entities

    def _extract_java_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract Java entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (class_declaration name: (identifier) @class_name) @class
            (method_declaration name: (identifier) @method_name) @method
            (constructor_declaration name: (identifier) @ctor_name) @constructor
            (interface_declaration name: (identifier) @iface_name) @interface
            (field_declaration declarator: (variable_declarator name: (identifier) @field_name)) @field
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["class", "method", "constructor", "interface", "field"]:
                name_node = None
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break
                    if child.type == "variable_declarator":
                        for sub in child.children:
                            if sub.type == "identifier":
                                name_node = sub
                                break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "class": EntityType.CLASS,
                    "method": EntityType.METHOD,
                    "constructor": EntityType.METHOD,
                    "interface": EntityType.INTERFACE,
                    "field": EntityType.VARIABLE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature(node, lines),
                    )
                )

        return entities

    def _extract_c_cpp_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract C/C++ entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (function_definition declarator: (function_declarator declarator: (identifier) @func_name)) @function
            (struct_specifier name: (type_identifier) @struct_name) @struct
            (declaration declarator: (init_declarator declarator: (identifier) @var_name)) @variable
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "struct", "variable"]:
                name_node = None
                # For function_definition, name is nested deeper
                if capture_name == "function":
                    for child in node.children:
                        if child.type == "function_declarator":
                            for sub in child.children:
                                if sub.type == "identifier":
                                    name_node = sub
                                    break
                elif capture_name == "struct":
                    for child in node.children:
                        if child.type == "type_identifier":
                            name_node = child
                            break
                else:
                    for child in node.children:
                        if child.type == "init_declarator":
                            for sub in child.children:
                                if sub.type == "identifier":
                                    name_node = sub
                                    break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "struct": EntityType.STRUCT,
                    "variable": EntityType.VARIABLE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature(node, lines),
                    )
                )

        return entities

    def _extract_go_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract Go entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (function_declaration name: (identifier) @func_name) @function
            (method_declaration name: (field_identifier) @method_name) @method
            (type_declaration (type_spec name: (type_identifier) @type_name)) @type
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "method", "type"]:
                name_node = None
                if capture_name == "function":
                    for child in node.children:
                        if child.type == "identifier":
                            name_node = child
                            break
                elif capture_name == "method":
                    for child in node.children:
                        if child.type == "field_identifier":
                            name_node = child
                            break
                elif capture_name == "type":
                    for child in node.children:
                        if child.type == "type_spec":
                            for sub in child.children:
                                if sub.type == "type_identifier":
                                    name_node = sub
                                    break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "method": EntityType.METHOD,
                    "type": EntityType.CLASS,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature(node, lines),
                    )
                )

        return entities

    def _extract_rust_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract Rust entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (function_item name: (identifier) @func_name) @function
            (struct_item name: (type_identifier) @struct_name) @struct
            (impl_item type: (type_identifier) @impl_name) @impl
            (enum_item name: (type_identifier) @enum_name) @enum
            (trait_item name: (type_identifier) @trait_name) @trait
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "struct", "impl", "enum", "trait"]:
                name_node = None
                name_types = {
                    "function": "identifier",
                    "struct": "type_identifier",
                    "impl": "type_identifier",
                    "enum": "type_identifier",
                    "trait": "type_identifier",
                }
                target_type = name_types.get(capture_name, "identifier")
                for child in node.children:
                    if child.type == target_type:
                        name_node = child
                        break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "struct": EntityType.STRUCT,
                    "impl": EntityType.CLASS,
                    "enum": EntityType.ENUM,
                    "trait": EntityType.INTERFACE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature(node, lines),
                    )
                )

        return entities

    def _extract_csharp_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract C# entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (class_declaration name: (identifier) @class_name) @class
            (method_declaration name: (identifier) @method_name) @method
            (interface_declaration name: (identifier) @iface_name) @interface
            (struct_declaration name: (identifier) @struct_name) @struct
            (enum_declaration name: (identifier) @enum_name) @enum
            (namespace_declaration name: (identifier) @ns_name) @namespace
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["class", "method", "interface", "struct", "enum", "namespace"]:
                name_node = None
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "class": EntityType.CLASS,
                    "method": EntityType.METHOD,
                    "interface": EntityType.INTERFACE,
                    "struct": EntityType.STRUCT,
                    "enum": EntityType.ENUM,
                    "namespace": EntityType.NAMESPACE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature_generic(node, lines),
                    )
                )

        return entities

    def _extract_php_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract PHP entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (function_definition name: (name) @func_name) @function
            (class_declaration name: (name) @class_name) @class
            (method_declaration name: (name) @method_name) @method
            (interface_declaration name: (name) @iface_name) @interface
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["function", "class", "method", "interface"]:
                name_node = None
                for child in node.children:
                    if child.type == "name":
                        name_node = child
                        break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "function": EntityType.FUNCTION,
                    "class": EntityType.CLASS,
                    "method": EntityType.METHOD,
                    "interface": EntityType.INTERFACE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature_generic(node, lines),
                    )
                )

        return entities

    def _extract_ruby_entities(self, content: str, tree: Any) -> list[CodeEntity]:
        """Extract Ruby entities using tree-sitter."""
        entities = []
        lines = content.split("\n")

        query = tree_sitter.Query(
            tree.language,
            """
            (method name: (identifier) @method_name) @method
            (class name: (constant) @class_name) @class
            (module name: (constant) @module_name) @module
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1

            if capture_name in ["method", "class", "module"]:
                name_node = None
                target_types = {
                    "method": "identifier",
                    "class": "constant",
                    "module": "constant",
                }
                target = target_types.get(capture_name, "identifier")
                for child in node.children:
                    if child.type == target:
                        name_node = child
                        break

                entity_name = name_node.text.decode("utf-8") if name_node else f"anonymous_{start_line}"
                entity_type_map = {
                    "method": EntityType.METHOD,
                    "class": EntityType.CLASS,
                    "module": EntityType.MODULE,
                }

                entities.append(
                    CodeEntity(
                        id=f"{entity_name}_{start_line}_{end_line}",
                        name=entity_name,
                        entity_type=entity_type_map.get(capture_name, EntityType.UNKNOWN_ENTITY),
                        file_path=Path(""),
                        start_line=start_line,
                        end_line=end_line,
                        language=self.language,
                        signature=self._extract_signature_generic(node, lines),
                    )
                )

        return entities

    def _extract_signature_generic(self, node: Any, lines: list[str]) -> str | None:
        """Extract signature from first line of a node (language-agnostic)."""
        try:
            start_line = node.start_point[0]
            if start_line < len(lines):
                sig = lines[start_line].strip()
                # Truncate at opening brace for curly-brace languages
                if "{" in sig:
                    sig = sig.split("{")[0].strip()
                return sig
        except Exception:
            pass
        return None

    def _extract_signature(self, node: Any, lines: list[str]) -> str | None:
        """Extract function/class signature."""
        try:
            # For functions, get the definition line
            start_line = node.start_point[0]
            if start_line < len(lines):
                signature_line: str = lines[start_line].strip()
                # Remove body for functions
                if ":" in signature_line:
                    signature_line = signature_line.split(":")[0] + ":"
                return signature_line
        except Exception:
            pass
        return None

    def _extract_docstring(self, node: Any, lines: list[str]) -> str | None:
        """Extract docstring for functions/classes."""
        try:
            # Look for string literal as first statement in body
            for child in node.children:
                if child.type == "block":
                    for stmt in child.children:
                        if stmt.type == "expression_statement":
                            for expr_child in stmt.children:
                                if expr_child.type == "string":
                                    docstring_text: str = expr_child.text.decode("utf-8").strip(
                                        "\"'"
                                    )
                                    return docstring_text
        except Exception:
            pass
        return None

    def analyze_dependencies(self, content: str) -> list[str]:
        """Analyze dependencies using tree-sitter."""
        if not self.parser:
            return self._analyze_dependencies_regex(content)

        try:
            tree = self.parser.parse(content.encode("utf-8"))
            dependencies = []

            if self.language == Language.PYTHON:
                dependencies.extend(self._analyze_python_dependencies(tree))
            elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                dependencies.extend(self._analyze_javascript_dependencies(tree))
            elif self.language == Language.JAVA:
                dependencies.extend(self._analyze_java_dependencies(tree))
            elif self.language in [Language.C, Language.CPP]:
                dependencies.extend(self._analyze_c_cpp_dependencies(tree))
            elif self.language == Language.GO:
                dependencies.extend(self._analyze_go_dependencies(tree))
            elif self.language == Language.RUST:
                dependencies.extend(self._analyze_rust_dependencies(tree))
            elif self.language == Language.CSHARP:
                dependencies.extend(self._analyze_csharp_dependencies(tree))
            elif self.language == Language.PHP:
                dependencies.extend(self._analyze_php_dependencies(tree))
            elif self.language == Language.RUBY:
                dependencies.extend(self._analyze_ruby_dependencies(tree))

            return dependencies
        except Exception as e:
            logger.error(f"Error analyzing dependencies for {self.language}: {e}")
            return self._analyze_dependencies_regex(content)

    def _analyze_python_dependencies(self, tree: Any) -> list[str]:
        """Analyze Python dependencies using tree-sitter."""
        dependencies = []

        # Query for imports
        query = tree_sitter.Query(
            tree.language,
            """
            (import_statement name: (dotted_name) @import)
            (import_from_statement module_name: (dotted_name) @from_import)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_javascript_dependencies(self, tree: Any) -> list[str]:
        """Analyze JavaScript/TypeScript dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (import_statement source: (string) @source)
            (call_expression
                function: (identifier) @func_name
                arguments: (arguments (string) @arg)
                (#eq? @func_name "require"))
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            if capture_name == "source":
                # Strip quotes from the import path
                dep_name = node.text.decode("utf-8").strip("'\"")
                dependencies.append(dep_name)
            elif capture_name == "arg":
                dep_name = node.text.decode("utf-8").strip("'\"")
                dependencies.append(dep_name)

        return dependencies

    def _analyze_java_dependencies(self, tree: Any) -> list[str]:
        """Analyze Java dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (import_declaration (scoped_identifier) @import)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_c_cpp_dependencies(self, tree: Any) -> list[str]:
        """Analyze C/C++ dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (preproc_include path: (string_literal) @include_path)
            (preproc_include path: (system_lib_string) @system_include)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8").strip("\"'<>")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_go_dependencies(self, tree: Any) -> list[str]:
        """Analyze Go dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (import_spec path: (interpreted_string_literal) @import_path)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8").strip('"')
            dependencies.append(dep_name)

        return dependencies

    def _analyze_rust_dependencies(self, tree: Any) -> list[str]:
        """Analyze Rust dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (use_declaration argument: (scoped_identifier) @use_path)
            (use_declaration argument: (identifier) @use_ident)
            (use_declaration argument: (scoped_use_list) @use_list)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_csharp_dependencies(self, tree: Any) -> list[str]:
        """Analyze C# dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (using_directive (identifier) @using)
            (using_directive (qualified_name) @using_qualified)
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_php_dependencies(self, tree: Any) -> list[str]:
        """Analyze PHP dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (namespace_use_declaration (namespace_use_clause (qualified_name) @use))
        """,
        )

        captures = query.captures(tree.root_node)

        for node, _capture_name in captures:
            dep_name = node.text.decode("utf-8")
            dependencies.append(dep_name)

        return dependencies

    def _analyze_ruby_dependencies(self, tree: Any) -> list[str]:
        """Analyze Ruby dependencies using tree-sitter."""
        dependencies = []

        query = tree_sitter.Query(
            tree.language,
            """
            (call method: (identifier) @method_name
                 arguments: (argument_list (string (string_content) @arg))
                 (#match? @method_name "^(require|require_relative|load)$"))
        """,
        )

        captures = query.captures(tree.root_node)

        for node, capture_name in captures:
            if capture_name == "arg":
                dep_name = node.text.decode("utf-8")
                dependencies.append(dep_name)

        return dependencies

    def _analyze_dependencies_regex(self, content: str) -> list[str]:
        """Fallback regex-based dependency analysis."""
        dependencies: list[str] = []
        import_patterns: list[str] = []

        if self.language == Language.PYTHON:
            import_patterns = [
                r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)",
                r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import",
            ]
        elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            import_patterns = [
                r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]',
                r'^\s*const.*=\s*require\([\'"]([^\'"]+)[\'"]\)',
            ]
        elif self.language == Language.JAVA:
            import_patterns = [
                r"^\s*import\s+(?:static\s+)?([a-zA-Z_][\w.]*)\s*;",
            ]
        elif self.language in [Language.C, Language.CPP]:
            import_patterns = [
                r'^\s*#include\s*[<"]([^>"]+)[>"]',
            ]
        elif self.language == Language.GO:
            import_patterns = [
                r'^\s*"([^"]+)"',
            ]
        elif self.language == Language.RUST:
            import_patterns = [
                r"^\s*use\s+([\w:]+)",
                r"^\s*extern\s+crate\s+(\w+)",
            ]
        elif self.language == Language.CSHARP:
            import_patterns = [
                r"^\s*using\s+([a-zA-Z_][\w.]*)\s*;",
            ]
        elif self.language == Language.PHP:
            import_patterns = [
                r"^\s*use\s+([a-zA-Z_\\][\w\\]*)",
                r'^\s*(?:require|include)(?:_once)?\s*[\(]?\s*[\'"]([^\'"]+)[\'"]',
            ]
        elif self.language == Language.RUBY:
            import_patterns = [
                r'^\s*require\s+[\'"]([^\'"]+)[\'"]',
                r'^\s*require_relative\s+[\'"]([^\'"]+)[\'"]',
            ]
        elif self.language == Language.KOTLIN:
            import_patterns = [
                r"^\s*import\s+([a-zA-Z_][\w.]*)",
            ]
        elif self.language == Language.SWIFT:
            import_patterns = [
                r"^\s*import\s+(\w+)",
            ]
        elif self.language == Language.SCALA:
            import_patterns = [
                r"^\s*import\s+([a-zA-Z_][\w.]*)",
            ]
        elif self.language == Language.DART:
            import_patterns = [
                r'^\s*import\s+[\'"]([^\'"]+)[\'"]',
            ]
        elif self.language == Language.LUA:
            import_patterns = [
                r'^\s*(?:local\s+\w+\s*=\s*)?require\s*[\(]?\s*[\'"]([^\'"]+)[\'"]',
            ]
        elif self.language == Language.PERL:
            import_patterns = [
                r"^\s*use\s+([A-Z]\w+(?:::\w+)*)",
                r'^\s*require\s+[\'"]?([^\'";\s]+)',
            ]
        elif self.language == Language.ELIXIR:
            import_patterns = [
                r"^\s*(?:import|use|alias|require)\s+(\w[\w.]*)",
            ]
        elif self.language == Language.HASKELL:
            import_patterns = [
                r"^\s*import\s+(?:qualified\s+)?([A-Z]\w+(?:\.\w+)*)",
            ]
        elif self.language == Language.JULIA:
            import_patterns = [
                r"^\s*(?:using|import)\s+(\w[\w.]*)",
            ]
        elif self.language == Language.GROOVY:
            import_patterns = [
                r"^\s*import\s+([a-zA-Z_][\w.]*)",
            ]
        else:
            return dependencies

        for pattern in import_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                dependencies.append(match.group(1))

        return dependencies

    def calculate_complexity(self, content: str) -> float:
        """Calculate code complexity score."""
        if not self.parser:
            return self._calculate_complexity_basic(content)

        try:
            tree = self.parser.parse(content.encode("utf-8"))

            # Count various complexity indicators
            node_count = self._count_nodes(tree.root_node)
            max_depth = self._calculate_depth(tree.root_node)

            # Language-specific complexity factors
            if self.language == Language.PYTHON:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_statement", "for_statement", "while_statement", "try_statement",
                     "except_clause", "with_statement", "assert_statement",
                     "raise_statement", "boolean_operator"},
                )
            elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_statement", "for_statement", "for_in_statement", "while_statement",
                     "do_statement", "switch_statement", "try_statement", "catch_clause",
                     "ternary_expression", "binary_expression"},
                )
            elif self.language == Language.JAVA:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_statement", "for_statement", "enhanced_for_statement",
                     "while_statement", "do_statement", "switch_expression",
                     "try_statement", "catch_clause", "ternary_expression",
                     "throw_statement"},
                )
            elif self.language in [Language.C, Language.CPP]:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_statement", "for_statement", "while_statement", "do_statement",
                     "switch_statement", "case_statement", "conditional_expression",
                     "goto_statement"},
                )
            elif self.language == Language.GO:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_statement", "for_statement", "expression_switch_statement",
                     "type_switch_statement", "select_statement", "go_statement",
                     "defer_statement"},
                )
            elif self.language == Language.RUST:
                complexity_nodes = self._count_complexity_nodes(
                    tree.root_node,
                    {"if_expression", "for_expression", "while_expression", "loop_expression",
                     "match_expression", "match_arm", "if_let_expression",
                     "while_let_expression"},
                )
            else:
                complexity_nodes = 0

            # Normalize to 0-1 range
            base_complexity = min(1.0, node_count / 1000.0)
            depth_complexity = min(1.0, max_depth / 20.0)
            specific_complexity = min(1.0, complexity_nodes / 50.0)

            return (base_complexity + depth_complexity + specific_complexity) / 3.0

        except Exception as e:
            logger.error(f"Error calculating complexity: {e}")
            return self._calculate_complexity_basic(content)

    def _calculate_complexity_basic(self, content: str) -> float:
        """Basic complexity calculation without tree-sitter."""
        lines = content.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]

        # Simple heuristic based on line count and control structures
        control_keywords = ["if", "for", "while", "try", "except", "with", "def", "class"]
        control_count = sum(
            1 for line in non_empty_lines for keyword in control_keywords if keyword in line
        )

        # Normalize to 0-1 range
        line_complexity = min(1.0, len(non_empty_lines) / 100.0)
        control_complexity = min(1.0, control_count / 20.0)

        return (line_complexity + control_complexity) / 2.0

    def _count_complexity_nodes(self, node: Any, complexity_types: set[str]) -> int:
        """Count language-specific complexity indicator nodes recursively."""
        count = 0
        if node.type in complexity_types:
            count += 1

        for child in node.children:
            count += self._count_complexity_nodes(child, complexity_types)

        return count


@dataclass
class RegexLanguageConfig:
    """Configuration for regex-based language processing."""

    function_patterns: list[str] = field(default_factory=list)
    class_patterns: list[str] = field(default_factory=list)
    import_patterns: list[str] = field(default_factory=list)
    struct_patterns: list[str] = field(default_factory=list)
    module_patterns: list[str] = field(default_factory=list)
    interface_patterns: list[str] = field(default_factory=list)
    comment_single: str = "//"
    comment_multi_start: str = "/*"
    comment_multi_end: str = "*/"
    block_end_pattern: str | None = None  # e.g. "end" for Ruby/Elixir


# Pre-defined regex configs for languages without tree-sitter
REGEX_LANGUAGE_CONFIGS: dict[Language, RegexLanguageConfig] = {
    Language.KOTLIN: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:(?:public|private|protected|internal|override|open|abstract|suspend)\s+)*fun\s+(\w+)"],
        class_patterns=[r"^\s*(?:(?:public|private|protected|internal|open|abstract|data|sealed|enum|annotation)\s+)*class\s+(\w+)"],
        import_patterns=[r"^\s*import\s+([\w.]+)"],
        interface_patterns=[r"^\s*(?:(?:public|private|protected|internal)\s+)*interface\s+(\w+)"],
    ),
    Language.SWIFT: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:(?:public|private|internal|open|fileprivate|static|class|override|mutating)\s+)*func\s+(\w+)"],
        class_patterns=[r"^\s*(?:(?:public|private|internal|open|fileprivate|final)\s+)*class\s+(\w+)"],
        import_patterns=[r"^\s*import\s+(\w+)"],
        struct_patterns=[r"^\s*(?:(?:public|private|internal)\s+)*struct\s+(\w+)"],
        interface_patterns=[r"^\s*(?:(?:public|private|internal)\s+)*protocol\s+(\w+)"],
    ),
    Language.SCALA: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:(?:override|private|protected)\s+)*def\s+(\w+)"],
        class_patterns=[r"^\s*(?:(?:abstract|sealed|final|case|implicit)\s+)*class\s+(\w+)"],
        import_patterns=[r"^\s*import\s+([\w.]+)"],
        module_patterns=[r"^\s*(?:case\s+)?object\s+(\w+)"],
        interface_patterns=[r"^\s*(?:sealed\s+)?trait\s+(\w+)"],
    ),
    Language.SHELL: RegexLanguageConfig(
        function_patterns=[
            r"^\s*(\w+)\s*\(\)\s*\{",
            r"^\s*function\s+(\w+)",
        ],
        import_patterns=[
            r"^\s*(?:source|\.)\s+(.+)",
        ],
        comment_single="#",
        comment_multi_start="",
        comment_multi_end="",
    ),
    Language.POWERSHELL: RegexLanguageConfig(
        function_patterns=[r"^\s*function\s+(\w[\w-]*)"],
        class_patterns=[r"^\s*class\s+(\w+)"],
        import_patterns=[
            r"^\s*(?:Import-Module|using\s+module)\s+(\S+)",
        ],
        comment_single="#",
        comment_multi_start="<#",
        comment_multi_end="#>",
    ),
    Language.LUA: RegexLanguageConfig(
        function_patterns=[
            r"^\s*function\s+([\w.]+)\s*\(",
            r"^\s*local\s+function\s+(\w+)\s*\(",
        ],
        import_patterns=[
            r'^\s*(?:local\s+\w+\s*=\s*)?require\s*[\(]?\s*[\'"]([^\'"]+)[\'"]',
        ],
        comment_single="--",
        comment_multi_start="--[[",
        comment_multi_end="]]",
        block_end_pattern="end",
    ),
    Language.PERL: RegexLanguageConfig(
        function_patterns=[r"^\s*sub\s+(\w+)"],
        import_patterns=[
            r"^\s*use\s+([A-Z]\w+(?:::\w+)*)",
            r'^\s*require\s+[\'"]?([^\'";\s]+)',
        ],
        comment_single="#",
        comment_multi_start="=pod",
        comment_multi_end="=cut",
    ),
    Language.DART: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:(?:static|async|Future|void|int|double|String|bool|dynamic)\s+)+(\w+)\s*\("],
        class_patterns=[r"^\s*(?:abstract\s+)?class\s+(\w+)"],
        import_patterns=[r'^\s*import\s+[\'"]([^\'"]+)[\'"]'],
        interface_patterns=[r"^\s*(?:abstract\s+)?class\s+(\w+)"],
    ),
    Language.ELIXIR: RegexLanguageConfig(
        function_patterns=[
            r"^\s*def\s+(\w+)",
            r"^\s*defp\s+(\w+)",
        ],
        class_patterns=[r"^\s*defmodule\s+([\w.]+)"],
        import_patterns=[
            r"^\s*(?:import|use|alias|require)\s+([\w.]+)",
        ],
        comment_single="#",
        comment_multi_start="",
        comment_multi_end="",
        block_end_pattern="end",
    ),
    Language.HASKELL: RegexLanguageConfig(
        function_patterns=[r"^(\w+)\s+::\s+"],
        class_patterns=[r"^\s*(?:data|newtype|type)\s+(\w+)"],
        import_patterns=[r"^\s*import\s+(?:qualified\s+)?([A-Z]\w+(?:\.\w+)*)"],
        comment_single="--",
        comment_multi_start="{-",
        comment_multi_end="-}",
    ),
    Language.JULIA: RegexLanguageConfig(
        function_patterns=[r"^\s*function\s+(\w+)"],
        class_patterns=[r"^\s*(?:mutable\s+)?struct\s+(\w+)"],
        import_patterns=[r"^\s*(?:using|import)\s+([\w.]+)"],
        module_patterns=[r"^\s*module\s+(\w+)"],
        block_end_pattern="end",
    ),
    Language.GROOVY: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:(?:public|private|protected|static|def)\s+)*(?:def\s+)?(\w+)\s*\("],
        class_patterns=[r"^\s*(?:(?:public|private|protected|abstract)\s+)*class\s+(\w+)"],
        import_patterns=[r"^\s*import\s+([\w.]+)"],
        interface_patterns=[r"^\s*(?:(?:public|private|protected)\s+)*interface\s+(\w+)"],
    ),
    Language.OBJECTIVE_C: RegexLanguageConfig(
        function_patterns=[r"^\s*[-+]\s*\([^)]*\)\s*(\w+)"],
        class_patterns=[r"^\s*@(?:interface|implementation)\s+(\w+)"],
        import_patterns=[r'^\s*#import\s*[<"]([^>"]+)[>"]'],
        interface_patterns=[r"^\s*@protocol\s+(\w+)"],
    ),
    Language.ZIG: RegexLanguageConfig(
        function_patterns=[r"^\s*(?:pub\s+)?fn\s+(\w+)"],
        struct_patterns=[r"^\s*(?:pub\s+)?const\s+(\w+)\s*=\s*(?:packed\s+)?struct"],
        import_patterns=[r'^\s*const\s+\w+\s*=\s*@import\s*\(\s*"([^"]+)"\s*\)'],
    ),
    Language.R: RegexLanguageConfig(
        function_patterns=[r"^\s*(\w+)\s*<-\s*function\s*\("],
        import_patterns=[
            r"^\s*library\s*\(\s*(\w+)\s*\)",
            r"^\s*require\s*\(\s*(\w+)\s*\)",
        ],
        comment_single="#",
        comment_multi_start="",
        comment_multi_end="",
    ),
    Language.MATLAB: RegexLanguageConfig(
        function_patterns=[r"^\s*function\s+(?:.*=\s*)?(\w+)\s*\("],
        class_patterns=[r"^\s*classdef\s+(\w+)"],
        import_patterns=[],
        comment_single="%",
        comment_multi_start="%{",
        comment_multi_end="%}",
        block_end_pattern="end",
    ),
    Language.SQL: RegexLanguageConfig(
        function_patterns=[
            r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:FUNCTION|PROCEDURE)\s+(\w+)",
        ],
        class_patterns=[
            r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)",
        ],
        import_patterns=[],
        comment_single="--",
        comment_multi_start="/*",
        comment_multi_end="*/",
    ),
}


class RegexProcessor(LanguageProcessor):
    """Regex-based language processor for languages without tree-sitter support."""

    def __init__(
        self,
        language: Language,
        config: LanguageConfig,
        regex_config: RegexLanguageConfig | None = None,
    ):
        super().__init__(language, config)
        self.regex_config = regex_config or REGEX_LANGUAGE_CONFIGS.get(
            language, RegexLanguageConfig()
        )

    async def chunk_code(
        self,
        content: str,
        max_chunk_size: int,
    ) -> AsyncGenerator[CodeChunk, None]:
        """Regex-based code chunking using function/class boundaries."""
        lines = content.split("\n")
        # Collect boundary line numbers
        boundaries: list[int] = [0]

        all_patterns = (
            self.regex_config.function_patterns
            + self.regex_config.class_patterns
            + self.regex_config.struct_patterns
            + self.regex_config.module_patterns
            + self.regex_config.interface_patterns
        )

        for i, line in enumerate(lines):
            for pattern in all_patterns:
                if re.search(pattern, line):
                    if i > 0 and i not in boundaries:
                        boundaries.append(i)
                    break

        boundaries.append(len(lines))
        boundaries.sort()

        # Build chunks from boundary segments
        current_chunk: list[str] = []
        current_size = 0
        chunk_start = 1

        for i in range(len(boundaries) - 1):
            segment = lines[boundaries[i] : boundaries[i + 1]]
            segment_text = "\n".join(segment)
            segment_size = len(segment_text) + 1

            if current_size + segment_size > max_chunk_size and current_chunk:
                yield CodeChunk(
                    content="\n".join(current_chunk),
                    start_line=chunk_start,
                    end_line=boundaries[i],
                    language=self.language,
                    chunk_type="code",
                )
                current_chunk = segment
                current_size = segment_size
                chunk_start = boundaries[i] + 1
            else:
                current_chunk.extend(segment)
                current_size += segment_size

        if current_chunk:
            yield CodeChunk(
                content="\n".join(current_chunk),
                start_line=chunk_start,
                end_line=len(lines),
                language=self.language,
                chunk_type="code",
            )

    def extract_entities(self, content: str) -> list[CodeEntity]:
        """Extract code entities using regex patterns."""
        entities: list[CodeEntity] = []
        lines = content.split("\n")

        pattern_map: list[tuple[list[str], EntityType]] = [
            (self.regex_config.function_patterns, EntityType.FUNCTION),
            (self.regex_config.class_patterns, EntityType.CLASS),
            (self.regex_config.struct_patterns, EntityType.STRUCT),
            (self.regex_config.module_patterns, EntityType.MODULE),
            (self.regex_config.interface_patterns, EntityType.INTERFACE),
        ]

        for line_num, line in enumerate(lines, 1):
            for patterns, entity_type in pattern_map:
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        entity_name = match.group(1)
                        # Estimate end line by looking for block end or next entity
                        end_line = self._find_block_end(lines, line_num - 1)
                        entities.append(
                            CodeEntity(
                                id=f"{entity_name}_{line_num}_{end_line}",
                                name=entity_name,
                                entity_type=entity_type,
                                file_path=Path(""),
                                start_line=line_num,
                                end_line=end_line,
                                language=self.language,
                                signature=line.strip(),
                            )
                        )
                        break  # Only match one pattern per line

        return entities

    def _find_block_end(self, lines: list[str], start_idx: int) -> int:
        """Estimate block end line using indentation or block-end keywords."""
        if self.regex_config.block_end_pattern:
            # For languages with explicit end keywords (Ruby, Lua, Elixir, etc.)
            depth = 1
            block_start_keywords = {"def", "class", "module", "do", "if", "for", "while",
                                    "function", "defmodule", "defp", "struct"}
            for i in range(start_idx + 1, min(start_idx + 500, len(lines))):
                stripped = lines[i].strip()
                # Check for nested blocks
                for kw in block_start_keywords:
                    if stripped.startswith(kw + " ") or stripped.startswith(kw + "("):
                        depth += 1
                        break
                if stripped == self.regex_config.block_end_pattern or stripped.startswith(
                    self.regex_config.block_end_pattern + " "
                ):
                    depth -= 1
                    if depth <= 0:
                        return i + 1
            return min(start_idx + 50, len(lines))
        else:
            # For curly-brace languages, track brace depth
            depth = 0
            found_open = False
            for i in range(start_idx, min(start_idx + 500, len(lines))):
                for ch in lines[i]:
                    if ch == "{":
                        depth += 1
                        found_open = True
                    elif ch == "}":
                        depth -= 1
                        if found_open and depth <= 0:
                            return i + 1
            # Fallback: use indentation
            if start_idx < len(lines):
                base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
                for i in range(start_idx + 1, min(start_idx + 200, len(lines))):
                    if lines[i].strip():
                        current_indent = len(lines[i]) - len(lines[i].lstrip())
                        if current_indent <= base_indent and i > start_idx + 1:
                            return i
            return min(start_idx + 50, len(lines))

    def analyze_dependencies(self, content: str) -> list[str]:
        """Analyze dependencies using regex patterns."""
        dependencies: list[str] = []
        for pattern in self.regex_config.import_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                dependencies.append(match.group(1))
        return dependencies

    def calculate_complexity(self, content: str) -> float:
        """Calculate code complexity using heuristics."""
        lines = content.split("\n")
        non_empty = [line for line in lines if line.strip()]

        control_keywords = [
            "if", "else", "elif", "for", "while", "switch", "case",
            "try", "catch", "except", "finally", "match", "when",
            "do", "loop", "foreach", "unless", "until",
        ]
        control_count = sum(
            1
            for line in non_empty
            for kw in control_keywords
            if re.search(rf"\b{kw}\b", line)
        )

        line_complexity = min(1.0, len(non_empty) / 100.0)
        control_complexity = min(1.0, control_count / 20.0)
        return (line_complexity + control_complexity) / 2.0


class LanguageRegistry:
    """Registry of all supported language processors."""

    def __init__(self) -> None:
        self.processors: dict[Language, LanguageProcessor] = {}
        self._initialize_processors()

    def _initialize_processors(self) -> None:
        """Initialize processors for all supported languages."""
        default_config = LanguageConfig()

        # Languages with tree-sitter support
        tree_sitter_languages = [
            Language.PYTHON,
            Language.JAVASCRIPT,
            Language.TYPESCRIPT,
            Language.JAVA,
            Language.C,
            Language.CPP,
            Language.GO,
            Language.RUST,
        ]

        # Conditionally add tree-sitter languages based on availability
        if TREE_SITTER_CSHARP_AVAILABLE:
            tree_sitter_languages.append(Language.CSHARP)
        if TREE_SITTER_PHP_AVAILABLE:
            tree_sitter_languages.append(Language.PHP)
        if TREE_SITTER_RUBY_AVAILABLE:
            tree_sitter_languages.append(Language.RUBY)

        for language in tree_sitter_languages:
            self.processors[language] = TreeSitterProcessor(language, default_config)

        # Register regex-based processors for remaining languages
        for language, regex_config in REGEX_LANGUAGE_CONFIGS.items():
            if language not in self.processors:
                self.processors[language] = RegexProcessor(language, default_config, regex_config)

        # Also register C#/PHP/Ruby as regex fallback if tree-sitter not available
        if not TREE_SITTER_CSHARP_AVAILABLE and Language.CSHARP not in self.processors:
            self.processors[Language.CSHARP] = RegexProcessor(
                Language.CSHARP,
                default_config,
                RegexLanguageConfig(
                    function_patterns=[r"^\s*(?:(?:public|private|protected|internal|static|virtual|override|abstract|async)\s+)*\w+\s+(\w+)\s*\("],
                    class_patterns=[r"^\s*(?:(?:public|private|protected|internal|static|abstract|sealed|partial)\s+)*class\s+(\w+)"],
                    import_patterns=[r"^\s*using\s+([a-zA-Z_][\w.]*)\s*;"],
                    interface_patterns=[r"^\s*(?:(?:public|private|protected|internal)\s+)*interface\s+(\w+)"],
                    struct_patterns=[r"^\s*(?:(?:public|private|protected|internal)\s+)*struct\s+(\w+)"],
                ),
            )
        if not TREE_SITTER_PHP_AVAILABLE and Language.PHP not in self.processors:
            self.processors[Language.PHP] = RegexProcessor(
                Language.PHP,
                default_config,
                RegexLanguageConfig(
                    function_patterns=[r"^\s*(?:(?:public|private|protected|static)\s+)*function\s+(\w+)"],
                    class_patterns=[r"^\s*(?:(?:abstract|final)\s+)?class\s+(\w+)"],
                    import_patterns=[
                        r"^\s*use\s+([a-zA-Z_\\][\w\\]*)",
                        r'^\s*(?:require|include)(?:_once)?\s*[\(]?\s*[\'"]([^\'"]+)[\'"]',
                    ],
                    interface_patterns=[r"^\s*interface\s+(\w+)"],
                ),
            )
        if not TREE_SITTER_RUBY_AVAILABLE and Language.RUBY not in self.processors:
            self.processors[Language.RUBY] = RegexProcessor(
                Language.RUBY,
                default_config,
                RegexLanguageConfig(
                    function_patterns=[r"^\s*def\s+(\w+[?!=]?)"],
                    class_patterns=[r"^\s*class\s+(\w+)"],
                    import_patterns=[
                        r'^\s*require\s+[\'"]([^\'"]+)[\'"]',
                        r'^\s*require_relative\s+[\'"]([^\'"]+)[\'"]',
                    ],
                    module_patterns=[r"^\s*module\s+(\w+)"],
                    block_end_pattern="end",
                ),
            )

    def get_processor(self, language: Language) -> LanguageProcessor | None:
        """Get processor for a specific language."""
        return self.processors.get(language)

    def get_supported_languages(self) -> set[Language]:
        """Get all supported languages."""
        return set(self.processors.keys())

    def register_processor(self, language: Language, processor: LanguageProcessor) -> None:
        """Register a custom language processor."""
        self.processors[language] = processor
        logger.info(f"Registered custom processor for {language}")


# Global registry instance
language_registry = LanguageRegistry()
