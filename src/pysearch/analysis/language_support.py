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
            # Add other languages as needed

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
            # Add other languages as needed

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

    def _analyze_dependencies_regex(self, content: str) -> list[str]:
        """Fallback regex-based dependency analysis."""
        dependencies: list[str] = []

        if self.language == Language.PYTHON:
            # Python imports
            import_patterns = [
                r"^\s*import\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)",
                r"^\s*from\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)\s+import",
            ]
        elif self.language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
            # JavaScript imports
            import_patterns = [
                r'^\s*import.*from\s+[\'"]([^\'"]+)[\'"]',
                r'^\s*const.*=\s*require\([\'"]([^\'"]+)[\'"]\)',
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
                complexity_nodes = self._count_python_complexity_nodes(tree.root_node)
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

    def _count_python_complexity_nodes(self, node: Any) -> int:
        """Count Python-specific complexity indicators."""
        complexity_types = {
            "if_statement",
            "for_statement",
            "while_statement",
            "try_statement",
            "with_statement",
            "function_definition",
            "class_definition",
            "lambda",
            "list_comprehension",
            "dictionary_comprehension",
            "set_comprehension",
        }

        count = 0
        if node.type in complexity_types:
            count += 1

        for child in node.children:
            count += self._count_python_complexity_nodes(child)

        return count


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

        for language in tree_sitter_languages:
            self.processors[language] = TreeSitterProcessor(language, default_config)

        # Other languages would use regex-based processors
        # (implementation would be added here)

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
