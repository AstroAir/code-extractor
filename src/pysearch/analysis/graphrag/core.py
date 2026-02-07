"""
GraphRAG (Graph Retrieval-Augmented Generation) module for pysearch.

This module implements comprehensive GraphRAG functionality, building knowledge
graphs from code structures and enabling graph-based retrieval for enhanced
semantic search and code understanding.

Classes:
    EntityExtractor: Extracts code entities from source files
    RelationshipMapper: Maps relationships between code entities
    KnowledgeGraphBuilder: Builds and maintains knowledge graphs
    GraphRAGEngine: Main engine for GraphRAG operations

Features:
    - Multi-language entity extraction (functions, classes, variables, etc.)
    - Relationship mapping (calls, inheritance, dependencies, etc.)
    - Knowledge graph construction and maintenance
    - Graph-based retrieval and search
    - Integration with vector databases for semantic similarity
    - Incremental graph updates for code changes
    - Graph traversal and path finding algorithms

Example:
    Basic GraphRAG usage:
        >>> from pysearch.graphrag import GraphRAGEngine
        >>> from pysearch.config import SearchConfig
        >>>
        >>> config = SearchConfig(paths=["./src"])
        >>> engine = GraphRAGEngine(config)
        >>> await engine.build_knowledge_graph()
        >>>
        >>> # Query the knowledge graph
        >>> from pysearch.types import GraphRAGQuery
        >>> query = GraphRAGQuery(
        ...     pattern="database connection handling",
        ...     max_hops=2,
        ...     include_relationships=True
        ... )
        >>> results = await engine.query_graph(query)

    Advanced entity extraction:
        >>> extractor = EntityExtractor()
        >>> entities = await extractor.extract_from_file(Path("example.py"))
        >>> for entity in entities:
        ...     print(f"{entity.entity_type}: {entity.name} at {entity.start_line}")
"""

from __future__ import annotations

import ast
import logging
import re
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

from ...core.config import SearchConfig
from ...core.types import (
    CodeEntity,
    EntityRelationship,
    EntityType,
    Language,
    RelationType,
)
from ...utils.helpers import read_text_safely
from ..dependency_analysis import DependencyAnalyzer
from ..language_detection import detect_language

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extracts code entities from source files across multiple programming languages.

    This class analyzes source code to identify and extract various types of
    code entities such as functions, classes, variables, imports, etc.
    """

    def __init__(self) -> None:
        self.language_extractors = {
            Language.PYTHON: self._extract_python_entities,
            Language.JAVASCRIPT: self._extract_javascript_entities,
            Language.TYPESCRIPT: self._extract_javascript_entities,
            Language.JAVA: self._extract_java_entities,
            Language.CSHARP: self._extract_csharp_entities,
        }

    async def extract_from_file(self, file_path: Path) -> list[CodeEntity]:
        """Extract entities from a single file."""
        language = detect_language(file_path)

        if language not in self.language_extractors:
            logger.debug(f"No entity extractor for language {language} in {file_path}")
            return []

        try:
            content = read_text_safely(file_path)
            if not content:
                return []

            extractor = self.language_extractors[language]
            entities = await extractor(content, file_path, language)

            logger.debug(f"Extracted {len(entities)} entities from {file_path}")
            return entities

        except Exception as e:
            logger.error(f"Failed to extract entities from {file_path}: {e}")
            return []

    async def extract_from_directory(
        self, directory: Path, config: SearchConfig
    ) -> list[CodeEntity]:
        """Extract entities from all files in a directory."""
        from ...indexing.indexer import Indexer

        indexer = Indexer(config)
        all_entities = []

        # Use the scan method to get files, then iterate through them
        changed_files, removed_files, total_files = indexer.scan()
        for file_path in changed_files:
            entities = await self.extract_from_file(file_path)
            all_entities.extend(entities)

        logger.info(f"Extracted {len(all_entities)} entities from {directory}")
        return all_entities

    async def _extract_python_entities(
        self, content: str, file_path: Path, language: Language
    ) -> list[CodeEntity]:
        """Extract entities from Python source code using AST."""
        entities = []

        try:
            tree = ast.parse(content)

            class EntityVisitor(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.entities: list[CodeEntity] = []
                    self.scope_stack: list[str] = ["global"]

                def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                    entity = CodeEntity(
                        id=f"func_{node.name}_{node.lineno}_{uuid4().hex[:8]}",
                        name=node.name,
                        entity_type=EntityType.FUNCTION,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=self._get_function_signature(node),
                        docstring=ast.get_docstring(node),
                        language=language,
                        scope=".".join(self.scope_stack),
                        properties={
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [
                                self._get_decorator_name(d) for d in node.decorator_list
                            ],
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "returns": self._get_return_annotation(node),
                        },
                    )
                    self.entities.append(entity)

                    # Visit function body with updated scope
                    self.scope_stack.append(node.name)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                    # Handle async functions similar to regular functions
                    entity = CodeEntity(
                        id=f"func_{node.name}_{node.lineno}_{uuid4().hex[:8]}",
                        name=node.name,
                        entity_type=EntityType.FUNCTION,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        signature=self._get_function_signature(node),
                        docstring=ast.get_docstring(node),
                        language=language,
                        scope=".".join(self.scope_stack),
                        properties={
                            "args": [arg.arg for arg in node.args.args],
                            "decorators": [
                                self._get_decorator_name(d) for d in node.decorator_list
                            ],
                            "is_async": True,
                            "returns": self._get_return_annotation(node),
                        },
                    )
                    self.entities.append(entity)

                    # Visit function body with updated scope
                    self.scope_stack.append(node.name)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_ClassDef(self, node: ast.ClassDef) -> None:
                    entity = CodeEntity(
                        id=f"class_{node.name}_{node.lineno}_{uuid4().hex[:8]}",
                        name=node.name,
                        entity_type=EntityType.CLASS,
                        file_path=file_path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        docstring=ast.get_docstring(node),
                        language=language,
                        scope=".".join(self.scope_stack),
                        properties={
                            "bases": [self._get_base_name(base) for base in node.bases],
                            "decorators": [
                                self._get_decorator_name(d) for d in node.decorator_list
                            ],
                            "methods": [],
                            "attributes": [],
                        },
                    )
                    self.entities.append(entity)

                    # Visit class body with updated scope
                    self.scope_stack.append(node.name)
                    self.generic_visit(node)
                    self.scope_stack.pop()

                def visit_Import(self, node: ast.Import) -> None:
                    for alias in node.names:
                        entity = CodeEntity(
                            id=f"import_{alias.name}_{node.lineno}_{uuid4().hex[:8]}",
                            name=alias.asname or alias.name,
                            entity_type=EntityType.IMPORT,
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=node.lineno,
                            language=language,
                            scope=".".join(self.scope_stack),
                            properties={
                                "module": alias.name,
                                "alias": alias.asname,
                                "import_type": "import",
                            },
                        )
                        self.entities.append(entity)

                def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
                    module = node.module or ""
                    for alias in node.names:
                        entity = CodeEntity(
                            id=f"import_{alias.name}_{node.lineno}_{uuid4().hex[:8]}",
                            name=alias.asname or alias.name,
                            entity_type=EntityType.IMPORT,
                            file_path=file_path,
                            start_line=node.lineno,
                            end_line=node.lineno,
                            language=language,
                            scope=".".join(self.scope_stack),
                            properties={
                                "module": module,
                                "imported_name": alias.name,
                                "alias": alias.asname,
                                "import_type": "from_import",
                                "level": node.level,
                            },
                        )
                        self.entities.append(entity)

                def visit_Assign(self, node: ast.Assign) -> None:
                    # Extract variable assignments
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            entity = CodeEntity(
                                id=f"var_{target.id}_{node.lineno}_{uuid4().hex[:8]}",
                                name=target.id,
                                entity_type=EntityType.VARIABLE,
                                file_path=file_path,
                                start_line=node.lineno,
                                end_line=node.lineno,
                                language=language,
                                scope=".".join(self.scope_stack),
                                properties={
                                    "assignment_type": "simple",
                                    "value_type": type(node.value).__name__,
                                },
                            )
                            self.entities.append(entity)

                def _get_function_signature(
                    self, node: ast.FunctionDef | ast.AsyncFunctionDef
                ) -> str:
                    """Extract function signature as string."""
                    args = []
                    for arg in node.args.args:
                        arg_str = arg.arg
                        if arg.annotation:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        args.append(arg_str)

                    sig = f"def {node.name}({', '.join(args)})"
                    if node.returns:
                        sig += f" -> {ast.unparse(node.returns)}"
                    sig += ":"
                    return sig

                def _get_decorator_name(self, decorator: ast.expr) -> str:
                    """Extract decorator name."""
                    if isinstance(decorator, ast.Name):
                        return decorator.id
                    elif isinstance(decorator, ast.Attribute):
                        return ast.unparse(decorator)
                    else:
                        return ast.unparse(decorator)

                def _get_base_name(self, base: ast.expr) -> str:
                    """Extract base class name."""
                    if isinstance(base, ast.Name):
                        return base.id
                    else:
                        return ast.unparse(base)

                def _get_return_annotation(
                    self, node: ast.FunctionDef | ast.AsyncFunctionDef
                ) -> str | None:
                    """Extract return type annotation."""
                    if node.returns:
                        return ast.unparse(node.returns)
                    return None

            visitor = EntityVisitor()
            visitor.visit(tree)
            entities = visitor.entities

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error parsing Python file {file_path}: {e}")

        return entities

    async def _extract_javascript_entities(
        self, content: str, file_path: Path, language: Language
    ) -> list[CodeEntity]:
        """Extract entities from JavaScript/TypeScript using regex patterns."""
        entities = []

        # Function declarations
        func_pattern = re.compile(r"^\s*(async\s+)?function\s+(\w+)\s*\([^)]*\)", re.MULTILINE)
        for match in func_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            entity = CodeEntity(
                id=f"func_{match.group(2)}_{line_num}_{uuid4().hex[:8]}",
                name=match.group(2),
                entity_type=EntityType.FUNCTION,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                signature=match.group(0).strip(),
                language=language,
                properties={"is_async": bool(match.group(1))},
            )
            entities.append(entity)

        # Class declarations
        class_pattern = re.compile(r"^\s*class\s+(\w+)(?:\s+extends\s+(\w+))?\s*{", re.MULTILINE)
        for match in class_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            entity = CodeEntity(
                id=f"class_{match.group(1)}_{line_num}_{uuid4().hex[:8]}",
                name=match.group(1),
                entity_type=EntityType.CLASS,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                language=language,
                properties={"extends": match.group(2) if match.group(2) else None},
            )
            entities.append(entity)

        # Import statements
        import_pattern = re.compile(
            r'^\s*import\s+(?:{([^}]+)}|\*\s+as\s+(\w+)|(\w+))\s+from\s+[\'"]([^\'"]+)[\'"]',
            re.MULTILINE,
        )
        for match in import_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            if match.group(1):  # Named imports
                imports = [name.strip() for name in match.group(1).split(",")]
                for imp in imports:
                    entity = CodeEntity(
                        id=f"import_{imp}_{line_num}_{uuid4().hex[:8]}",
                        name=imp,
                        entity_type=EntityType.IMPORT,
                        file_path=file_path,
                        start_line=line_num,
                        end_line=line_num,
                        language=language,
                        properties={"module": match.group(4), "import_type": "named"},
                    )
                    entities.append(entity)
            elif match.group(2):  # Namespace import
                entity = CodeEntity(
                    id=f"import_{match.group(2)}_{line_num}_{uuid4().hex[:8]}",
                    name=match.group(2),
                    entity_type=EntityType.IMPORT,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=line_num,
                    language=language,
                    properties={"module": match.group(4), "import_type": "namespace"},
                )
                entities.append(entity)
            elif match.group(3):  # Default import
                entity = CodeEntity(
                    id=f"import_{match.group(3)}_{line_num}_{uuid4().hex[:8]}",
                    name=match.group(3),
                    entity_type=EntityType.IMPORT,
                    file_path=file_path,
                    start_line=line_num,
                    end_line=line_num,
                    language=language,
                    properties={"module": match.group(4), "import_type": "default"},
                )
                entities.append(entity)

        return entities

    async def _extract_java_entities(
        self, content: str, file_path: Path, language: Language
    ) -> list[CodeEntity]:
        """Extract entities from Java using regex patterns."""
        entities = []

        # Class declarations
        class_pattern = re.compile(
            r"^\s*(?:public|private|protected)?\s*(?:abstract|final)?\s*class\s+(\w+)"
            r"(?:\s+extends\s+(\w+))?(?:\s+implements\s+([^{]+))?\s*{",
            re.MULTILINE,
        )
        for match in class_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            entity = CodeEntity(
                id=f"class_{match.group(1)}_{line_num}_{uuid4().hex[:8]}",
                name=match.group(1),
                entity_type=EntityType.CLASS,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                language=language,
                properties={
                    "extends": match.group(2) if match.group(2) else None,
                    "implements": match.group(3).strip() if match.group(3) else None,
                },
            )
            entities.append(entity)

        # Method declarations
        method_pattern = re.compile(
            r"^\s*(?:public|private|protected)?\s*(?:static)?\s*(?:final)?\s*(\w+)\s+(\w+)"
            r"\s*\([^)]*\)\s*(?:throws\s+[^{]+)?\s*{",
            re.MULTILINE,
        )
        for match in method_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            entity = CodeEntity(
                id=f"method_{match.group(2)}_{line_num}_{uuid4().hex[:8]}",
                name=match.group(2),
                entity_type=EntityType.METHOD,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                signature=match.group(0).strip(),
                language=language,
                properties={"return_type": match.group(1)},
            )
            entities.append(entity)

        return entities

    async def _extract_csharp_entities(
        self, content: str, file_path: Path, language: Language
    ) -> list[CodeEntity]:
        """Extract entities from C# using regex patterns."""
        entities = []

        # Class declarations
        class_pattern = re.compile(
            r"^\s*(?:public|private|protected|internal)?\s*(?:abstract|sealed|static)?\s*class\s+(\w+)"
            r"(?:\s*:\s*([^{]+))?\s*{",
            re.MULTILINE,
        )
        for match in class_pattern.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            entity = CodeEntity(
                id=f"class_{match.group(1)}_{line_num}_{uuid4().hex[:8]}",
                name=match.group(1),
                entity_type=EntityType.CLASS,
                file_path=file_path,
                start_line=line_num,
                end_line=line_num,
                language=language,
                properties={"inheritance": match.group(2).strip() if match.group(2) else None},
            )
            entities.append(entity)

        return entities


class RelationshipMapper:
    """
    Maps relationships between code entities to build comprehensive knowledge graphs.

    This class analyzes code entities and their interactions to identify various
    types of relationships such as function calls, inheritance, dependencies, etc.
    """

    def __init__(self) -> None:
        self.dependency_analyzer = DependencyAnalyzer()

    async def map_relationships(
        self, entities: list[CodeEntity], file_contents: dict[Path, str]
    ) -> list[EntityRelationship]:
        """Map relationships between entities."""
        relationships = []

        # Group entities by file for efficient processing
        entities_by_file = defaultdict(list)
        for entity in entities:
            entities_by_file[entity.file_path].append(entity)

        # Map different types of relationships
        relationships.extend(await self._map_inheritance_relationships(entities))
        relationships.extend(await self._map_call_relationships(entities, file_contents))
        relationships.extend(await self._map_import_relationships(entities))
        relationships.extend(await self._map_containment_relationships(entities))

        logger.info(f"Mapped {len(relationships)} relationships between entities")
        return relationships

    async def _map_inheritance_relationships(
        self, entities: list[CodeEntity]
    ) -> list[EntityRelationship]:
        """Map inheritance relationships between classes."""
        relationships = []

        # Create a mapping of class names to entities
        class_entities = {
            entity.name: entity for entity in entities if entity.entity_type == EntityType.CLASS
        }

        for entity in entities:
            if entity.entity_type == EntityType.CLASS:
                # Check for inheritance in properties
                if "bases" in entity.properties:
                    for base_name in entity.properties["bases"]:
                        if base_name in class_entities:
                            relationship = EntityRelationship(
                                id=f"inherit_{entity.id}_{class_entities[base_name].id}",
                                source_entity_id=entity.id,
                                target_entity_id=class_entities[base_name].id,
                                relation_type=RelationType.INHERITS,
                                confidence=0.9,
                                context=f"Class {entity.name} inherits from {base_name}",
                                file_path=entity.file_path,
                                line_number=entity.start_line,
                            )
                            relationships.append(relationship)

                # Check for JavaScript/Java extends
                if "extends" in entity.properties and entity.properties["extends"]:
                    base_name = entity.properties["extends"]
                    if base_name in class_entities:
                        relationship = EntityRelationship(
                            id=f"extend_{entity.id}_{class_entities[base_name].id}",
                            source_entity_id=entity.id,
                            target_entity_id=class_entities[base_name].id,
                            relation_type=RelationType.EXTENDS,
                            confidence=0.9,
                            context=f"Class {entity.name} extends {base_name}",
                            file_path=entity.file_path,
                            line_number=entity.start_line,
                        )
                        relationships.append(relationship)

        return relationships

    async def _map_call_relationships(
        self, entities: list[CodeEntity], file_contents: dict[Path, str]
    ) -> list[EntityRelationship]:
        """Map function/method call relationships."""
        relationships = []

        # Create mappings for quick lookup
        function_entities = {
            entity.name: entity
            for entity in entities
            if entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]
        }

        for entity in entities:
            if entity.entity_type in [EntityType.FUNCTION, EntityType.METHOD]:
                file_content = file_contents.get(entity.file_path, "")
                if not file_content:
                    continue

                # Extract function body (simplified approach)
                lines = file_content.split("\n")
                start_idx = max(0, entity.start_line - 1)
                end_idx = min(len(lines), entity.end_line)
                function_body = "\n".join(lines[start_idx:end_idx])

                # Look for function calls in the body
                for func_name, target_entity in function_entities.items():
                    if func_name != entity.name and func_name in function_body:
                        # Simple pattern matching for function calls
                        call_pattern = re.compile(rf"\b{re.escape(func_name)}\s*\(")
                        if call_pattern.search(function_body):
                            relationship = EntityRelationship(
                                id=f"call_{entity.id}_{target_entity.id}",
                                source_entity_id=entity.id,
                                target_entity_id=target_entity.id,
                                relation_type=RelationType.CALLS,
                                confidence=0.7,
                                context=f"Function {entity.name} calls {func_name}",
                                file_path=entity.file_path,
                            )
                            relationships.append(relationship)

        return relationships

    async def _map_import_relationships(
        self, entities: list[CodeEntity]
    ) -> list[EntityRelationship]:
        """Map import relationships."""
        relationships = []

        # Group entities by file
        entities_by_file = defaultdict(list)
        for entity in entities:
            entities_by_file[entity.file_path].append(entity)

        for file_path, file_entities in entities_by_file.items():
            import_entities = [e for e in file_entities if e.entity_type == EntityType.IMPORT]
            other_entities = [e for e in file_entities if e.entity_type != EntityType.IMPORT]

            # Map imports to usage within the same file
            for import_entity in import_entities:
                for other_entity in other_entities:
                    if other_entity.start_line > import_entity.start_line:
                        relationship = EntityRelationship(
                            id=f"import_{import_entity.id}_{other_entity.id}",
                            source_entity_id=other_entity.id,
                            target_entity_id=import_entity.id,
                            relation_type=RelationType.IMPORTS,
                            confidence=0.8,
                            context=f"Entity {other_entity.name} uses import {import_entity.name}",
                            file_path=file_path,
                            line_number=import_entity.start_line,
                        )
                        relationships.append(relationship)

        return relationships

    async def _map_containment_relationships(
        self, entities: list[CodeEntity]
    ) -> list[EntityRelationship]:
        """Map containment relationships (classes contain methods, etc.)."""
        relationships = []

        # Group entities by file and sort by line number
        entities_by_file = defaultdict(list)
        for entity in entities:
            entities_by_file[entity.file_path].append(entity)

        for file_path, file_entities in entities_by_file.items():
            file_entities.sort(key=lambda e: e.start_line)

            # Find containment relationships based on line ranges and scope
            for i, entity in enumerate(file_entities):
                if entity.entity_type == EntityType.CLASS:
                    # Find methods and attributes within this class
                    for j in range(i + 1, len(file_entities)):
                        other_entity = file_entities[j]

                        # Stop if we've moved past this class
                        if (
                            other_entity.entity_type == EntityType.CLASS
                            and other_entity.start_line > entity.end_line
                        ):
                            break

                        # Check if the other entity is contained within this class
                        if (
                            other_entity.start_line > entity.start_line
                            and other_entity.end_line <= entity.end_line
                            and other_entity.entity_type in [EntityType.METHOD, EntityType.FUNCTION]
                        ):

                            relationship = EntityRelationship(
                                id=f"contain_{entity.id}_{other_entity.id}",
                                source_entity_id=entity.id,
                                target_entity_id=other_entity.id,
                                relation_type=RelationType.CONTAINS,
                                confidence=0.95,
                                context=(
                                    f"Class {entity.name} contains {other_entity.entity_type.value} "
                                    f"{other_entity.name}"
                                ),
                                file_path=file_path,
                                line_number=other_entity.start_line,
                            )
                            relationships.append(relationship)

        return relationships
