"""
GraphRAG type definitions for pysearch knowledge graph functionality.

This module contains all the data types, enumerations, and data classes
specifically used for GraphRAG (Graph Retrieval-Augmented Generation) operations,
including knowledge graph construction, entity relationships, and graph-based queries.

Key Types:
    EntityType: Enumeration of code entity types for knowledge graphs
    RelationType: Enumeration of relationship types between entities
    CodeEntity: Represents a code entity (function, class, variable, etc.)
    EntityRelationship: Represents relationships between code entities
    KnowledgeGraph: Complete knowledge graph structure
    GraphRAGQuery: Query specification for GraphRAG operations
    GraphRAGResult: Results from GraphRAG queries with graph context

Example:
    Creating a GraphRAG query:
        >>> from pysearch.core.types.graphrag_types import GraphRAGQuery, EntityType, RelationType
        >>>
        >>> query = GraphRAGQuery(
        ...     pattern="database connection handling",
        ...     entity_types=[EntityType.FUNCTION, EntityType.CLASS],
        ...     relation_types=[RelationType.CALLS, RelationType.USES],
        ...     max_hops=2,
        ...     include_relationships=True
        ... )

    Working with knowledge graphs:
        >>> from pysearch.core.types.graphrag_types import KnowledgeGraph, CodeEntity
        >>>
        >>> graph = KnowledgeGraph()
        >>> graph.add_entity(entity)
        >>> graph.add_relationship(relationship)
        >>> related_entities = graph.get_related_entities("func_main_123")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from .basic_types import Language


class EntityType(str, Enum):
    """Types of code entities for GraphRAG knowledge graphs."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    CONSTANT = "constant"
    MODULE = "module"
    PACKAGE = "package"
    IMPORT = "import"
    DECORATOR = "decorator"
    PROPERTY = "property"
    ATTRIBUTE = "attribute"
    PARAMETER = "parameter"
    RETURN_TYPE = "return_type"
    EXCEPTION = "exception"
    INTERFACE = "interface"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"
    ANNOTATION = "annotation"
    COMMENT = "comment"
    DOCSTRING = "docstring"
    TEST = "test"
    CONFIGURATION = "configuration"
    UNKNOWN_ENTITY = "unknown_entity"


class RelationType(str, Enum):
    """Types of relationships between code entities."""

    # Structural relationships
    CONTAINS = "contains"  # Class contains method, module contains function
    INHERITS = "inherits"  # Class inheritance
    IMPLEMENTS = "implements"  # Interface implementation
    EXTENDS = "extends"  # Extension relationships
    OVERRIDES = "overrides"  # Method overriding

    # Usage relationships
    CALLS = "calls"  # Function calls another function
    USES = "uses"  # General usage relationship
    IMPORTS = "imports"  # Import relationships
    REFERENCES = "references"  # Variable/type references
    INSTANTIATES = "instantiates"  # Object instantiation

    # Data flow relationships
    RETURNS = "returns"  # Function returns type
    ACCEPTS = "accepts"  # Function accepts parameter
    ASSIGNS = "assigns"  # Variable assignment
    MODIFIES = "modifies"  # State modification

    # Dependency relationships
    DEPENDS_ON = "depends_on"  # General dependency
    REQUIRES = "requires"  # Required dependency
    PROVIDES = "provides"  # Service provision

    # Semantic relationships
    SIMILAR_TO = "similar_to"  # Semantic similarity
    RELATED_TO = "related_to"  # General semantic relation
    EQUIVALENT_TO = "equivalent_to"  # Functional equivalence

    # Test relationships
    TESTS = "tests"  # Test relationship
    MOCKS = "mocks"  # Mocking relationship

    # Documentation relationships
    DOCUMENTS = "documents"  # Documentation relationship
    DESCRIBES = "describes"  # Description relationship

    # Configuration relationships
    CONFIGURES = "configures"  # Configuration relationship

    UNKNOWN_RELATION = "unknown_relation"


@dataclass(slots=True)
class CodeEntity:
    """
    Represents a code entity in the GraphRAG knowledge graph.

    This class captures detailed information about code elements like functions,
    classes, variables, etc., including their location, properties, and metadata
    for building comprehensive knowledge graphs.

    Attributes:
        id: Unique identifier for the entity
        name: Entity name (function name, class name, etc.)
        entity_type: Type of the entity (function, class, variable, etc.)
        file_path: Path to the file containing the entity
        start_line: Starting line number in the file
        end_line: Ending line number in the file
        signature: Function/method signature or variable declaration
        docstring: Associated documentation string
        properties: Additional properties specific to the entity type
        embedding: Vector embedding for semantic similarity
        confidence: Confidence score for entity extraction (0.0-1.0)
        language: Programming language of the entity
        scope: Scope information (global, class, function, etc.)
        access_modifier: Access level (public, private, protected, etc.)

    Example:
        >>> entity = CodeEntity(
        ...     id="func_main_123",
        ...     name="main",
        ...     entity_type=EntityType.FUNCTION,
        ...     file_path=Path("app.py"),
        ...     start_line=10,
        ...     end_line=15,
        ...     signature="def main() -> None:",
        ...     docstring="Main entry point of the application"
        ... )
    """

    id: str
    name: str
    entity_type: EntityType
    file_path: Path
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    confidence: float = 1.0
    language: Language = Language.UNKNOWN
    scope: str | None = None
    access_modifier: str | None = None


@dataclass(slots=True)
class EntityRelationship:
    """
    Represents a relationship between two code entities.

    This class captures the connections between code elements, enabling
    graph-based analysis and retrieval operations.

    Attributes:
        id: Unique identifier for the relationship
        source_entity_id: ID of the source entity
        target_entity_id: ID of the target entity
        relation_type: Type of relationship
        properties: Additional relationship properties
        confidence: Confidence score for relationship extraction (0.0-1.0)
        weight: Relationship strength/importance weight
        context: Contextual information about the relationship
        file_path: File where the relationship is observed
        line_number: Line number where the relationship occurs

    Example:
        >>> relationship = EntityRelationship(
        ...     id="rel_123",
        ...     source_entity_id="func_main_123",
        ...     target_entity_id="func_helper_456",
        ...     relation_type=RelationType.CALLS,
        ...     confidence=0.95,
        ...     weight=1.0,
        ...     context="Function call in main execution flow"
        ... )
    """

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    weight: float = 1.0
    context: str | None = None
    file_path: Path | None = None
    line_number: int | None = None


@dataclass(slots=True)
class KnowledgeGraph:
    """
    Complete knowledge graph structure for GraphRAG operations.

    This class represents the entire knowledge graph built from code analysis,
    containing entities, relationships, and metadata for efficient querying
    and retrieval operations.

    Attributes:
        entities: Dictionary mapping entity IDs to CodeEntity objects
        relationships: List of EntityRelationship objects
        entity_index: Index mapping entity names to IDs for fast lookup
        type_index: Index mapping entity types to lists of entity IDs
        file_index: Index mapping file paths to lists of entity IDs
        metadata: Graph-level metadata and statistics
        version: Version identifier for the knowledge graph
        created_at: Timestamp when the graph was created
        updated_at: Timestamp when the graph was last updated

    Example:
        >>> graph = KnowledgeGraph()
        >>> graph.add_entity(entity)
        >>> graph.add_relationship(relationship)
        >>> related_entities = graph.get_related_entities("func_main_123")
    """

    entities: dict[str, CodeEntity] = field(default_factory=dict)
    relationships: list[EntityRelationship] = field(default_factory=list)
    entity_index: dict[str, list[str]] = field(default_factory=dict)
    type_index: dict[EntityType, list[str]] = field(default_factory=dict)
    file_index: dict[Path, list[str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: float | None = None
    updated_at: float | None = None

    def add_entity(self, entity: CodeEntity) -> None:
        """Add an entity to the knowledge graph."""
        self.entities[entity.id] = entity

        # Update indexes
        if entity.name not in self.entity_index:
            self.entity_index[entity.name] = []
        self.entity_index[entity.name].append(entity.id)

        if entity.entity_type not in self.type_index:
            self.type_index[entity.entity_type] = []
        self.type_index[entity.entity_type].append(entity.id)

        if entity.file_path not in self.file_index:
            self.file_index[entity.file_path] = []
        self.file_index[entity.file_path].append(entity.id)

    def add_relationship(self, relationship: EntityRelationship) -> None:
        """Add a relationship to the knowledge graph."""
        self.relationships.append(relationship)

    def get_entity(self, entity_id: str) -> CodeEntity | None:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_name(self, name: str) -> list[CodeEntity]:
        """Get all entities with the given name."""
        entity_ids = self.entity_index.get(name, [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_entities_by_type(self, entity_type: EntityType) -> list[CodeEntity]:
        """Get all entities of the given type."""
        entity_ids = self.type_index.get(entity_type, [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_entities_in_file(self, file_path: Path) -> list[CodeEntity]:
        """Get all entities in the given file."""
        entity_ids = self.file_index.get(file_path, [])
        return [self.entities[eid] for eid in entity_ids if eid in self.entities]

    def get_related_entities(
        self,
        entity_id: str,
        relation_types: list[RelationType] | None = None,
        max_hops: int = 1
    ) -> list[tuple[CodeEntity, EntityRelationship]]:
        """Get entities related to the given entity."""
        related = []

        for relationship in self.relationships:
            if relationship.source_entity_id == entity_id:
                if relation_types is None or relationship.relation_type in relation_types:
                    target_entity = self.entities.get(
                        relationship.target_entity_id)
                    if target_entity:
                        related.append((target_entity, relationship))
            elif relationship.target_entity_id == entity_id:
                if relation_types is None or relationship.relation_type in relation_types:
                    source_entity = self.entities.get(
                        relationship.source_entity_id)
                    if source_entity:
                        related.append((source_entity, relationship))

        return related


@dataclass(slots=True)
class GraphRAGQuery:
    """
    Query specification for GraphRAG operations.

    This class extends the basic Query with graph-specific parameters
    for performing retrieval-augmented generation using knowledge graphs.

    Attributes:
        pattern: The search pattern or question
        entity_types: Specific entity types to focus on
        relation_types: Specific relationship types to consider
        include_relationships: Whether to include relationship information
        max_hops: Maximum number of hops in the knowledge graph
        min_confidence: Minimum confidence threshold for entities/relationships
        semantic_threshold: Minimum semantic similarity threshold
        use_vector_search: Whether to use vector similarity search
        context_window: Size of context window for related entities
        ranking_strategy: Strategy for ranking results

    Example:
        >>> query = GraphRAGQuery(
        ...     pattern="database connection handling",
        ...     entity_types=[EntityType.FUNCTION, EntityType.CLASS],
        ...     relation_types=[RelationType.CALLS, RelationType.USES],
        ...     max_hops=2,
        ...     include_relationships=True
        ... )
    """

    pattern: str
    entity_types: list[EntityType] | None = None
    relation_types: list[RelationType] | None = None
    include_relationships: bool = True
    max_hops: int = 2
    min_confidence: float = 0.5
    semantic_threshold: float = 0.7
    use_vector_search: bool = True
    context_window: int = 5
    ranking_strategy: str = "relevance"


@dataclass(slots=True)
class GraphRAGResult:
    """
    Results from GraphRAG query operations.

    This class contains the results of a GraphRAG query, including
    matched entities, relationships, and contextual information.

    Attributes:
        entities: List of matched entities
        relationships: List of relevant relationships
        context_entities: Additional entities providing context
        similarity_scores: Similarity scores for each entity
        graph_paths: Paths through the knowledge graph
        metadata: Additional result metadata
        query: Original query that produced these results

    Example:
        >>> result = GraphRAGResult(
        ...     entities=[entity1, entity2],
        ...     relationships=[rel1, rel2],
        ...     similarity_scores={"entity1": 0.95, "entity2": 0.87}
        ... )
    """

    entities: list[CodeEntity] = field(default_factory=list)
    relationships: list[EntityRelationship] = field(default_factory=list)
    context_entities: list[CodeEntity] = field(default_factory=list)
    similarity_scores: dict[str, float] = field(default_factory=dict)
    graph_paths: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    query: GraphRAGQuery | None = None
