"""
Core type definitions for pysearch.

This module contains all the fundamental data types, enumerations, and data classes
used throughout the pysearch system. It defines the structure for search queries,
results, configuration options, metadata, and GraphRAG functionality.

Key Types:
    OutputFormat: Enumeration of supported output formats
    Language: Enumeration of supported programming languages
    ASTFilters: Configuration for AST-based filtering
    FileMetadata: Extended file metadata for advanced filtering
    SearchItem: Individual search result item with context
    SearchResult: Complete search results with statistics
    Query: Search query specification
    MetadataFilters: Advanced metadata-based filters

GraphRAG Types:
    EntityType: Enumeration of code entity types for knowledge graphs
    RelationType: Enumeration of relationship types between entities
    CodeEntity: Represents a code entity (function, class, variable, etc.)
    EntityRelationship: Represents relationships between code entities
    KnowledgeGraph: Complete knowledge graph structure
    GraphRAGQuery: Query specification for GraphRAG operations
    GraphRAGResult: Results from GraphRAG queries with graph context

Example:
    Creating a search query:
        >>> from pysearch.types import Query, ASTFilters, OutputFormat
        >>>
        >>> # Basic text search
        >>> query = Query(pattern="def main", use_regex=True)
        >>>
        >>> # AST-based search with filters
        >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
        >>> query = Query(pattern="def", use_ast=True, ast_filters=filters)
        >>>
        >>> # Semantic search
        >>> query = Query(pattern="database connection", use_semantic=True)
        >>>
        >>> # GraphRAG search
        >>> from pysearch.types import GraphRAGQuery
        >>> graph_query = GraphRAGQuery(
        ...     pattern="database operations",
        ...     include_relationships=True,
        ...     max_hops=2
        ... )

    Working with results:
        >>> from pysearch.types import SearchResult, SearchItem
        >>>
        >>> # Process search results
        >>> for item in results.items:
        ...     print(f"Found in {item.file} at lines {item.start_line}-{item.end_line}")
        ...     for line in item.lines:
        ...         print(f"  {line}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class OutputFormat(str, Enum):
    TEXT = "text"
    JSON = "json"
    HIGHLIGHT = "highlight"


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
class ASTFilters:
    r"""
    Configuration for AST-based filtering during search operations.

    This class defines regex patterns for filtering search results based on
    Abstract Syntax Tree (AST) elements like function names, class names,
    decorators, and import statements.

    Attributes:
        func_name: Regex pattern to match function names
        class_name: Regex pattern to match class names
        decorator: Regex pattern to match decorator names
        imported: Regex pattern to match imported symbols (including module prefix)

    Example:
        >>> from pysearch.types import ASTFilters
        >>>
        >>> # Filter for handler functions with cache decorators
        >>> filters = ASTFilters(
        ...     func_name=r".*_handler$",
        ...     decorator=r"(lru_cache|cache)"
        ... )
        >>>
        >>> # Filter for controller classes
        >>> filters = ASTFilters(class_name=r".*Controller$")
        >>>
        >>> # Filter for specific imports
        >>> filters = ASTFilters(imported=r"requests\.(get|post)")
    """

    func_name: str | None = None
    class_name: str | None = None
    decorator: str | None = None
    imported: str | None = None


@dataclass(slots=True)
class FileMetadata:
    """Extended file metadata for advanced filtering."""

    path: Path
    size: int
    mtime: float
    language: Language
    encoding: str = "utf-8"
    line_count: int | None = None
    author: str | None = None  # From git blame or file attributes
    created_date: float | None = None
    modified_date: float | None = None


# match_spans: list of (line_index, (start_col, end_col))
MatchSpan = tuple[int, tuple[int, int]]


@dataclass(slots=True)
class SearchItem:
    """
    Represents a single search result item with context.

    This class contains information about a matched location in a file,
    including the file path, line range, content lines, and precise
    match positions within those lines.

    Attributes:
        file: Path to the file containing the match
        start_line: Starting line number (1-based) of the match context
        end_line: Ending line number (1-based) of the match context
        lines: List of content lines including context around the match
        match_spans: List of precise match positions as (line_index, (start_col, end_col))

    Example:
        >>> item = SearchItem(
        ...     file=Path("example.py"),
        ...     start_line=10,
        ...     end_line=12,
        ...     lines=["    # Context line", "    def main():", "        pass"],
        ...     match_spans=[(1, (4, 12))]  # "def main" at line 1, cols 4-12
        ... )
    """

    file: Path
    start_line: int
    end_line: int
    lines: list[str]
    match_spans: list[MatchSpan] = field(default_factory=list)


@dataclass(slots=True)
class SearchStats:
    """
    Performance and result statistics for a search operation.

    Attributes:
        files_scanned: Total number of files examined during search
        files_matched: Number of files containing at least one match
        items: Total number of search result items found
        elapsed_ms: Total search time in milliseconds
        indexed_files: Number of files processed by the indexer
    """

    files_scanned: int = 0
    files_matched: int = 0
    items: int = 0
    elapsed_ms: float = 0.0
    indexed_files: int = 0


@dataclass(slots=True)
class SearchResult:
    """
    Complete search results including items and performance statistics.

    This is the main result object returned by search operations, containing
    both the matched items and metadata about the search performance.

    Attributes:
        items: List of SearchItem objects representing matches
        stats: SearchStats object with performance metrics

    Example:
        >>> result = SearchResult(
        ...     items=[item1, item2, item3],
        ...     stats=SearchStats(files_scanned=100, files_matched=3, items=3, elapsed_ms=45.2)
        ... )
        >>> print(f"Found {len(result.items)} matches in {result.stats.elapsed_ms}ms")
    """

    items: list[SearchItem] = field(default_factory=list)
    stats: SearchStats = field(default_factory=SearchStats)


@dataclass(slots=True)
class MetadataFilters:
    """Advanced metadata-based filters for search."""

    min_size: int | None = None  # Minimum file size in bytes
    max_size: int | None = None  # Maximum file size in bytes
    modified_after: float | None = None  # Unix timestamp
    modified_before: float | None = None  # Unix timestamp
    created_after: float | None = None  # Unix timestamp
    created_before: float | None = None  # Unix timestamp
    min_lines: int | None = None  # Minimum line count
    max_lines: int | None = None  # Maximum line count
    author_pattern: str | None = None  # Regex pattern for author name
    encoding_pattern: str | None = None  # Regex pattern for file encoding
    languages: set[Language] | None = None  # Specific languages to include


@dataclass(slots=True)
class Query:
    r"""
    Search query specification with all search parameters.

    This class encapsulates all the parameters needed to execute a search,
    including the pattern, search modes, filters, and output preferences.

    Attributes:
        pattern: The search pattern (text, regex, or semantic query)
        use_regex: Whether to interpret pattern as regular expression
        use_ast: Whether to use AST-based structural matching
        context: Number of context lines to include around matches
        output: Output format for results
        filters: AST-based filters for structural matching
        metadata_filters: File metadata-based filters
        search_docstrings: Whether to search in docstrings
        search_comments: Whether to search in comments
        search_strings: Whether to search in string literals

    Examples:
        Simple text search:
            >>> query = Query(pattern="def main")
            >>> # Searches for literal text "def main"

        Regex search:
            >>> query = Query(pattern=r"def \w+_handler", use_regex=True)
            >>> # Searches for functions ending with "_handler"

        AST search with filters:
            >>> from pysearch.types import ASTFilters
            >>> filters = ASTFilters(func_name="main", decorator="lru_cache")
            >>> query = Query(pattern="def", use_ast=True, filters=filters)
            >>> # Finds function definitions matching the AST filters

        Semantic search:
            >>> query = Query(pattern="database connection", use_semantic=True)
            >>> # Finds code semantically related to database connections

        Complex query with metadata filters:
            >>> from pysearch.types import MetadataFilters, Language
            >>> metadata = MetadataFilters(min_lines=50, languages={Language.PYTHON})
            >>> query = Query(
            ...     pattern="class.*Test",
            ...     use_regex=True,
            ...     context=5,
            ...     metadata_filters=metadata,
            ...     search_docstrings=True,
            ...     search_comments=False
            ... )
            >>> # Searches for test classes in substantial Python files
    """

    pattern: str
    use_regex: bool = False
    use_ast: bool = False
    use_semantic: bool = False
    context: int = 2
    output: OutputFormat = OutputFormat.TEXT
    filters: ASTFilters | None = None
    metadata_filters: MetadataFilters | None = None
    search_docstrings: bool = True
    search_comments: bool = True
    search_strings: bool = True


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
