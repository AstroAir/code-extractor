# Analysis Module

[根目录](../../../CLAUDE.md) > [src](../../) > [pysearch](../) > **analysis**

---

## Change Log (Changelog)

### 2026-01-19 - Module Documentation Initial Version
- Created comprehensive Analysis module documentation

---

## Module Responsibility

The **Analysis** module provides code analysis and understanding capabilities:

1. **Dependency Analysis**: Code dependency tracking and graph generation
2. **Language Detection**: Programming language identification and support
3. **Content Analysis**: Code content addressing and hashing
4. **GraphRAG**: Graph-based retrieval-augmented generation

---

## Key Files

### Core Analysis
| File | Purpose | Description |
|------|---------|-------------|
| `dependency_analysis.py` | Dependency Analysis | Dependency graph generation and analysis |
| `language_detection.py` | Language Detection | Programming language identification |
| `language_support.py` | Language Support | Enhanced language-specific processing |
| `content_addressing.py` | Content Addressing | Content hashing and addressing |

### GraphRAG
| File | Purpose | Description |
|------|---------|-------------|
| `graphrag/__init__.py` | GraphRAG Interface | GraphRAG exports |
| `graphrag/core.py` | GraphRAG Core | Core GraphRAG functionality |
| `graphrag/engine.py` | GraphRAG Engine | GraphRAG query engine |

---

## Dependency Analysis

### Overview
Provides dependency graph construction and analysis for code understanding.

### Key Classes
```python
class DependencyAnalyzer:
    def analyze_directory(self, directory: Path) -> DependencyGraph
    def analyze_file(self, file_path: Path) -> list[Dependency]

class DependencyGraph:
    nodes: dict[str, CodeEntity]
    edges: list[EntityRelationship]
    add_node(entity: CodeEntity) -> None
    add_edge(relationship: EntityRelationship) -> None
    find_cycles() -> list[list[str]]

class CircularDependencyDetector:
    def __init__(self, graph: DependencyGraph)
    def find_cycles() -> list[list[str]]
    def detect_cycles_strongly_connected() -> list[list[str]]
```

### Analysis Features
- **Dependency Graphs**: Visual representation of code dependencies
- **Circular Dependency Detection**: Identify problematic dependency cycles
- **Module Coupling Analysis**: Analyze module relationships
- **Impact Analysis**: Determine impact of changes

### Usage
```python
from pysearch.analysis import DependencyAnalyzer

analyzer = DependencyAnalyzer()
graph = analyzer.analyze_directory("./src")

# Check for circular dependencies
from pysearch.analysis.dependency_analysis import CircularDependencyDetector
detector = CircularDependencyDetector(graph)
cycles = detector.find_cycles()
```

---

## Language Detection

### Overview
Provides programming language identification for files.

### Key Functions
```python
def detect_language(file_path: Path | str) -> Language
def detect_language_from_content(content: str, filename: str) -> Language
def get_supported_languages() -> list[Language]
def get_language_extensions(language: Language) -> list[str]
```

### Supported Languages
- **Python**: `.py`, `.pyx`, `.pyi`
- **JavaScript**: `.js`, `.jsx`, `.mjs`
- **TypeScript**: `.ts`, `.tsx`
- **Java**: `.java`
- **C/C++**: `.c`, `.cpp`, `.h`, `.hpp`
- **Go**: `.go`
- **Rust**: `.rs`
- **PHP**: `.php`
- **Ruby**: `.rb`
- **Swift**: `.swift`
- **Kotlin**: `.kt`, `.kts`
- **Scala**: `.scala`
- **Shell**: `.sh`, `.bash`, `.zsh`
- **SQL**: `.sql`
- **HTML**: `.html`, `.htm`
- **CSS**: `.css`
- **JSON**: `.json`
- **YAML**: `.yaml`, `.yml`
- **Markdown**: `.md`

### Usage
```python
from pysearch.analysis import detect_language, get_supported_languages

# Detect language from file
language = detect_language("example.py")
print(language)  # Language.PYTHON

# Get all supported languages
languages = get_supported_languages()
```

---

## Language Support

### Overview
Enhanced language-specific processing for better code analysis.

### Key Classes
```python
class LanguageProcessor:
    def extract_entities(self, content: str, language: Language) -> list[CodeEntity]
    def analyze_dependencies(self, content: str, language: Language) -> list[Dependency]
    def get_syntax_tree(self, content: str, language: Language) -> Any
```

### Capabilities
- **Entity Extraction**: Functions, classes, methods, etc.
- **Dependency Analysis**: Import/require statements
- **Syntax Tree**: AST/tree-sitter parsing
- **Language-specific Features**: Custom processing per language

---

## Content Addressing

### Overview
Provides content hashing and addressing for efficient deduplication.

### Key Functions
```python
def compute_content_hash(content: str) -> str
def compute_file_hash(file_path: Path) -> str
def create_content_address(file_path: Path) -> ContentAddress
```

### Content Address
```python
@dataclass
class ContentAddress:
    path: str              # File path
    content_hash: str      # SHA256 hash
    size: int              # File size
    mtime: float           # Modification time
```

---

## GraphRAG (Graph-based RAG)

### Overview
Graph-based retrieval-augmented generation for enhanced code understanding.

### Key Components

#### GraphRAG Core
```python
class KnowledgeGraph:
    entities: dict[str, CodeEntity]
    relationships: list[EntityRelationship]
    embeddings: dict[str, np.ndarray]

    def add_entity(self, entity: CodeEntity) -> None
    def add_relationship(self, rel: EntityRelationship) -> None
    def find_related(self, entity_id: str, hops: int = 1) -> list[CodeEntity]
```

#### GraphRAG Engine
```python
class GraphRAGEngine:
    def __init__(self, config: SearchConfig, qdrant_config: QdrantConfig)
    async def initialize(self) -> None
    async def build_knowledge_graph(self, force_rebuild: bool = False) -> bool
    async def query_graph(self, query: GraphRAGQuery) -> GraphRAGResult | None
    async def close(self) -> None
```

### GraphRAG Query
```python
@dataclass
class GraphRAGQuery:
    query: str                    # Query text
    max_hops: int = 2             # Maximum graph traversal hops
    min_confidence: float = 0.5   # Minimum confidence threshold
    semantic_threshold: float = 0.7  # Semantic similarity threshold
    context_window: int = 5       # Context window size
```

### GraphRAG Features
- **Knowledge Graph Construction**: Automated graph building
- **Vector Integration**: Qdrant vector database integration
- **Multi-hop Traversal**: Navigate relationships across hops
- **Semantic Search**: Embedding-based similarity
- **Context Extraction**: Rich context around matches

### Usage
```python
from pysearch import PySearch, SearchConfig
from pysearch.types import GraphRAGQuery

# Enable GraphRAG
config = SearchConfig(
    paths=["./src"],
    enable_graphrag=True,
    qdrant_enabled=True
)

engine = PySearch(config)
await engine.initialize_graphrag()
await engine.build_knowledge_graph()

# Query the graph
query = GraphRAGQuery(
    query="database connection handling",
    max_hops=2
)

result = await engine.graphrag_search(query)
```

---

## Dependencies

### Internal Dependencies
- `pysearch.core`: Configuration and types
- `pysearch.utils`: File utilities
- `pysearch.storage`: Vector database integration

### External Dependencies
- `tree-sitter`: Parser library (optional, for enhanced language support)
- `qdrant-client`: Vector database client (for GraphRAG)
- `numpy`: Numerical operations (for embeddings)

---

## Testing

### Integration Tests
Located in `tests/integration/`:
- `test_dependency_analysis.py` - Dependency analysis tests
- `test_graphrag.py` - GraphRAG tests

### Unit Tests
Located in `tests/unit/core/`:
- `test_language_detection_*.py` - Language detection tests

---

## Common Usage Patterns

### Dependency Analysis
```python
from pysearch.analysis import DependencyAnalyzer
from pysearch.api import PySearch, SearchConfig

config = SearchConfig(paths=["./src"])
engine = PySearch(config)

# Analyze dependencies
graph = engine.analyze_dependencies("./src")

# Get metrics
metrics = engine.get_dependency_metrics(graph)
print(f"Total modules: {metrics.total_modules}")
print(f"Circular dependencies: {metrics.circular_dependencies}")

# Find impact
impact = engine.find_dependency_impact("src.core.database")
print(f"Modules affected: {impact['total_affected_modules']}")
```

### Language Detection
```python
from pysearch.analysis import detect_language

# Detect language from file extension
language = detect_language("example.py")

# Use language-specific processing
if language == Language.PYTHON:
    # Python-specific processing
    pass
```

### GraphRAG
```python
from pysearch.types import GraphRAGQuery

# Build knowledge graph
await engine.build_knowledge_graph(force_rebuild=True)

# Query with GraphRAG
query = GraphRAGQuery(
    query="error handling patterns",
    max_hops=2,
    min_confidence=0.6
)

result = await engine.graphrag_search(query)
```

---

## Related Files
- `README.md` - Module overview
- `docs/architecture.md` - Architecture details
- `docs/graphrag_guide.md` - GraphRAG guide
- `docs/api/language_detection.md` - Language detection API
