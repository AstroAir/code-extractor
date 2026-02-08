# Roadmap

The following is the phased planning for pysearch, sorted by priority and impact.

## Completed

The following features from previous roadmap phases have been implemented:

- ✅ **Enhanced Semantic Search**: Optional transformer-based embedding backends (OpenAI, HuggingFace, local models) via `SemanticSearchEngine`
- ✅ **Parallel Indexing**: Multi-process/multi-threading via `DistributedIndexingEngine` with configurable worker pools
- ✅ **Result Deduplication & Aggregation**: Unified deduplication, similarity clustering, and file grouping via `scorer.py`
- ✅ **IDE Integration**: Jump-to-definition, find references, completions, hover info, and diagnostics via `ide_hooks.py`
- ✅ **Persistent Caching**: SQLite-based metadata storage, multi-level cache (in-memory + disk) with LRU eviction
- ✅ **Cross-language Foundation**: Tree-sitter integration for 20+ languages via `LanguageRegistry` and `TreeSitterProcessor`
- ✅ **Performance Profiling**: Real-time profiling, metrics collection, and optimization suggestions via `PerformanceMonitor` and `PerformanceProfiler`
- ✅ **Distributed Indexing**: Single-machine multi-process parallel indexing via `DistributedIndexingEngine`
- ✅ **Vector Database Support**: LanceDB, Qdrant, and Chroma backends with unified `VectorDatabase` interface
- ✅ **GraphRAG**: Knowledge graph construction and graph-based code search via `GraphRAGEngine`
- ✅ **Fuzzy Search**: Multiple algorithms (Levenshtein, Jaro-Winkler, n-gram) via `rapidfuzz`
- ✅ **Boolean Search**: `AND`/`OR`/`NOT` logical query composition
- ✅ **Content-Addressed Caching**: SHA256-based deduplication with cross-branch support
- ✅ **Search History**: Session tracking, bookmarks, and analytics
- ✅ **Advanced Error Handling**: Circuit breaker pattern, recovery manager, error aggregation
- ✅ **MCP Integration**: Model Context Protocol servers for LLM tool integration
- ✅ **Dependency Analysis**: Import graph generation, circular dependency detection, coupling metrics
- ✅ **File Watching**: Real-time file monitoring with automatic index updates

## Near-term (Priority)

- **Full LSP Server**: Complete Language Server Protocol implementation for broader IDE support
- **VS Code Extension**: Native VS Code extension packaging with marketplace distribution
- **Richer Output**: Markdown/HTML export with highlighting and jump links
- **Enhanced AST Queries**: DSL-based pattern templates and placeholders for structural queries
- **Cross-language Search**: Unified semantic search across language boundaries

## Medium-term

- **Learning to Rank**: ML-based result ranking with interactive user feedback
- **Query Understanding**: Natural language query processing for more intuitive searches
- **Polyglot Repositories**: Enhanced support for mixed-language monorepos with cross-language dependency tracking
- **Streaming Results**: Live result streaming for interactive search experiences
- **Plugin System**: Formal plugin API for custom matchers, formatters, and scorers

## Long-term

- **Multi-node Distributed Search**: Search across multiple machines for ultra-large monolithic repositories
- **Team Workflows**: Server/service deployment with unified search portal and shared bookmarks
- **Collaborative Features**: Team search sharing, annotations, and search pattern libraries
- **Code Understanding**: Deep code comprehension beyond pattern matching using LLMs
- **Personalization**: User-specific search optimization and result ranking

Welcome to contribute suggestions or participate in implementation through Issues/PRs. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the detailed contribution process.
