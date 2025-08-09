# Roadmap

The following is the phased planning for pysearch, sorted by priority and impact.

## Near-term (Priority)

- **Enhanced Semantic Search**: Introduce optional embedding backends (local/remote), support toggles and pluggable strategies
- **Parallel Indexing**: Multi-process/multi-threading to improve large repository scanning performance
- **Result Deduplication & Aggregation**: Unified aggregation across matchers and files
- **Richer Output**: Markdown/HTML export with highlighting and jump links
- **IDE Integration**: VS Code extension (minimal viable), protocol-based CLI output

## Medium-term

- **Persistent Caching**: SQLite/disk structured caching for cross-session reuse
- **Cross-language Foundation**: Abstract universal matching pipeline, explore basic support for TS/Go
- **Enhanced AST Queries**: DSL-based with pattern templates and placeholders
- **Performance Profiling**: Generate flame graphs/metrics export for benchmarking and regression analysis

## Long-term

- **Distributed Indexing & Search**: Multi-node collaboration for ultra-large monolithic repositories
- **Smarter Ranking**: Learning to Rank (LTR) with interactive feedback
- **Team Workflows**: Server/service deployment with unified search portal

Welcome to contribute suggestions or participate in implementation through Issues/PRs. See CONTRIBUTING.md for detailed contribution process.
