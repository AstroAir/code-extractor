"""
Code dependency analysis integration.

This module provides comprehensive dependency analysis capabilities,
including dependency graph construction, metrics calculation, and
refactoring suggestions.

Classes:
    DependencyIntegrationManager: Manages dependency analysis functionality

Key Features:
    - Dependency graph construction and analysis
    - Circular dependency detection
    - Module coupling analysis
    - Impact analysis for changes
    - Refactoring opportunity suggestions

Example:
    Using dependency analysis:
        >>> from pysearch.core.integrations.dependency_integration import DependencyIntegrationManager
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> manager = DependencyIntegrationManager(config)
        >>> graph = manager.analyze_dependencies()
        >>> metrics = manager.get_dependency_metrics(graph)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import SearchConfig


class DependencyIntegrationManager:
    """Manages dependency analysis functionality."""

    def __init__(self, config: SearchConfig) -> None:
        self.config = config
        self.dependency_analyzer: Any = None
        self._logger = None

    def _ensure_dependency_analyzer(self) -> None:
        """Lazy load the dependency analyzer to avoid circular imports."""
        if self.dependency_analyzer is None:
            from ...analysis.dependency_analysis import DependencyAnalyzer

            self.dependency_analyzer = DependencyAnalyzer()

    def set_logger(self, logger: Any) -> None:
        """Set logger for dependency analysis operations."""
        self._logger = logger

    def analyze_dependencies(self, directory: Path | None = None, recursive: bool = True) -> Any:
        """
        Analyze code dependencies and build a dependency graph.

        This method performs comprehensive dependency analysis including:
        - Import statement extraction across multiple languages
        - Dependency graph construction
        - Circular dependency detection
        - Module coupling analysis

        Args:
            directory: Directory to analyze (defaults to first configured path)
            recursive: Whether to analyze subdirectories recursively

        Returns:
            DependencyGraph with complete dependency information
        """
        self._ensure_dependency_analyzer()

        if directory is None:
            if self.config.paths:
                directory = Path(self.config.paths[0])
            else:
                directory = Path.cwd()

        if self._logger:
            self._logger.info(f"Starting dependency analysis for: {directory}")

        start_time = None
        try:
            import time

            start_time = time.time()
        except ImportError:
            pass

        # Perform dependency analysis
        if self.dependency_analyzer:
            graph = self.dependency_analyzer.analyze_directory(directory, recursive)
        else:
            # Return empty graph structure if analyzer not available
            graph = type("Graph", (), {"nodes": [], "edges": {}})()

        if self._logger and start_time:
            import time

            elapsed_ms = (time.time() - start_time) * 1000
            self._logger.info(
                f"Dependency analysis completed: {len(graph.nodes)} modules, "
                f"{sum(len(edges) for edges in graph.edges.values())} dependencies, "
                f"time={elapsed_ms:.2f}ms"
            )

        return graph

    def get_dependency_metrics(self, graph: Any | None = None) -> Any:
        """
        Calculate comprehensive dependency metrics.

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            DependencyMetrics with detailed analysis results
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and calculate metrics
        if self.dependency_analyzer:
            self.dependency_analyzer.graph = graph
            return self.dependency_analyzer.calculate_metrics()
        return {}

    def find_dependency_impact(self, module: str, graph: Any | None = None) -> dict[str, Any]:
        """
        Analyze the impact of changes to a specific module.

        This method identifies all modules that would be affected by changes
        to the specified module, helping with impact analysis for refactoring.

        Args:
            module: Module name to analyze
            graph: Dependency graph to use (if None, analyzes current project)

        Returns:
            Dictionary with impact analysis results
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and perform impact analysis
        if self.dependency_analyzer:
            self.dependency_analyzer.graph = graph
            return self.dependency_analyzer.find_impact_analysis(module)  # type: ignore[no-any-return]
        return {}

    def suggest_refactoring_opportunities(self, graph: Any | None = None) -> list[dict[str, Any]]:
        """
        Suggest refactoring opportunities based on dependency analysis.

        Analyzes the dependency graph to identify potential improvements:
        - Circular dependencies to break
        - Highly coupled modules to split
        - Dead code to remove
        - Architecture improvements

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of refactoring suggestions with priorities and rationale
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        # Update analyzer's graph and get suggestions
        if self.dependency_analyzer:
            self.dependency_analyzer.graph = graph
            return self.dependency_analyzer.suggest_refactoring_opportunities()  # type: ignore[no-any-return]
        return []

    def detect_circular_dependencies(self, graph: Any | None = None) -> list[list[str]]:
        """
        Detect circular dependencies in the codebase.

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of circular dependency chains
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        try:
            from ...analysis.dependency_analysis import CircularDependencyDetector

            detector = CircularDependencyDetector(graph)
            return detector.find_cycles()
        except Exception:
            return []

    def get_module_coupling_metrics(self, graph: Any | None = None) -> dict[str, Any]:
        """
        Calculate module coupling metrics.

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            Dictionary with coupling metrics for each module
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        try:
            coupling_metrics = {}

            for module in graph.nodes:
                # Calculate afferent coupling (incoming dependencies)
                # graph.edges values are lists of DependencyEdge objects
                afferent = len(
                    [
                        source
                        for source, edges in graph.edges.items()
                        if source != module
                        and any(edge.target == module for edge in edges)
                    ]
                )

                # Calculate efferent coupling (outgoing dependencies)
                efferent = len(graph.edges.get(module, []))

                # Calculate instability (Ce / (Ca + Ce))
                total_coupling = afferent + efferent
                instability = efferent / total_coupling if total_coupling > 0 else 0

                coupling_metrics[module] = {
                    "afferent_coupling": afferent,
                    "efferent_coupling": efferent,
                    "instability": instability,
                    "total_coupling": total_coupling,
                }

            return coupling_metrics
        except Exception:
            return {}

    def find_dead_code(self, graph: Any | None = None) -> list[str]:
        """
        Identify potentially dead code (unused modules).

        Args:
            graph: Dependency graph to analyze (if None, analyzes current project)

        Returns:
            List of module names that appear to be unused
        """
        self._ensure_dependency_analyzer()

        if graph is None:
            graph = self.analyze_dependencies()

        try:
            dead_modules = []

            for module in graph.nodes:
                # Check if module has no incoming dependencies (not imported by others)
                # graph.edges values are lists of DependencyEdge objects
                has_incoming = any(
                    any(edge.target == module for edge in edges)
                    for edges in graph.edges.values()
                )

                # Skip entry points and main modules
                if (
                    not has_incoming
                    and not module.endswith("__main__")
                    and "main" not in module.lower()
                ):
                    dead_modules.append(module)

            return dead_modules
        except Exception:
            return []

    def export_dependency_graph(self, graph: Any, format: str = "dot") -> str:
        """
        Export dependency graph in specified format.

        Args:
            graph: Dependency graph to export
            format: Export format ("dot", "json", "csv")

        Returns:
            String representation of the graph in specified format
        """
        try:
            if format == "dot":
                return self._export_to_dot(graph)
            elif format == "json":
                return self._export_to_json(graph)
            elif format == "csv":
                return self._export_to_csv(graph)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception:
            return ""

    def _export_to_dot(self, graph: Any) -> str:
        """Export graph to DOT format for Graphviz."""
        lines = ["digraph dependencies {"]

        for source, edges in graph.edges.items():
            for edge in edges:
                lines.append(f'  "{source}" -> "{edge.target}";')

        lines.append("}")
        return "\n".join(lines)

    def _export_to_json(self, graph: Any) -> str:
        """Export graph to JSON format."""
        import json

        graph_data = {
            "nodes": list(graph.nodes),
            "edges": [
                {"source": source, "target": edge.target, "weight": edge.weight}
                for source, edges in graph.edges.items()
                for edge in edges
            ],
        }

        return json.dumps(graph_data, indent=2)

    def _export_to_csv(self, graph: Any) -> str:
        """Export graph to CSV format."""
        lines = ["source,target,weight"]

        for source, edges in graph.edges.items():
            for edge in edges:
                lines.append(f'"{source}","{edge.target}",{edge.weight}')

        return "\n".join(lines)
