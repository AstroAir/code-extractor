"""
Code dependency analysis module for pysearch.

This module provides comprehensive dependency analysis capabilities including:
- Import graph generation and visualization
- Dependency tracking across files and modules
- Circular dependency detection
- Dependency impact analysis
- Module coupling metrics
- Dead code detection based on usage patterns

Classes:
    ImportNode: Represents a single import statement
    DependencyGraph: Graph structure for tracking dependencies
    DependencyAnalyzer: Main analyzer for dependency operations
    CircularDependencyDetector: Specialized detector for circular dependencies
    DependencyMetrics: Metrics calculator for dependency analysis

Features:
    - Multi-language import detection (Python, JavaScript, Java, etc.)
    - Transitive dependency resolution
    - Dependency clustering and grouping
    - Impact analysis for code changes
    - Visualization-ready graph export

Example:
    Basic dependency analysis:
        >>> from pysearch.dependency_analysis import DependencyAnalyzer
        >>> analyzer = DependencyAnalyzer()
        >>> graph = analyzer.analyze_directory("./src")
        >>> print(f"Found {len(graph.nodes)} modules with {len(graph.edges)} dependencies")

    Circular dependency detection:
        >>> detector = CircularDependencyDetector(graph)
        >>> cycles = detector.find_cycles()
        >>> for cycle in cycles:
        ...     print(f"Circular dependency: {' -> '.join(cycle)}")
"""

from __future__ import annotations

import ast
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.types import Language
from ..utils.helpers import read_text_safely
from .language_detection import detect_language


@dataclass
class ImportNode:
    """Represents a single import statement with metadata."""

    module: str  # The imported module/package name
    alias: str | None = None  # Import alias (as name)
    from_module: str | None = None  # For 'from X import Y' statements
    line_number: int = 0
    is_relative: bool = False  # Relative import (e.g., from .module import x)
    import_type: str = "import"  # "import", "from", "require", etc.
    language: Language = Language.PYTHON
    file_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyEdge:
    """Represents a dependency relationship between two modules."""

    source: str  # Source module
    target: str  # Target module
    import_nodes: list[ImportNode] = field(default_factory=list)
    weight: int = 1  # Number of imports between modules
    edge_type: str = "direct"  # "direct", "transitive", "circular"


@dataclass
class DependencyMetrics:
    """Metrics for dependency analysis."""

    total_modules: int = 0
    total_dependencies: int = 0
    circular_dependencies: int = 0
    max_depth: int = 0
    average_dependencies_per_module: float = 0.0
    coupling_metrics: dict[str, float] = field(default_factory=dict)
    dead_modules: list[str] = field(default_factory=list)
    highly_coupled_modules: list[str] = field(default_factory=list)


class DependencyGraph:
    """
    Graph structure for tracking module dependencies.

    Provides efficient storage and querying of dependency relationships
    with support for various graph algorithms and analysis operations.
    """

    def __init__(self) -> None:
        self.nodes: set[str] = set()  # Module names
        self.edges: dict[str, list[DependencyEdge]] = defaultdict(list)
        self.reverse_edges: dict[str, list[DependencyEdge]] = defaultdict(list)
        self.import_map: dict[str, list[ImportNode]] = defaultdict(list)

    def add_node(self, module: str) -> None:
        """Add a module node to the graph."""
        self.nodes.add(module)

    def add_edge(self, source: str, target: str, import_node: ImportNode) -> None:
        """Add a dependency edge between two modules."""
        self.add_node(source)
        self.add_node(target)

        # Check if edge already exists
        existing_edge = None
        for edge in self.edges[source]:
            if edge.target == target:
                existing_edge = edge
                break

        if existing_edge:
            existing_edge.import_nodes.append(import_node)
            existing_edge.weight += 1
        else:
            edge = DependencyEdge(
                source=source, target=target, import_nodes=[import_node], weight=1
            )
            self.edges[source].append(edge)
            self.reverse_edges[target].append(edge)

        self.import_map[source].append(import_node)

    def get_dependencies(self, module: str) -> list[str]:
        """Get direct dependencies of a module."""
        return [edge.target for edge in self.edges.get(module, [])]

    def get_dependents(self, module: str) -> list[str]:
        """Get modules that depend on this module."""
        return [edge.source for edge in self.reverse_edges.get(module, [])]

    def get_transitive_dependencies(self, module: str, max_depth: int = 10) -> set[str]:
        """Get all transitive dependencies of a module."""
        visited = set()
        queue = deque([(module, 0)])

        while queue:
            current, depth = queue.popleft()
            if current in visited or depth >= max_depth:
                continue

            visited.add(current)
            for dependency in self.get_dependencies(current):
                if dependency not in visited:
                    queue.append((dependency, depth + 1))

        visited.discard(module)  # Remove the starting module
        return visited

    def has_path(self, source: str, target: str, max_depth: int = 10) -> bool:
        """Check if there's a path from source to target."""
        if source == target:
            return True

        visited = set()
        queue = deque([(source, 0)])

        while queue:
            current, depth = queue.popleft()
            if current in visited or depth >= max_depth:
                continue

            visited.add(current)
            if current == target:
                return True

            for dependency in self.get_dependencies(current):
                if dependency not in visited:
                    queue.append((dependency, depth + 1))

        return False

    def to_dict(self) -> dict[str, Any]:
        """Export graph to dictionary format for serialization."""
        return {
            "nodes": list(self.nodes),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "type": edge.edge_type,
                    "imports": len(edge.import_nodes),
                }
                for edges in self.edges.values()
                for edge in edges
            ],
        }


class CircularDependencyDetector:
    """
    Specialized detector for circular dependencies using Tarjan's algorithm.

    Efficiently detects strongly connected components in the dependency graph
    which represent circular dependency cycles.
    """

    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.index_counter = 0
        self.stack: list[str] = []
        self.lowlinks: dict[str, int] = {}
        self.index: dict[str, int] = {}
        self.on_stack: set[str] = set()
        self.sccs: list[list[str]] = []

    def find_cycles(self) -> list[list[str]]:
        """
        Find all circular dependency cycles in the graph.

        Returns:
            List of cycles, where each cycle is a list of module names
        """
        self.sccs = []
        self.index_counter = 0
        self.stack = []
        self.lowlinks = {}
        self.index = {}
        self.on_stack = set()

        for node in self.graph.nodes:
            if node not in self.index:
                self._strongconnect(node)

        # Filter out single-node SCCs (not cycles)
        cycles = [scc for scc in self.sccs if len(scc) > 1]
        return cycles

    def _strongconnect(self, node: str) -> None:
        """Tarjan's strongly connected components algorithm."""
        self.index[node] = self.index_counter
        self.lowlinks[node] = self.index_counter
        self.index_counter += 1
        self.stack.append(node)
        self.on_stack.add(node)

        for dependency in self.graph.get_dependencies(node):
            if dependency not in self.index:
                self._strongconnect(dependency)
                self.lowlinks[node] = min(self.lowlinks[node], self.lowlinks[dependency])
            elif dependency in self.on_stack:
                self.lowlinks[node] = min(self.lowlinks[node], self.index[dependency])

        if self.lowlinks[node] == self.index[node]:
            scc = []
            while True:
                w = self.stack.pop()
                self.on_stack.remove(w)
                scc.append(w)
                if w == node:
                    break
            self.sccs.append(scc)


class DependencyAnalyzer:
    """
    Main analyzer for code dependency operations.

    Provides comprehensive dependency analysis including import extraction,
    graph building, metrics calculation, and various analysis operations.
    """

    def __init__(self) -> None:
        self.graph = DependencyGraph()
        self.language_parsers = {
            Language.PYTHON: self._parse_python_imports,
            Language.JAVASCRIPT: self._parse_javascript_imports,
            Language.TYPESCRIPT: self._parse_javascript_imports,  # Similar to JS
            Language.JAVA: self._parse_java_imports,
            Language.CSHARP: self._parse_csharp_imports,
            Language.GO: self._parse_go_imports,
        }

    def analyze_file(self, file_path: Path) -> list[ImportNode]:
        """
        Analyze a single file for import statements.

        Args:
            file_path: Path to the file to analyze

        Returns:
            List of import nodes found in the file
        """
        language = detect_language(file_path)
        if language not in self.language_parsers:
            return []

        try:
            content = read_text_safely(file_path)
            if not content:
                return []

            parser = self.language_parsers[language]
            imports = parser(content, file_path)

            # Add imports to graph
            module_name = self._get_module_name(file_path, language)
            for import_node in imports:
                import_node.file_path = file_path
                target_module = import_node.from_module or import_node.module
                self.graph.add_edge(module_name, target_module, import_node)

            return imports

        except Exception:
            return []

    def analyze_directory(self, directory: Path, recursive: bool = True) -> DependencyGraph:
        """
        Analyze all files in a directory for dependencies.

        Args:
            directory: Directory to analyze
            recursive: Whether to analyze subdirectories

        Returns:
            Complete dependency graph for the directory
        """
        self.graph = DependencyGraph()

        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"

        for file_path in directory.glob(pattern):
            if file_path.is_file():
                self.analyze_file(file_path)

        return self.graph

    def _parse_python_imports(self, content: str, file_path: Path) -> list[ImportNode]:
        """Parse Python import statements using AST."""
        imports = []

        try:
            tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        import_node = ImportNode(
                            module=alias.name,
                            alias=alias.asname,
                            line_number=node.lineno,
                            import_type="import",
                            language=Language.PYTHON,
                        )
                        imports.append(import_node)

                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    level = node.level or 0
                    is_relative = level > 0

                    for alias in node.names:
                        import_node = ImportNode(
                            module=alias.name,
                            alias=alias.asname,
                            from_module=module,
                            line_number=node.lineno,
                            is_relative=is_relative,
                            import_type="from",
                            language=Language.PYTHON,
                            metadata={"level": level},
                        )
                        imports.append(import_node)

        except SyntaxError:
            # Fallback to regex parsing
            imports.extend(self._parse_python_imports_regex(content))

        return imports

    def _parse_python_imports_regex(self, content: str) -> list[ImportNode]:
        """Fallback regex-based Python import parsing."""
        imports = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # import module [as alias]
            import_match = re.match(
                r"import\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*(?:as\s+([a-zA-Z_][a-zA-Z0-9_]*))?", line
            )
            if import_match:
                module, alias = import_match.groups()
                import_node = ImportNode(
                    module=module,
                    alias=alias,
                    line_number=line_num,
                    import_type="import",
                    language=Language.PYTHON,
                )
                imports.append(import_node)

            # from module import name [as alias]
            from_match = re.match(
                r"from\s+(\.*)([a-zA-Z_][a-zA-Z0-9_.]*)\s+import\s+"
                r"([a-zA-Z_][a-zA-Z0-9_.*]*)\s*(?:as\s+([a-zA-Z_][a-zA-Z0-9_]*))?",
                line,
            )
            if from_match:
                dots, module, name, alias = from_match.groups()
                is_relative = bool(dots)
                import_node = ImportNode(
                    module=name,
                    alias=alias,
                    from_module=module,
                    line_number=line_num,
                    is_relative=is_relative,
                    import_type="from",
                    language=Language.PYTHON,
                )
                imports.append(import_node)

        return imports

    def _parse_javascript_imports(self, content: str, file_path: Path) -> list[ImportNode]:
        """Parse JavaScript/TypeScript import statements."""
        imports = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # import name from 'module'
            import_match = re.match(
                r'import\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s+from\s+["\']([^"\']+)["\']', line
            )
            if import_match:
                name, module = import_match.groups()
                import_node = ImportNode(
                    module=module,
                    alias=name,
                    line_number=line_num,
                    import_type="import",
                    language=Language.JAVASCRIPT,
                )
                imports.append(import_node)

            # import { name } from 'module'
            destructure_match = re.match(
                r'import\s+\{\s*([^}]+)\s*\}\s+from\s+["\']([^"\']+)["\']', line
            )
            if destructure_match:
                names, module = destructure_match.groups()
                for name in names.split(","):
                    name = name.strip()
                    alias = None
                    if " as " in name:
                        name, alias = name.split(" as ")
                        name, alias = name.strip(), alias.strip()

                    import_node = ImportNode(
                        module=name,
                        alias=alias,
                        from_module=module,
                        line_number=line_num,
                        import_type="destructure",
                        language=Language.JAVASCRIPT,
                    )
                    imports.append(import_node)

            # require('module')
            require_match = re.match(
                r'(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*require\s*\(\s*["\']([^"\']+)["\']\s*\)',
                line,
            )
            if require_match:
                name, module = require_match.groups()
                import_node = ImportNode(
                    module=module,
                    alias=name,
                    line_number=line_num,
                    import_type="require",
                    language=Language.JAVASCRIPT,
                )
                imports.append(import_node)

        return imports

    def _parse_java_imports(self, content: str, file_path: Path) -> list[ImportNode]:
        """Parse Java import statements."""
        imports = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # import package.Class;
            import_match = re.match(r"import\s+(?:static\s+)?([a-zA-Z_][a-zA-Z0-9_.]*)\s*;", line)
            if import_match:
                module = import_match.group(1)
                is_static = "static" in line
                import_node = ImportNode(
                    module=module,
                    line_number=line_num,
                    import_type="static" if is_static else "import",
                    language=Language.JAVA,
                )
                imports.append(import_node)

        return imports

    def _parse_csharp_imports(self, content: str, file_path: Path) -> list[ImportNode]:
        """Parse C# using statements."""
        imports = []
        lines = content.split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # using Namespace;
            using_match = re.match(r"using\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;", line)
            if using_match:
                namespace = using_match.group(1)
                import_node = ImportNode(
                    module=namespace,
                    line_number=line_num,
                    import_type="using",
                    language=Language.CSHARP,
                )
                imports.append(import_node)

        return imports

    def _parse_go_imports(self, content: str, file_path: Path) -> list[ImportNode]:
        """Parse Go import statements."""
        imports = []
        lines = content.split("\n")

        in_import_block = False
        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            # import "package"
            single_import = re.match(r'import\s+"([^"]+)"', line)
            if single_import:
                package = single_import.group(1)
                import_node = ImportNode(
                    module=package, line_number=line_num, import_type="import", language=Language.GO
                )
                imports.append(import_node)

            # import ( ... )
            if line.startswith("import ("):
                in_import_block = True
                continue

            if in_import_block:
                if line == ")":
                    in_import_block = False
                    continue

                # "package" or alias "package"
                import_match = re.match(r'(?:([a-zA-Z_][a-zA-Z0-9_]*)\s+)?"([^"]+)"', line)
                if import_match:
                    alias, package = import_match.groups()
                    import_node = ImportNode(
                        module=package,
                        alias=alias,
                        line_number=line_num,
                        import_type="import",
                        language=Language.GO,
                    )
                    imports.append(import_node)

        return imports

    def _get_module_name(self, file_path: Path, language: Language) -> str:
        """Get module name from file path based on language conventions."""
        if language == Language.PYTHON:
            # Convert file path to Python module notation
            parts = file_path.with_suffix("").parts
            if parts[-1] == "__init__":
                parts = parts[:-1]
            return ".".join(parts)

        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            # Use relative path for JS/TS
            return str(file_path.with_suffix(""))

        elif language == Language.JAVA:
            # Extract package from file content if possible
            try:
                content = read_text_safely(file_path)
                if content:
                    package_match = re.search(r"package\s+([a-zA-Z_][a-zA-Z0-9_.]*)\s*;", content)
                    if package_match:
                        package = package_match.group(1)
                        class_name = file_path.stem
                        return f"{package}.{class_name}"
            except Exception:
                pass
            return file_path.stem

        else:
            # Default to file name
            return file_path.stem

    def calculate_metrics(self) -> DependencyMetrics:
        """
        Calculate comprehensive dependency metrics.

        Returns:
            DependencyMetrics object with various analysis results
        """
        metrics = DependencyMetrics()

        # Basic counts
        metrics.total_modules = len(self.graph.nodes)
        metrics.total_dependencies = sum(len(edges) for edges in self.graph.edges.values())

        if metrics.total_modules > 0:
            metrics.average_dependencies_per_module = (
                metrics.total_dependencies / metrics.total_modules
            )

        # Circular dependencies
        detector = CircularDependencyDetector(self.graph)
        cycles = detector.find_cycles()
        metrics.circular_dependencies = len(cycles)

        # Maximum dependency depth
        metrics.max_depth = self._calculate_max_depth()

        # Coupling metrics
        metrics.coupling_metrics = self._calculate_coupling_metrics()

        # Dead modules (modules with no dependents)
        metrics.dead_modules = [
            module for module in self.graph.nodes if not self.graph.get_dependents(module)
        ]

        # Highly coupled modules (modules with many dependencies)
        dependency_counts = {
            module: len(self.graph.get_dependencies(module)) for module in self.graph.nodes
        }
        if dependency_counts:
            avg_deps = sum(dependency_counts.values()) / len(dependency_counts)
            threshold = avg_deps * 2  # Modules with 2x average dependencies
            metrics.highly_coupled_modules = [
                module for module, count in dependency_counts.items() if count > threshold
            ]

        return metrics

    def _calculate_max_depth(self) -> int:
        """Calculate maximum dependency depth in the graph."""
        max_depth = 0

        for module in self.graph.nodes:
            depth = self._get_module_depth(module)
            max_depth = max(max_depth, depth)

        return max_depth

    def _get_module_depth(self, module: str, visited: set[str] | None = None) -> int:
        """Get the maximum dependency depth for a specific module."""
        if visited is None:
            visited = set()

        if module in visited:
            return 0  # Circular dependency, stop here

        visited.add(module)
        dependencies = self.graph.get_dependencies(module)

        if not dependencies:
            return 0

        max_child_depth = 0
        for dep in dependencies:
            child_depth = self._get_module_depth(dep, visited.copy())
            max_child_depth = max(max_child_depth, child_depth)

        return max_child_depth + 1

    def _calculate_coupling_metrics(self) -> dict[str, Any]:
        """Calculate various coupling metrics."""
        metrics: dict[str, Any] = {}

        if not self.graph.nodes:
            return metrics

        # Afferent coupling (Ca) - number of modules that depend on this module
        # Efferent coupling (Ce) - number of modules this module depends on
        afferent_coupling = {}
        efferent_coupling = {}

        for module in self.graph.nodes:
            afferent_coupling[module] = len(self.graph.get_dependents(module))
            efferent_coupling[module] = len(self.graph.get_dependencies(module))

        # Instability (I) = Ce / (Ca + Ce)
        # Ranges from 0 (stable) to 1 (unstable)
        instability = {}
        for module in self.graph.nodes:
            ca = afferent_coupling[module]
            ce = efferent_coupling[module]
            if ca + ce > 0:
                instability[module] = ce / (ca + ce)
            else:
                instability[module] = 0.0

        # Average metrics
        metrics["average_afferent_coupling"] = sum(afferent_coupling.values()) / len(
            afferent_coupling
        )
        metrics["average_efferent_coupling"] = sum(efferent_coupling.values()) / len(
            efferent_coupling
        )
        metrics["average_instability"] = sum(instability.values()) / len(instability)

        # Find most/least stable modules
        if instability:
            most_stable = min(instability.items(), key=lambda x: x[1])
            most_unstable = max(instability.items(), key=lambda x: x[1])
            metrics["most_stable_module"] = most_stable[0]
            metrics["most_stable_score"] = most_stable[1]
            metrics["most_unstable_module"] = most_unstable[0]
            metrics["most_unstable_score"] = most_unstable[1]

        return metrics

    def find_impact_analysis(self, module: str) -> dict[str, Any]:
        """
        Analyze the impact of changes to a specific module.

        Args:
            module: Module to analyze

        Returns:
            Dictionary with impact analysis results
        """
        if module not in self.graph.nodes:
            return {"error": f"Module {module} not found in dependency graph"}

        # Direct dependents (modules that would be immediately affected)
        direct_dependents = self.graph.get_dependents(module)

        # Transitive dependents (all modules that could be affected)
        transitive_dependents = set()
        for dependent in direct_dependents:
            transitive_dependents.update(self.graph.get_transitive_dependencies(dependent))

        # Dependencies that would need to be considered
        direct_dependencies = self.graph.get_dependencies(module)
        transitive_dependencies = self.graph.get_transitive_dependencies(module)

        return {
            "module": module,
            "direct_dependents": direct_dependents,
            "transitive_dependents": list(transitive_dependents),
            "total_affected_modules": len(direct_dependents) + len(transitive_dependents),
            "direct_dependencies": direct_dependencies,
            "transitive_dependencies": list(transitive_dependencies),
            "impact_score": self._calculate_impact_score(
                len(direct_dependents), len(transitive_dependents)
            ),
        }

    def _calculate_impact_score(self, direct: int, transitive: int) -> float:
        """Calculate a normalized impact score (0.0 to 1.0)."""
        if not self.graph.nodes:
            return 0.0

        total_modules = len(self.graph.nodes)
        # Weight direct impacts more heavily than transitive
        weighted_impact = (direct * 2 + transitive) / (total_modules * 3)
        return min(weighted_impact, 1.0)

    def suggest_refactoring_opportunities(self) -> list[dict[str, Any]]:
        """
        Suggest refactoring opportunities based on dependency analysis.

        Returns:
            List of refactoring suggestions with rationale
        """
        suggestions = []
        metrics = self.calculate_metrics()

        # Suggest breaking circular dependencies
        detector = CircularDependencyDetector(self.graph)
        cycles = detector.find_cycles()
        for cycle in cycles:
            suggestions.append(
                {
                    "type": "break_circular_dependency",
                    "priority": "high",
                    "modules": cycle,
                    "description": f"Break circular dependency: {' -> '.join(cycle)}",
                    "rationale": "Circular dependencies make code harder to test and maintain",
                }
            )

        # Suggest splitting highly coupled modules
        for module in metrics.highly_coupled_modules:
            dependencies = self.graph.get_dependencies(module)
            suggestion: dict[str, Any] = {
                "type": "reduce_coupling",
                "priority": "medium",
                "module": module,
                "dependency_count": len(dependencies),
                "description": f"Consider splitting {module} (has {len(dependencies)} dependencies)",
                "rationale": "High coupling makes modules harder to maintain and test",
            }
            suggestions.append(suggestion)

        # Suggest removing dead modules
        for module in metrics.dead_modules:
            suggestions.append(
                {
                    "type": "remove_dead_code",
                    "priority": "low",
                    "module": module,
                    "description": f"Consider removing unused module: {module}",
                    "rationale": "Dead code increases maintenance burden",
                }
            )

        return suggestions
