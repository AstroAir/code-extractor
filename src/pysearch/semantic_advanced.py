"""
Advanced semantic search module for pysearch.

This module provides sophisticated semantic search capabilities including:
- Embedding-based vector similarity search
- Enhanced concept matching with contextual understanding
- Code structure-aware semantic analysis
- Multi-modal semantic scoring (text + structure + context)

Classes:
    SemanticEmbedding: Lightweight embedding model for code similarity
    CodeSemanticAnalyzer: Advanced semantic analysis for code structures
    SemanticSearchEngine: Main engine for semantic search operations

Features:
    - TF-IDF based lightweight embeddings (no external dependencies)
    - Code structure awareness (functions, classes, imports)
    - Contextual similarity scoring
    - Semantic concept expansion
    - Multi-language semantic patterns

Example:
    Basic semantic search:
        >>> from pysearch.semantic_advanced import SemanticSearchEngine
        >>> engine = SemanticSearchEngine()
        >>> results = engine.search_semantic("database connection", content)
        >>> print(f"Semantic similarity: {results[0].semantic_score}")

    Advanced semantic analysis:
        >>> analyzer = CodeSemanticAnalyzer()
        >>> concepts = analyzer.extract_concepts(code_content)
        >>> similarity = analyzer.calculate_similarity("web api", concepts)
"""

from __future__ import annotations

import ast
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .language_detection import detect_language
from .types import Language, SearchItem
from .utils import split_lines_keepends


@dataclass
class SemanticConcept:
    """Represents a semantic concept extracted from code."""
    
    name: str
    category: str  # function, class, variable, import, comment, etc.
    context: str  # surrounding context
    confidence: float  # confidence score 0.0-1.0
    line_number: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticMatch:
    """Represents a semantic search match with detailed scoring."""
    
    item: SearchItem
    semantic_score: float
    concept_matches: list[SemanticConcept]
    structural_score: float
    contextual_score: float
    combined_score: float


class SemanticEmbedding:
    """
    Lightweight embedding model for code similarity using TF-IDF.
    
    This class provides vector-based similarity without external dependencies,
    optimized for code content and programming concepts.
    """
    
    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.idf_scores: dict[str, float] = {}
        self.is_fitted = False
        
        # Code-specific stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
            'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
    
    def _tokenize_code(self, text: str) -> list[str]:
        """
        Tokenize code text into meaningful tokens.
        
        Args:
            text: Code content to tokenize
            
        Returns:
            List of tokens extracted from the code
        """
        # Extract identifiers, keywords, and meaningful terms
        tokens = []
        
        # Programming identifiers (camelCase, snake_case, etc.)
        identifier_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        identifiers = re.findall(identifier_pattern, text)
        tokens.extend(identifiers)
        
        # Split camelCase and snake_case
        for identifier in identifiers:
            # Split camelCase
            camel_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', identifier)
            tokens.extend([part.lower() for part in camel_parts if len(part) > 1])
            
            # Split snake_case
            snake_parts = identifier.split('_')
            tokens.extend([part.lower() for part in snake_parts if len(part) > 1])
        
        # Extract string literals content
        string_pattern = r'["\']([^"\']*)["\']'
        strings = re.findall(string_pattern, text)
        for string_content in strings:
            string_tokens = re.findall(r'\b\w+\b', string_content.lower())
            tokens.extend(string_tokens)
        
        # Extract comments content
        comment_patterns = [
            r'#\s*(.+)',  # Python comments
            r'//\s*(.+)',  # C-style comments
            r'/\*\s*(.+?)\s*\*/',  # Multi-line comments
        ]
        for pattern in comment_patterns:
            comments = re.findall(pattern, text, re.DOTALL)
            for comment in comments:
                comment_tokens = re.findall(r'\b\w+\b', comment.lower())
                tokens.extend(comment_tokens)
        
        # Filter out stop words and short tokens
        filtered_tokens = [
            token.lower() for token in tokens 
            if len(token) > 2 and token.lower() not in self.stop_words
        ]
        
        return filtered_tokens
    
    def fit(self, documents: list[str]) -> None:
        """
        Fit the embedding model on a collection of documents.
        
        Args:
            documents: List of code documents to train on
        """
        if not documents:
            return
        
        # Tokenize all documents
        all_tokens = []
        doc_tokens = []
        
        for doc in documents:
            tokens = self._tokenize_code(doc)
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Build vocabulary
        token_counts = Counter(all_tokens)
        self.vocabulary = {token: idx for idx, (token, _) in enumerate(token_counts.most_common())}
        
        # Calculate IDF scores
        doc_count = len(documents)
        token_doc_counts = defaultdict(int)
        
        for tokens in doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                token_doc_counts[token] += 1
        
        for token in self.vocabulary:
            df = token_doc_counts[token]
            idf = math.log(doc_count / (df + 1))  # +1 for smoothing
            self.idf_scores[token] = idf
        
        self.is_fitted = True
    
    def transform(self, text: str) -> dict[int, float]:
        """
        Transform text into TF-IDF vector representation.
        
        Args:
            text: Text to transform
            
        Returns:
            Sparse vector as dictionary {token_id: tfidf_score}
        """
        if not self.is_fitted:
            return {}
        
        tokens = self._tokenize_code(text)
        if not tokens:
            return {}
        
        # Calculate term frequencies
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # Calculate TF-IDF scores
        vector = {}
        for token, count in token_counts.items():
            if token in self.vocabulary:
                tf = count / total_tokens
                idf = self.idf_scores.get(token, 0)
                tfidf = tf * idf
                if tfidf > 0:
                    vector[self.vocabulary[token]] = tfidf
        
        return vector
    
    def cosine_similarity(self, vec1: dict[int, float], vec2: dict[int, float]) -> float:
        """
        Calculate cosine similarity between two sparse vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0.0 to 1.0)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        common_keys = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[key] * vec2[key] for key in common_keys)
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
        mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)


class CodeSemanticAnalyzer:
    """
    Advanced semantic analyzer for code structures.
    
    Provides deep semantic understanding of code including:
    - Function and class semantic roles
    - Import dependency semantics
    - Variable usage patterns
    - Code structure relationships
    """
    
    def __init__(self):
        self.embedding_model = SemanticEmbedding()
        
        # Semantic categories for code elements
        self.semantic_categories = {
            'data_processing': [
                'process', 'transform', 'convert', 'parse', 'format', 'serialize',
                'deserialize', 'encode', 'decode', 'filter', 'map', 'reduce'
            ],
            'database': [
                'db', 'database', 'sql', 'query', 'select', 'insert', 'update',
                'delete', 'connection', 'session', 'transaction', 'commit'
            ],
            'web_api': [
                'api', 'http', 'request', 'response', 'get', 'post', 'put',
                'delete', 'endpoint', 'route', 'handler', 'middleware'
            ],
            'testing': [
                'test', 'assert', 'mock', 'fixture', 'setup', 'teardown',
                'verify', 'check', 'validate', 'expect'
            ],
            'configuration': [
                'config', 'settings', 'options', 'parameters', 'env',
                'environment', 'properties', 'preferences'
            ],
            'security': [
                'auth', 'authentication', 'authorization', 'login', 'password',
                'token', 'encrypt', 'decrypt', 'hash', 'security'
            ],
            'file_io': [
                'file', 'read', 'write', 'open', 'close', 'save', 'load',
                'path', 'directory', 'folder', 'stream'
            ],
            'networking': [
                'network', 'socket', 'client', 'server', 'connection',
                'protocol', 'tcp', 'udp', 'http', 'https'
            ]
        }
    
    def extract_concepts(self, code_content: str, file_path: Path | None = None) -> list[SemanticConcept]:
        """
        Extract semantic concepts from code content.
        
        Args:
            code_content: Source code to analyze
            file_path: Optional file path for context
            
        Returns:
            List of extracted semantic concepts
        """
        concepts = []
        lines = split_lines_keepends(code_content)
        
        # Detect language for language-specific analysis
        language = detect_language(file_path) if file_path else Language.PYTHON
        
        try:
            # Try AST-based analysis for supported languages
            if language == Language.PYTHON:
                concepts.extend(self._extract_python_concepts(code_content, lines))
            else:
                # Fallback to regex-based analysis
                concepts.extend(self._extract_regex_concepts(code_content, lines))
        except Exception:
            # Fallback to regex-based analysis on AST errors
            concepts.extend(self._extract_regex_concepts(code_content, lines))
        
        # Add comment-based concepts
        concepts.extend(self._extract_comment_concepts(code_content, lines))
        
        return concepts

    def _extract_python_concepts(self, code_content: str, lines: list[str]) -> list[SemanticConcept]:
        """Extract concepts using Python AST analysis."""
        concepts = []

        try:
            tree = ast.parse(code_content)

            for node in ast.walk(tree):
                concept = None

                if isinstance(node, ast.FunctionDef):
                    concept = SemanticConcept(
                        name=node.name,
                        category='function',
                        context=self._get_function_context(node, lines),
                        confidence=0.9,
                        line_number=node.lineno,
                        metadata={
                            'args': [arg.arg for arg in node.args.args],
                            'decorators': [self._ast_to_string(dec) for dec in node.decorator_list],
                            'returns': self._ast_to_string(node.returns) if node.returns else None
                        }
                    )

                elif isinstance(node, ast.ClassDef):
                    concept = SemanticConcept(
                        name=node.name,
                        category='class',
                        context=self._get_class_context(node, lines),
                        confidence=0.9,
                        line_number=node.lineno,
                        metadata={
                            'bases': [self._ast_to_string(base) for base in node.bases],
                            'decorators': [self._ast_to_string(dec) for dec in node.decorator_list]
                        }
                    )

                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        concept = SemanticConcept(
                            name=alias.name,
                            category='import',
                            context=f"from {node.module}" if isinstance(node, ast.ImportFrom) else "import",
                            confidence=0.8,
                            line_number=node.lineno,
                            metadata={
                                'module': node.module if isinstance(node, ast.ImportFrom) else None,
                                'alias': alias.asname
                            }
                        )
                        concepts.append(concept)
                    continue

                if concept:
                    concepts.append(concept)

        except SyntaxError:
            pass  # Fallback to regex-based extraction

        return concepts

    def _extract_regex_concepts(self, code_content: str, lines: list[str]) -> list[SemanticConcept]:
        """Extract concepts using regex patterns for any language."""
        concepts = []

        # Function patterns for various languages
        function_patterns = [
            (r'def\s+(\w+)\s*\(', 'function', 0.8),  # Python
            (r'function\s+(\w+)\s*\(', 'function', 0.8),  # JavaScript
            (r'(\w+)\s*\([^)]*\)\s*{', 'function', 0.7),  # C-style
            (r'public\s+\w+\s+(\w+)\s*\(', 'function', 0.8),  # Java/C#
        ]

        # Class patterns
        class_patterns = [
            (r'class\s+(\w+)', 'class', 0.9),  # Python, C++, Java, C#
            (r'interface\s+(\w+)', 'interface', 0.9),  # Java, C#, TypeScript
            (r'struct\s+(\w+)', 'struct', 0.9),  # C, C++, Go
        ]

        # Variable patterns
        variable_patterns = [
            (r'(\w+)\s*=\s*', 'variable', 0.6),  # Assignment
            (r'let\s+(\w+)', 'variable', 0.7),  # JavaScript
            (r'var\s+(\w+)', 'variable', 0.7),  # JavaScript, Go
            (r'const\s+(\w+)', 'constant', 0.8),  # JavaScript, C++
        ]

        all_patterns = function_patterns + class_patterns + variable_patterns

        for line_num, line in enumerate(lines, 1):
            for pattern, category, confidence in all_patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                for match in matches:
                    concept = SemanticConcept(
                        name=match.group(1),
                        category=category,
                        context=line.strip(),
                        confidence=confidence,
                        line_number=line_num
                    )
                    concepts.append(concept)

        return concepts

    def _extract_comment_concepts(self, code_content: str, lines: list[str]) -> list[SemanticConcept]:
        """Extract semantic concepts from comments."""
        concepts = []

        comment_patterns = [
            r'#\s*(.+)',  # Python, shell
            r'//\s*(.+)',  # C-style
            r'/\*\s*(.+?)\s*\*/',  # Multi-line C-style
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern in comment_patterns:
                matches = re.finditer(pattern, line, re.DOTALL)
                for match in matches:
                    comment_text = match.group(1).strip()
                    if len(comment_text) > 10:  # Only meaningful comments
                        concept = SemanticConcept(
                            name=comment_text[:50],  # Truncate long comments
                            category='comment',
                            context=line.strip(),
                            confidence=0.5,
                            line_number=line_num,
                            metadata={'full_text': comment_text}
                        )
                        concepts.append(concept)

        return concepts

    def _get_function_context(self, node: ast.FunctionDef, lines: list[str]) -> str:
        """Get contextual information for a function."""
        context_parts = []

        # Add docstring if available
        if (node.body and isinstance(node.body[0], ast.Expr) and
            isinstance(node.body[0].value, ast.Constant) and
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value.strip()
            context_parts.append(f"Doc: {docstring[:100]}")

        # Add function signature context
        if node.lineno <= len(lines):
            line_content = lines[node.lineno - 1].strip()
            context_parts.append(f"Signature: {line_content}")

        return " | ".join(context_parts)

    def _get_class_context(self, node: ast.ClassDef, lines: list[str]) -> str:
        """Get contextual information for a class."""
        context_parts = []

        # Add base classes
        if node.bases:
            bases = [self._ast_to_string(base) for base in node.bases]
            context_parts.append(f"Inherits: {', '.join(bases)}")

        # Add class line
        if node.lineno <= len(lines):
            line_content = lines[node.lineno - 1].strip()
            context_parts.append(f"Definition: {line_content}")

        return " | ".join(context_parts)

    def _ast_to_string(self, node: ast.AST) -> str:
        """Convert AST node to string representation."""
        try:
            return ast.unparse(node)
        except AttributeError:
            # Fallback for older Python versions
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Constant):
                return str(node.value)
            else:
                return str(type(node).__name__)

    def calculate_similarity(self, query: str, concepts: list[SemanticConcept]) -> float:
        """
        Calculate semantic similarity between query and extracted concepts.

        Args:
            query: Search query
            concepts: List of semantic concepts

        Returns:
            Similarity score (0.0 to 1.0)
        """
        if not concepts:
            return 0.0

        query_lower = query.lower()
        total_score = 0.0
        max_possible_score = 0.0

        for concept in concepts:
            concept_score = 0.0

            # Direct name match
            if query_lower in concept.name.lower():
                concept_score += 0.8 * concept.confidence

            # Context match
            if query_lower in concept.context.lower():
                concept_score += 0.6 * concept.confidence

            # Category-based semantic matching
            category_score = self._calculate_category_similarity(query_lower, concept)
            concept_score += category_score * concept.confidence

            total_score += concept_score
            max_possible_score += concept.confidence

        return min(total_score / max_possible_score, 1.0) if max_possible_score > 0 else 0.0

    def _calculate_category_similarity(self, query: str, concept: SemanticConcept) -> float:
        """Calculate similarity based on semantic categories."""
        for category, keywords in self.semantic_categories.items():
            if any(keyword in query for keyword in keywords):
                if any(keyword in concept.name.lower() or keyword in concept.context.lower()
                       for keyword in keywords):
                    return 0.4  # Moderate semantic match
        return 0.0


class SemanticSearchEngine:
    """
    Main engine for advanced semantic search operations.

    Combines multiple semantic analysis techniques:
    - Vector-based similarity using TF-IDF embeddings
    - Structural code analysis
    - Contextual understanding
    - Multi-modal scoring
    """

    def __init__(self) -> None:
        self.analyzer = CodeSemanticAnalyzer()
        self.embedding_model = SemanticEmbedding()
        self._concept_cache: dict[str, list[SemanticConcept]] = {}
        self._embedding_cache: dict[str, dict[int, float]] = {}

    def fit_corpus(self, documents: list[str]) -> None:
        """
        Fit the semantic model on a corpus of documents.

        Args:
            documents: List of code documents for training
        """
        self.embedding_model.fit(documents)

    def search_semantic(
        self,
        query: str,
        content: str,
        file_path: Path | None = None,
        threshold: float = 0.1
    ) -> list[SemanticMatch]:
        """
        Perform semantic search on content.

        Args:
            query: Semantic search query
            content: Code content to search
            file_path: Optional file path for context
            threshold: Minimum similarity threshold

        Returns:
            List of semantic matches sorted by relevance
        """
        # Extract concepts from content
        cache_key = str(hash(content))
        if cache_key not in self._concept_cache:
            concepts = self.analyzer.extract_concepts(content, file_path)
            self._concept_cache[cache_key] = concepts
        else:
            concepts = self._concept_cache[cache_key]

        # Calculate semantic similarity
        semantic_score = self.analyzer.calculate_similarity(query, concepts)

        # Calculate embedding similarity if model is fitted
        embedding_score = 0.0
        if self.embedding_model.is_fitted:
            query_vector = self.embedding_model.transform(query)
            content_vector = self.embedding_model.transform(content)
            embedding_score = self.embedding_model.cosine_similarity(query_vector, content_vector)

        # Calculate structural score
        structural_score = self._calculate_structural_score(query, concepts)

        # Calculate contextual score
        contextual_score = self._calculate_contextual_score(query, content, concepts)

        # Combine scores with weights
        combined_score = (
            0.4 * semantic_score +
            0.3 * embedding_score +
            0.2 * structural_score +
            0.1 * contextual_score
        )

        if combined_score < threshold:
            return []

        # Find relevant concepts for this query
        relevant_concepts = [
            concept for concept in concepts
            if (query.lower() in concept.name.lower() or
                query.lower() in concept.context.lower() or
                self.analyzer._calculate_category_similarity(query.lower(), concept) > 0)
        ]

        # Create search items for relevant concepts
        matches = []
        lines = split_lines_keepends(content)

        for concept in relevant_concepts:
            # Create a search item for this concept
            start_line = max(0, concept.line_number - 1)
            end_line = min(len(lines), concept.line_number + 1)

            item = SearchItem(
                file=file_path or Path("unknown"),
                start_line=concept.line_number,
                end_line=concept.line_number,
                lines=[lines[start_line:end_line]] if start_line < len(lines) else [],
                match_spans=[(0, (0, len(concept.name)))]
            )

            match = SemanticMatch(
                item=item,
                semantic_score=semantic_score,
                concept_matches=[concept],
                structural_score=structural_score,
                contextual_score=contextual_score,
                combined_score=combined_score
            )
            matches.append(match)

        # Sort by combined score
        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches

    def _calculate_structural_score(self, query: str, concepts: list[SemanticConcept]) -> float:
        """Calculate score based on code structure relevance."""
        if not concepts:
            return 0.0

        query_lower = query.lower()
        structure_weights = {
            'function': 0.8,
            'class': 0.9,
            'import': 0.6,
            'variable': 0.4,
            'comment': 0.3
        }

        total_score = 0.0
        total_weight = 0.0

        for concept in concepts:
            weight = structure_weights.get(concept.category, 0.5)
            if query_lower in concept.name.lower():
                total_score += weight * concept.confidence
            total_weight += weight

        return total_score / total_weight if total_weight > 0 else 0.0

    def _calculate_contextual_score(
        self,
        query: str,
        content: str,
        concepts: list[SemanticConcept]
    ) -> float:
        """Calculate score based on contextual relevance."""
        query_lower = query.lower()

        # Check for query terms in surrounding context
        context_score = 0.0
        lines = split_lines_keepends(content)

        for concept in concepts:
            if concept.line_number <= len(lines):
                # Check surrounding lines for context
                start_line = max(0, concept.line_number - 3)
                end_line = min(len(lines), concept.line_number + 3)

                context_text = ' '.join(lines[start_line:end_line]).lower()

                # Count query term occurrences in context
                query_words = query_lower.split()
                context_matches = sum(1 for word in query_words if word in context_text)

                if context_matches > 0:
                    context_score += (context_matches / len(query_words)) * concept.confidence

        return min(context_score / len(concepts), 1.0) if concepts else 0.0

    def expand_query_semantically(self, query: str) -> list[str]:
        """
        Expand a query with semantically related terms.

        Args:
            query: Original search query

        Returns:
            List of expanded query terms
        """
        expanded_terms = [query]
        query_lower = query.lower()

        # Add terms from semantic categories
        for category, keywords in self.analyzer.semantic_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                expanded_terms.extend(keywords)

        # Add common programming variations
        if '_' in query:
            # Convert snake_case to camelCase
            parts = query.split('_')
            camel_case = parts[0] + ''.join(word.capitalize() for word in parts[1:])
            expanded_terms.append(camel_case)

        # Add plural/singular variations
        if query.endswith('s') and len(query) > 3:
            expanded_terms.append(query[:-1])  # Remove 's'
        else:
            expanded_terms.append(query + 's')  # Add 's'

        return list(set(expanded_terms))  # Remove duplicates
