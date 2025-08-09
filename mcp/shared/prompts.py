#!/usr/bin/env python3
"""
MCP Prompt Templates for PySearch

This module implements MCP prompt templates for common search scenarios
including security vulnerabilities, performance bottlenecks, code patterns,
and best practices analysis.

Prompt templates provided:
- Security vulnerability detection
- Performance bottleneck identification
- Code quality assessment
- Architecture pattern analysis
- Testing coverage analysis
- Documentation completeness
- Dependency analysis
- Error handling patterns
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class PromptCategory(Enum):
    """Categories of prompt templates."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    QUALITY = "quality"
    ARCHITECTURE = "architecture"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    DEPENDENCIES = "dependencies"
    ERROR_HANDLING = "error_handling"


@dataclass
class PromptTemplate:
    """Template for MCP prompts."""

    name: str
    category: PromptCategory
    description: str
    prompt_text: str
    suggested_tools: list[str]
    parameters: dict[str, Any]


class MCPPromptManager:
    """
    Manages MCP prompt templates for common search scenarios.

    Provides pre-defined prompts for security analysis, performance
    optimization, code quality assessment, and other common tasks.
    """

    def __init__(self) -> None:
        self.templates = self._initialize_templates()

    def _initialize_templates(self) -> dict[str, PromptTemplate]:
        """Initialize all prompt templates."""
        templates = {}

        # Security vulnerability detection
        templates["security_vulnerabilities"] = PromptTemplate(
            name="Security Vulnerability Detection",
            category=PromptCategory.SECURITY,
            description="Identify potential security vulnerabilities in the codebase",
            prompt_text="""
            Analyze the codebase for potential security vulnerabilities. Focus on:
            
            1. SQL injection vulnerabilities (unsanitized database queries)
            2. Cross-site scripting (XSS) vulnerabilities
            3. Authentication and authorization issues
            4. Input validation problems
            5. Cryptographic weaknesses
            6. File system access vulnerabilities
            7. Command injection possibilities
            8. Insecure data transmission
            
            Use the following search patterns to identify these issues:
            - Search for SQL query construction without parameterization
            - Look for user input handling without validation
            - Find hardcoded credentials or API keys
            - Identify insecure cryptographic practices
            - Check for unsafe file operations
            
            Provide specific examples and recommendations for each finding.
            """,
            suggested_tools=["search_multi_pattern", "search_regex", "search_with_filters"],
            parameters={
                "security_patterns": [
                    r"SELECT.*\+.*",  # SQL concatenation
                    r"eval\s*\(",  # Code evaluation
                    r"exec\s*\(",  # Code execution
                    r"password\s*=\s*[\"'][^\"']+[\"']",  # Hardcoded passwords
                    r"api_key\s*=\s*[\"'][^\"']+[\"']",  # Hardcoded API keys
                ],
                "file_extensions": [".py", ".js", ".php", ".java", ".cs"],
                "exclude_patterns": ["**/test/**", "**/tests/**"],
            },
        )

        # Performance bottleneck identification
        templates["performance_bottlenecks"] = PromptTemplate(
            name="Performance Bottleneck Analysis",
            category=PromptCategory.PERFORMANCE,
            description="Identify potential performance bottlenecks and optimization opportunities",
            prompt_text="""
            Analyze the codebase for performance bottlenecks and optimization opportunities:
            
            1. Inefficient algorithms and data structures
            2. Database query optimization issues
            3. Memory leaks and excessive memory usage
            4. Synchronous operations that could be asynchronous
            5. Unnecessary loops and iterations
            6. Large file operations without streaming
            7. Unoptimized regular expressions
            8. Missing caching mechanisms
            
            Search for:
            - Nested loops with high complexity
            - Database queries in loops (N+1 problem)
            - Large data processing without pagination
            - Synchronous I/O operations
            - Missing indexes or inefficient queries
            
            Provide performance improvement recommendations for each finding.
            """,
            suggested_tools=["search_with_ranking", "analyze_file_content", "search_ast"],
            parameters={
                "performance_patterns": [
                    r"for.*for.*for",  # Nested loops
                    r"while.*while",  # Nested while loops
                    r"\.query\(.*\)",  # Database queries
                    r"sleep\s*\(",  # Blocking sleep calls
                    r"\.join\(\)",  # Thread joins
                ],
                "complexity_threshold": 15.0,
                "file_size_threshold": 100000,  # 100KB
            },
        )

        # Code quality assessment
        templates["code_quality"] = PromptTemplate(
            name="Code Quality Assessment",
            category=PromptCategory.QUALITY,
            description="Assess overall code quality and identify improvement areas",
            prompt_text="""
            Perform a comprehensive code quality assessment:
            
            1. Code complexity and maintainability
            2. Adherence to coding standards and conventions
            3. Code duplication and redundancy
            4. Function and class size appropriateness
            5. Comment quality and documentation
            6. Error handling completeness
            7. Code organization and structure
            8. Dependency management
            
            Analyze:
            - Functions with high cyclomatic complexity
            - Large classes and methods
            - Code duplication patterns
            - Missing or poor documentation
            - Inconsistent naming conventions
            - Poor error handling practices
            
            Provide specific recommendations for quality improvements.
            """,
            suggested_tools=["get_file_statistics", "analyze_file_content", "search_with_filters"],
            parameters={
                "quality_thresholds": {
                    "max_function_lines": 50,
                    "max_class_lines": 500,
                    "min_comment_ratio": 0.1,
                    "max_complexity": 10,
                },
                "include_analysis": True,
            },
        )

        # Architecture pattern analysis
        templates["architecture_patterns"] = PromptTemplate(
            name="Architecture Pattern Analysis",
            category=PromptCategory.ARCHITECTURE,
            description="Analyze architectural patterns and design principles",
            prompt_text="""
            Analyze the codebase architecture and design patterns:
            
            1. Design pattern usage (Singleton, Factory, Observer, etc.)
            2. SOLID principles adherence
            3. Separation of concerns
            4. Dependency injection patterns
            5. MVC/MVP/MVVM architecture
            6. Microservices vs monolithic structure
            7. API design patterns
            8. Data access patterns
            
            Look for:
            - Design pattern implementations
            - Architectural boundaries and layers
            - Dependency relationships
            - Interface and abstraction usage
            - Configuration management patterns
            
            Evaluate architectural quality and suggest improvements.
            """,
            suggested_tools=["search_ast", "search_semantic", "search_multi_pattern"],
            parameters={
                "pattern_keywords": [
                    "singleton",
                    "factory",
                    "observer",
                    "strategy",
                    "decorator",
                    "adapter",
                    "facade",
                    "proxy",
                ],
                "architecture_terms": [
                    "controller",
                    "service",
                    "repository",
                    "model",
                    "view",
                    "interface",
                    "abstract",
                    "dependency",
                ],
            },
        )

        # Testing coverage analysis
        templates["testing_coverage"] = PromptTemplate(
            name="Testing Coverage Analysis",
            category=PromptCategory.TESTING,
            description="Analyze test coverage and testing practices",
            prompt_text="""
            Analyze testing practices and coverage in the codebase:
            
            1. Unit test coverage and quality
            2. Integration test presence
            3. Test organization and structure
            4. Mock and stub usage
            5. Test data management
            6. Performance testing
            7. Security testing
            8. Edge case coverage
            
            Examine:
            - Test file organization and naming
            - Test method coverage for production code
            - Assertion quality and completeness
            - Test data setup and teardown
            - Mock usage patterns
            - Test execution performance
            
            Identify gaps in testing and recommend improvements.
            """,
            suggested_tools=["search_with_filters", "get_file_statistics", "search_fuzzy"],
            parameters={
                "test_patterns": ["test_", "_test", "spec_", "_spec"],
                "test_frameworks": ["pytest", "unittest", "jest", "mocha", "junit"],
                "test_file_extensions": [".test.py", ".spec.js", ".test.js", "_test.py"],
            },
        )

        # Documentation completeness
        templates["documentation_analysis"] = PromptTemplate(
            name="Documentation Completeness Analysis",
            category=PromptCategory.DOCUMENTATION,
            description="Analyze documentation quality and completeness",
            prompt_text="""
            Assess documentation quality and completeness:
            
            1. API documentation coverage
            2. Code comment quality and frequency
            3. README and setup documentation
            4. Architecture documentation
            5. User guides and tutorials
            6. Inline documentation standards
            7. Documentation maintenance
            8. Examples and usage patterns
            
            Review:
            - Function and class docstrings
            - API endpoint documentation
            - Configuration documentation
            - Installation and setup guides
            - Code examples and tutorials
            - Change logs and release notes
            
            Identify documentation gaps and improvement opportunities.
            """,
            suggested_tools=["search_regex", "analyze_file_content", "search_with_filters"],
            parameters={
                "doc_file_extensions": [".md", ".rst", ".txt", ".doc"],
                "comment_patterns": [r'""".*?"""', r"'''.*?'''", r"//.*", r"#.*"],
                "min_comment_ratio": 0.15,
            },
        )

        return templates

    def get_available_prompts(self) -> list[dict[str, Any]]:
        """Get list of available prompt templates."""
        return [
            {
                "name": template.name,
                "id": template_id,
                "category": template.category.value,
                "description": template.description,
                "suggested_tools": template.suggested_tools,
            }
            for template_id, template in self.templates.items()
        ]

    def get_prompt_by_id(self, prompt_id: str) -> PromptTemplate | None:
        """Get a specific prompt template by ID."""
        return self.templates.get(prompt_id)

    def get_prompts_by_category(self, category: PromptCategory) -> list[PromptTemplate]:
        """Get all prompts in a specific category."""
        return [template for template in self.templates.values() if template.category == category]

    def generate_prompt_text(self, prompt_id: str, context: dict[str, Any] | None = None) -> str:
        """Generate customized prompt text with context."""
        template = self.get_prompt_by_id(prompt_id)
        if not template:
            raise ValueError(f"Prompt template not found: {prompt_id}")

        prompt_text = template.prompt_text

        # Add context-specific information if provided
        if context:
            if "file_paths" in context:
                prompt_text += (
                    f"\n\nFocus on these specific files: {', '.join(context['file_paths'])}"
                )

            if "languages" in context:
                prompt_text += f"\n\nPrioritize these languages: {', '.join(context['languages'])}"

            if "recent_changes" in context:
                prompt_text += "\n\nPay special attention to recently modified files."

        return prompt_text

    def get_suggested_search_parameters(self, prompt_id: str) -> dict[str, Any]:
        """Get suggested search parameters for a prompt."""
        template = self.get_prompt_by_id(prompt_id)
        if not template:
            raise ValueError(f"Prompt template not found: {prompt_id}")

        return template.parameters
