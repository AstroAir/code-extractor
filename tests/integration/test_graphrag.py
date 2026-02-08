"""
Tests for GraphRAG functionality.

This module contains comprehensive tests for the GraphRAG (Graph Retrieval-Augmented
Generation) functionality including entity extraction, relationship mapping,
knowledge graph building, and graph-based querying.
"""

from __future__ import annotations

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

from pysearch import (
    CodeEntity,
    EntityRelationship,
    EntityType,
    KnowledgeGraph,
    RelationType,
    SearchConfig,
)
from pysearch.analysis.graphrag import EntityExtractor, RelationshipMapper


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_code() -> str:
    """Sample Python code for testing."""
    return '''
"""Sample module for testing."""

import os
import sys
from typing import List, Dict

class DatabaseManager:
    """Manages database connections."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self) -> bool:
        """Establish database connection."""
        try:
            # Simulate connection logic
            self.connection = "connected"
            return True
        except Exception:
            return False
    
    def execute_query(self, query: str) -> List[Dict]:
        """Execute a database query."""
        if not self.connection:
            raise RuntimeError("Not connected to database")
        
        # Simulate query execution
        return [{"result": "data"}]

def create_manager(connection_string: str) -> DatabaseManager:
    """Factory function for creating database manager."""
    manager = DatabaseManager(connection_string)
    manager.connect()
    return manager

async def async_query(manager: DatabaseManager, query: str) -> List[Dict]:
    """Async wrapper for database queries."""
    return manager.execute_query(query)

# Constants
DEFAULT_CONNECTION = "sqlite:///test.db"
MAX_RETRIES = 3
'''


@pytest.fixture
def sample_javascript_code() -> str:
    """Sample JavaScript code for testing."""
    return """
/**
 * User management module
 */

import { Database } from './database.js';
import { Logger } from './logger.js';

class UserManager {
    constructor(database, logger) {
        this.database = database;
        this.logger = logger;
    }
    
    async createUser(userData) {
        try {
            this.logger.info('Creating user');
            const user = await this.database.insert('users', userData);
            return user;
        } catch (error) {
            this.logger.error('Failed to create user', error);
            throw error;
        }
    }
    
    async getUser(userId) {
        return await this.database.findById('users', userId);
    }
}

function createUserManager(config) {
    const database = new Database(config.database);
    const logger = new Logger(config.logger);
    return new UserManager(database, logger);
}

export { UserManager, createUserManager };
"""


@pytest.fixture
def config(temp_dir: Path) -> SearchConfig:
    """Create a test configuration."""
    return SearchConfig(
        paths=[str(temp_dir)],
        include=["**/*.py", "**/*.js"],
        enable_graphrag=True,
        enable_metadata_indexing=True,
    )


class TestEntityExtractor:
    """Test entity extraction functionality."""

    def test_init(self) -> None:
        """Test EntityExtractor initialization."""
        extractor = EntityExtractor()
        assert extractor is not None
        assert hasattr(extractor, "language_extractors")

    @pytest.mark.asyncio
    async def test_extract_python_entities(self, temp_dir: Path, sample_python_code: str) -> None:
        """Test Python entity extraction."""
        test_file = temp_dir / "test.py"
        test_file.write_text(sample_python_code)

        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(test_file)

        assert len(entities) > 0
        entity_types = {entity.entity_type for entity in entities}
        assert EntityType.CLASS in entity_types
        assert EntityType.FUNCTION in entity_types
        assert EntityType.IMPORT in entity_types

        class_entities = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(class_entities) >= 1
        assert any(e.name == "DatabaseManager" for e in class_entities)

        function_entities = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(function_entities) >= 3
        function_names = {e.name for e in function_entities}
        assert "connect" in function_names
        assert "execute_query" in function_names
        assert "create_manager" in function_names

    @pytest.mark.asyncio
    async def test_extract_javascript_entities(
        self, temp_dir: Path, sample_javascript_code: str
    ) -> None:
        """Test JavaScript entity extraction."""
        test_file = temp_dir / "test.js"
        test_file.write_text(sample_javascript_code)

        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(test_file)

        assert len(entities) > 0
        entity_types = {entity.entity_type for entity in entities}
        assert EntityType.CLASS in entity_types
        assert EntityType.FUNCTION in entity_types
        assert EntityType.IMPORT in entity_types

        class_entities = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert any(e.name == "UserManager" for e in class_entities)

        function_entities = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        function_names = {e.name for e in function_entities}
        assert "createUserManager" in function_names

    @pytest.mark.asyncio
    async def test_extract_from_directory(
        self, temp_dir: Path, sample_python_code: str, config: SearchConfig
    ) -> None:
        """Test directory-wide entity extraction."""
        (temp_dir / "module1.py").write_text(sample_python_code)
        (temp_dir / "module2.py").write_text("def helper_function(): pass")

        extractor = EntityExtractor()
        entities = await extractor.extract_from_directory(temp_dir, config)

        assert len(entities) > 0
        file_paths = {entity.file_path for entity in entities}
        assert len(file_paths) >= 2


class TestRelationshipMapper:
    """Test relationship mapping functionality."""

    def test_init(self) -> None:
        """Test RelationshipMapper initialization."""
        mapper = RelationshipMapper()
        assert mapper is not None
        assert hasattr(mapper, "dependency_analyzer")

    @pytest.mark.asyncio
    async def test_map_inheritance_relationships(self, temp_dir: Path) -> None:
        """Test inheritance relationship mapping."""
        base_entity = CodeEntity(
            id="base_class",
            name="BaseClass",
            entity_type=EntityType.CLASS,
            file_path=temp_dir / "test.py",
            start_line=1,
            end_line=5,
            properties={"bases": []},
        )

        derived_entity = CodeEntity(
            id="derived_class",
            name="DerivedClass",
            entity_type=EntityType.CLASS,
            file_path=temp_dir / "test.py",
            start_line=7,
            end_line=12,
            properties={"bases": ["BaseClass"]},
        )

        mapper = RelationshipMapper()
        relationships = await mapper._map_inheritance_relationships([base_entity, derived_entity])

        assert len(relationships) == 1
        rel = relationships[0]
        assert rel.relation_type == RelationType.INHERITS
        assert rel.source_entity_id == "derived_class"
        assert rel.target_entity_id == "base_class"

    @pytest.mark.asyncio
    async def test_map_call_relationships(self, temp_dir: Path) -> None:
        """Test function call relationship mapping."""
        caller_entity = CodeEntity(
            id="caller_func",
            name="caller",
            entity_type=EntityType.FUNCTION,
            file_path=temp_dir / "test.py",
            start_line=1,
            end_line=5,
        )

        callee_entity = CodeEntity(
            id="callee_func",
            name="callee",
            entity_type=EntityType.FUNCTION,
            file_path=temp_dir / "test.py",
            start_line=7,
            end_line=10,
        )

        file_contents = {
            temp_dir
            / "test.py": "def caller():\n    result = callee()\n    return result\n\ndef callee():\n    return 42"
        }

        mapper = RelationshipMapper()
        relationships = await mapper._map_call_relationships(
            [caller_entity, callee_entity], file_contents
        )

        assert len(relationships) >= 1
        call_rels = [r for r in relationships if r.relation_type == RelationType.CALLS]
        assert len(call_rels) >= 1


class TestKnowledgeGraph:
    """Test KnowledgeGraph functionality."""

    def test_init(self) -> None:
        """Test KnowledgeGraph initialization."""
        graph = KnowledgeGraph()
        assert graph is not None
        assert len(graph.entities) == 0
        assert len(graph.relationships) == 0

    def test_add_entity(self) -> None:
        """Test adding entities to the graph."""
        graph = KnowledgeGraph()
        entity = CodeEntity(
            id="test_entity",
            name="TestEntity",
            entity_type=EntityType.CLASS,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
        )

        graph.add_entity(entity)

        assert len(graph.entities) == 1
        assert "test_entity" in graph.entities
        assert graph.entities["test_entity"] == entity
        assert "TestEntity" in graph.entity_index
        assert EntityType.CLASS in graph.type_index
        assert Path("test.py") in graph.file_index

    def test_add_relationship(self) -> None:
        """Test adding relationships to the graph."""
        graph = KnowledgeGraph()
        relationship = EntityRelationship(
            id="test_rel",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relation_type=RelationType.CALLS,
        )

        graph.add_relationship(relationship)

        assert len(graph.relationships) == 1
        assert graph.relationships[0] == relationship

    def test_get_related_entities(self) -> None:
        """Test getting related entities."""
        graph = KnowledgeGraph()

        entity1 = CodeEntity(
            id="entity1",
            name="Entity1",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
        )
        entity2 = CodeEntity(
            id="entity2",
            name="Entity2",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=7,
            end_line=10,
        )

        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relationship = EntityRelationship(
            id="test_rel",
            source_entity_id="entity1",
            target_entity_id="entity2",
            relation_type=RelationType.CALLS,
        )
        graph.add_relationship(relationship)

        related = graph.get_related_entities("entity1")

        assert len(related) == 1
        related_entity, rel = related[0]
        assert related_entity.id == "entity2"
        assert rel.relation_type == RelationType.CALLS
