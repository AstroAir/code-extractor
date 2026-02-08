"""Tests for pysearch.core.types.graphrag_types module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.types.graphrag_types import (
    CodeEntity,
    EntityRelationship,
    EntityType,
    GraphRAGQuery,
    GraphRAGResult,
    KnowledgeGraph,
    RelationType,
)
from pysearch.core.types import Language


class TestEntityType:
    """Tests for EntityType enum."""

    def test_common_values(self):
        assert EntityType.FUNCTION == "function"
        assert EntityType.CLASS == "class"
        assert EntityType.METHOD == "method"
        assert EntityType.MODULE == "module"
        assert EntityType.IMPORT == "import"

    def test_is_string_enum(self):
        assert isinstance(EntityType.FUNCTION, str)

    def test_unknown_entity(self):
        assert EntityType.UNKNOWN_ENTITY == "unknown_entity"


class TestRelationType:
    """Tests for RelationType enum."""

    def test_structural_relations(self):
        assert RelationType.CONTAINS == "contains"
        assert RelationType.INHERITS == "inherits"
        assert RelationType.IMPLEMENTS == "implements"

    def test_usage_relations(self):
        assert RelationType.CALLS == "calls"
        assert RelationType.USES == "uses"
        assert RelationType.IMPORTS == "imports"

    def test_unknown_relation(self):
        assert RelationType.UNKNOWN_RELATION == "unknown_relation"


class TestCodeEntity:
    """Tests for CodeEntity dataclass."""

    def test_required_fields(self):
        e = CodeEntity(
            id="func_1",
            name="main",
            entity_type=EntityType.FUNCTION,
            file_path=Path("app.py"),
            start_line=1,
            end_line=10,
        )
        assert e.id == "func_1"
        assert e.name == "main"
        assert e.entity_type == EntityType.FUNCTION
        assert e.file_path == Path("app.py")

    def test_defaults(self):
        e = CodeEntity(
            id="x", name="x", entity_type=EntityType.CLASS,
            file_path=Path("a.py"), start_line=1, end_line=1,
        )
        assert e.signature is None
        assert e.docstring is None
        assert e.properties == {}
        assert e.embedding is None
        assert e.confidence == 1.0
        assert e.language == Language.UNKNOWN
        assert e.scope is None
        assert e.access_modifier is None

    def test_custom_fields(self):
        e = CodeEntity(
            id="func_2",
            name="helper",
            entity_type=EntityType.FUNCTION,
            file_path=Path("utils.py"),
            start_line=5,
            end_line=15,
            signature="def helper(x: int) -> str:",
            docstring="A helper function.",
            confidence=0.9,
            language=Language.PYTHON,
            scope="module",
            access_modifier="public",
        )
        assert e.signature == "def helper(x: int) -> str:"
        assert e.docstring == "A helper function."
        assert e.confidence == 0.9
        assert e.language == Language.PYTHON


class TestEntityRelationship:
    """Tests for EntityRelationship dataclass."""

    def test_required_fields(self):
        r = EntityRelationship(
            id="rel_1",
            source_entity_id="func_1",
            target_entity_id="func_2",
            relation_type=RelationType.CALLS,
        )
        assert r.id == "rel_1"
        assert r.source_entity_id == "func_1"
        assert r.target_entity_id == "func_2"
        assert r.relation_type == RelationType.CALLS

    def test_defaults(self):
        r = EntityRelationship(
            id="r", source_entity_id="a", target_entity_id="b",
            relation_type=RelationType.USES,
        )
        assert r.properties == {}
        assert r.confidence == 1.0
        assert r.weight == 1.0
        assert r.context is None
        assert r.file_path is None
        assert r.line_number is None


class TestKnowledgeGraph:
    """Tests for KnowledgeGraph dataclass."""

    def _make_entity(self, id: str, name: str, etype: EntityType, fpath: str = "a.py") -> CodeEntity:
        return CodeEntity(
            id=id, name=name, entity_type=etype,
            file_path=Path(fpath), start_line=1, end_line=1,
        )

    def _make_relationship(self, id: str, src: str, tgt: str, rtype: RelationType) -> EntityRelationship:
        return EntityRelationship(id=id, source_entity_id=src, target_entity_id=tgt, relation_type=rtype)

    def test_empty_graph(self):
        g = KnowledgeGraph()
        assert g.entities == {}
        assert g.relationships == []
        assert g.version == "1.0"

    def test_add_entity(self):
        g = KnowledgeGraph()
        e = self._make_entity("f1", "main", EntityType.FUNCTION)
        g.add_entity(e)
        assert "f1" in g.entities
        assert "main" in g.entity_index
        assert EntityType.FUNCTION in g.type_index
        assert Path("a.py") in g.file_index

    def test_add_relationship(self):
        g = KnowledgeGraph()
        r = self._make_relationship("r1", "f1", "f2", RelationType.CALLS)
        g.add_relationship(r)
        assert len(g.relationships) == 1

    def test_get_entity(self):
        g = KnowledgeGraph()
        e = self._make_entity("f1", "main", EntityType.FUNCTION)
        g.add_entity(e)
        assert g.get_entity("f1") is e
        assert g.get_entity("nonexistent") is None

    def test_get_entities_by_name(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "main", EntityType.FUNCTION)
        e2 = self._make_entity("f2", "main", EntityType.METHOD)
        g.add_entity(e1)
        g.add_entity(e2)
        result = g.get_entities_by_name("main")
        assert len(result) == 2

    def test_get_entities_by_type(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "a", EntityType.FUNCTION)
        e2 = self._make_entity("c1", "B", EntityType.CLASS)
        g.add_entity(e1)
        g.add_entity(e2)
        funcs = g.get_entities_by_type(EntityType.FUNCTION)
        assert len(funcs) == 1
        assert funcs[0].name == "a"

    def test_get_entities_in_file(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "a", EntityType.FUNCTION, "x.py")
        e2 = self._make_entity("f2", "b", EntityType.FUNCTION, "y.py")
        g.add_entity(e1)
        g.add_entity(e2)
        result = g.get_entities_in_file(Path("x.py"))
        assert len(result) == 1
        assert result[0].name == "a"

    def test_get_related_entities_source(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "main", EntityType.FUNCTION)
        e2 = self._make_entity("f2", "helper", EntityType.FUNCTION)
        g.add_entity(e1)
        g.add_entity(e2)
        r = self._make_relationship("r1", "f1", "f2", RelationType.CALLS)
        g.add_relationship(r)
        related = g.get_related_entities("f1")
        assert len(related) == 1
        assert related[0][0].name == "helper"
        assert related[0][1].relation_type == RelationType.CALLS

    def test_get_related_entities_target(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "main", EntityType.FUNCTION)
        e2 = self._make_entity("f2", "helper", EntityType.FUNCTION)
        g.add_entity(e1)
        g.add_entity(e2)
        r = self._make_relationship("r1", "f1", "f2", RelationType.CALLS)
        g.add_relationship(r)
        related = g.get_related_entities("f2")
        assert len(related) == 1
        assert related[0][0].name == "main"

    def test_get_related_entities_with_filter(self):
        g = KnowledgeGraph()
        e1 = self._make_entity("f1", "a", EntityType.FUNCTION)
        e2 = self._make_entity("f2", "b", EntityType.FUNCTION)
        e3 = self._make_entity("f3", "c", EntityType.FUNCTION)
        g.add_entity(e1)
        g.add_entity(e2)
        g.add_entity(e3)
        g.add_relationship(self._make_relationship("r1", "f1", "f2", RelationType.CALLS))
        g.add_relationship(self._make_relationship("r2", "f1", "f3", RelationType.USES))
        related = g.get_related_entities("f1", relation_types=[RelationType.CALLS])
        assert len(related) == 1
        assert related[0][0].name == "b"


class TestGraphRAGQuery:
    """Tests for GraphRAGQuery dataclass."""

    def test_minimal(self):
        q = GraphRAGQuery(pattern="database")
        assert q.pattern == "database"
        assert q.entity_types is None
        assert q.relation_types is None
        assert q.include_relationships is True
        assert q.max_hops == 2

    def test_defaults(self):
        q = GraphRAGQuery(pattern="x")
        assert q.min_confidence == 0.5
        assert q.semantic_threshold == 0.7
        assert q.use_vector_search is True
        assert q.context_window == 5
        assert q.ranking_strategy == "relevance"

    def test_custom_values(self):
        q = GraphRAGQuery(
            pattern="web api",
            entity_types=[EntityType.FUNCTION, EntityType.CLASS],
            relation_types=[RelationType.CALLS],
            max_hops=3,
            min_confidence=0.8,
        )
        assert len(q.entity_types) == 2
        assert q.max_hops == 3
        assert q.min_confidence == 0.8


class TestGraphRAGResult:
    """Tests for GraphRAGResult dataclass."""

    def test_defaults(self):
        r = GraphRAGResult()
        assert r.entities == []
        assert r.relationships == []
        assert r.context_entities == []
        assert r.similarity_scores == {}
        assert r.graph_paths == []
        assert r.metadata == {}
        assert r.query is None

    def test_with_data(self):
        e = CodeEntity(
            id="f1", name="main", entity_type=EntityType.FUNCTION,
            file_path=Path("a.py"), start_line=1, end_line=1,
        )
        q = GraphRAGQuery(pattern="test")
        r = GraphRAGResult(
            entities=[e],
            similarity_scores={"f1": 0.95},
            query=q,
        )
        assert len(r.entities) == 1
        assert r.similarity_scores["f1"] == 0.95
        assert r.query.pattern == "test"
