"""Tests for pysearch.analysis.graphrag.engine module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysearch.analysis.graphrag.engine import (
    GraphRAGEngine,
    KnowledgeGraphBuilder,
)
from pysearch.core.config import SearchConfig
from pysearch.core.types import (
    CodeEntity,
    EntityRelationship,
    EntityType,
    GraphRAGQuery,
    KnowledgeGraph,
    RelationType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_entity(
    name: str,
    entity_type: EntityType = EntityType.FUNCTION,
    entity_id: str | None = None,
    start_line: int = 1,
    end_line: int = 3,
    file_path: Path | None = None,
) -> CodeEntity:
    return CodeEntity(
        id=entity_id or f"{entity_type.value}_{name}_{start_line}",
        name=name,
        entity_type=entity_type,
        file_path=file_path or Path("test.py"),
        start_line=start_line,
        end_line=end_line,
        properties={},
    )


def _make_relationship(
    src_id: str,
    tgt_id: str,
    rel_type: RelationType = RelationType.CALLS,
    rel_id: str | None = None,
) -> EntityRelationship:
    return EntityRelationship(
        id=rel_id or f"rel_{src_id}_{tgt_id}",
        source_entity_id=src_id,
        target_entity_id=tgt_id,
        relation_type=rel_type,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# KnowledgeGraphBuilder
# ---------------------------------------------------------------------------
class TestKnowledgeGraphBuilder:
    """Tests for KnowledgeGraphBuilder class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        assert builder.config is cfg
        assert builder.entity_extractor is not None
        assert builder.relationship_mapper is not None
        assert builder.knowledge_graph is not None

    async def test_build_graph_empty_dir(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        graph = await builder.build_graph()
        assert graph is not None
        assert isinstance(graph, KnowledgeGraph)

    async def test_build_graph_with_files(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("def foo():\n    pass\n", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        graph = await builder.build_graph(force_rebuild=True)
        assert len(graph.entities) >= 1

    async def test_cache_and_load(self, tmp_path: Path):
        (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)])
        builder = KnowledgeGraphBuilder(cfg)
        await builder.build_graph(force_rebuild=True)
        # Cache should have been written
        assert builder._cache_path.exists()
        # Load from cache
        builder2 = KnowledgeGraphBuilder(cfg)
        loaded = await builder2._load_cached_graph()
        assert loaded is True


# ---------------------------------------------------------------------------
# GraphRAGEngine
# ---------------------------------------------------------------------------
class TestGraphRAGEngine:
    """Tests for GraphRAGEngine class."""

    def test_init(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        assert engine.config is cfg
        assert engine._initialized is False
        assert engine.vector_store is None

    async def test_initialize(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        assert engine._initialized is True

    async def test_close(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        await engine.close()

    def test_get_stats_empty(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        stats = engine.get_stats()
        assert isinstance(stats, dict)
        assert stats["initialized"] is False
        assert stats["total_entities"] == 0
        assert stats["total_relationships"] == 0

    async def test_get_stats_async(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        stats = await engine.get_stats_async()
        assert isinstance(stats, dict)
        assert stats["initialized"] is True

    async def test_add_entities(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        entity = _make_entity("foo")
        await engine.add_entities([entity])
        kg = engine.graph_builder.knowledge_graph
        assert entity.id in kg.entities

    async def test_add_relationships(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        e1 = _make_entity("a", entity_id="e1")
        e2 = _make_entity("b", entity_id="e2")
        await engine.add_entities([e1, e2])
        rel = _make_relationship("e1", "e2")
        await engine.add_relationships([rel])
        kg = engine.graph_builder.knowledge_graph
        assert len(kg.relationships) == 1

    async def test_delete_entity(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        entity = _make_entity("foo", entity_id="del_me")
        await engine.add_entities([entity])
        assert "del_me" in engine.graph_builder.knowledge_graph.entities
        await engine.delete_entity("del_me")
        assert "del_me" not in engine.graph_builder.knowledge_graph.entities

    async def test_delete_entity_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        with pytest.raises(ValueError, match="Entity not found"):
            await engine.delete_entity("nonexistent")

    async def test_update_entity(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        entity = _make_entity("foo", entity_id="upd_me")
        await engine.add_entities([entity])
        await engine.update_entity("upd_me", {"name": "bar", "custom_prop": "val"})
        updated = engine.graph_builder.knowledge_graph.get_entity("upd_me")
        assert updated.name == "bar"
        assert updated.properties.get("custom_prop") == "val"

    async def test_get_entity_context(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        entity = _make_entity("foo", entity_id="ctx_e")
        await engine.add_entities([entity])
        context = await engine.get_entity_context("ctx_e")
        assert context["entity"]["name"] == "foo"
        assert "relationships" in context
        assert "related_entities" in context

    async def test_get_entity_context_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        context = await engine.get_entity_context("missing")
        assert context == {}

    async def test_export_graph_json(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        e1 = _make_entity("a", entity_id="exp1")
        await engine.add_entities([e1])
        exported = await engine.export_graph(format="json")
        data = json.loads(exported)
        assert "entities" in data
        assert len(data["entities"]) == 1

    async def test_export_graph_unsupported_format(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        with pytest.raises(ValueError, match="Unsupported export format"):
            await engine.export_graph(format="xml")

    async def test_import_graph_json(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        # Export first
        e1 = _make_entity("a", entity_id="imp1")
        await engine.add_entities([e1])
        exported = await engine.export_graph(format="json")
        # Reset and import
        await engine.reset_graph()
        assert len(engine.graph_builder.knowledge_graph.entities) == 0
        await engine.import_graph(exported, format="json")
        assert len(engine.graph_builder.knowledge_graph.entities) == 1

    async def test_import_graph_unsupported_format(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        with pytest.raises(ValueError, match="Unsupported import format"):
            await engine.import_graph("{}", format="xml")

    async def test_reset_graph(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        await engine.add_entities([_make_entity("x", entity_id="rst1")])
        assert len(engine.graph_builder.knowledge_graph.entities) == 1
        await engine.reset_graph()
        assert len(engine.graph_builder.knowledge_graph.entities) == 0

    async def test_query_graph_text_search(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        e1 = _make_entity("calculate_total", entity_id="q1")
        e2 = _make_entity("print_report", entity_id="q2")
        await engine.add_entities([e1, e2])
        query = GraphRAGQuery(pattern="calculate")
        result = await engine.query_graph(query)
        entity_names = [e.name for e in result.entities]
        assert "calculate_total" in entity_names

    async def test_build_knowledge_graph(self, tmp_path: Path):
        (tmp_path / "sample.py").write_text("def greet():\n    pass\n", encoding="utf-8")
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        kg = await engine.build_knowledge_graph(force_rebuild=True)
        assert isinstance(kg, KnowledgeGraph)
        assert len(kg.entities) >= 1

    async def test_find_similar_entities_no_vector_store(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        await engine.add_entities([_make_entity("a", entity_id="sim1")])
        result = await engine.find_similar_entities("sim1")
        assert isinstance(result, list)

    async def test_batch_find_similar_no_vector_store(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)])
        engine = GraphRAGEngine(cfg)
        await engine.initialize()
        e1 = _make_entity("a", entity_id="batch1")
        e2 = _make_entity("b", entity_id="batch2")
        await engine.add_entities([e1, e2])
        results = await engine.batch_find_similar(["batch1", "batch2"])
        assert isinstance(results, dict)
        assert "batch1" in results
        assert "batch2" in results
