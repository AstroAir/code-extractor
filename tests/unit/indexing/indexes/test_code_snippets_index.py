"""Tests for pysearch.indexing.indexes.code_snippets_index module."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pysearch.analysis.content_addressing import (
    IndexTag,
    PathAndCacheKey,
    RefreshIndexResults,
)
from pysearch.core.types import EntityType
from pysearch.indexing.indexes.code_snippets_index import CodeSnippetsIndex


def _make_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cfg.resolve_cache_dir.return_value = cache_dir
    return cfg


def _make_tag() -> IndexTag:
    return IndexTag(directory="/repo", branch="main", artifact_id="enhanced_code_snippets")


def _make_refresh_results(
    compute=None,
    delete=None,
    add_tag=None,
    remove_tag=None,
) -> RefreshIndexResults:
    return RefreshIndexResults(
        compute=compute or [],
        delete=delete or [],
        add_tag=add_tag or [],
        remove_tag=remove_tag or [],
    )


def _make_entity(
    name: str = "my_func",
    entity_type: EntityType = EntityType.FUNCTION,
    signature: str | None = "def my_func(x: int) -> str:",
    docstring: str | None = "A function.",
    start_line: int = 1,
    end_line: int = 10,
    properties: dict | None = None,
) -> MagicMock:
    entity = MagicMock()
    entity.name = name
    entity.entity_type = entity_type
    entity.signature = signature
    entity.docstring = docstring
    entity.start_line = start_line
    entity.end_line = end_line
    entity.properties = properties or {}
    return entity


async def _seed_snippets(idx: CodeSnippetsIndex, tag: IndexTag, count: int = 3):
    """Seed test snippets into the database."""
    conn = await idx._get_connection()
    tag_string = tag.to_string()
    now = time.time()

    for i in range(count):
        cursor = conn.execute(
            """INSERT INTO code_snippets
            (path, content_hash, name, entity_type, signature, docstring,
             content, language, start_line, end_line, complexity_score,
             quality_score, dependencies, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                f"/test/file{i}.py",
                f"hash_{i}",
                f"entity_{i}",
                "function" if i % 2 == 0 else "class",
                f"def entity_{i}():",
                f"Docstring for entity {i}",
                f"def entity_{i}(): pass",
                "python",
                i * 10 + 1,
                i * 10 + 10,
                0.3 + i * 0.1,
                0.5 + i * 0.15,
                json.dumps(["os", "sys"]),
                json.dumps({"key": f"val_{i}"}),
                now,
            ),
        )
        snippet_id = cursor.lastrowid
        conn.execute(
            "INSERT INTO snippet_tags (snippet_id, tag, created_at) VALUES (?, ?, ?)",
            (snippet_id, tag_string, now),
        )
    conn.commit()


class TestCodeSnippetsIndexInit:
    """Tests for CodeSnippetsIndex initialization."""

    def test_artifact_id(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        assert idx.artifact_id == "enhanced_code_snippets"

    def test_relative_expected_time(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        assert idx.relative_expected_time == 1.5

    def test_init_attributes(self, tmp_path: Path):
        cfg = _make_config(tmp_path)
        idx = CodeSnippetsIndex(config=cfg)
        assert idx.config is cfg
        assert idx.db_path == cfg.resolve_cache_dir() / "code_snippets.db"
        assert idx._connection is None


class TestCodeSnippetsIndexConnection:
    """Tests for database connection and table creation."""

    @pytest.mark.asyncio
    async def test_get_connection_creates_db(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        assert conn is not None
        assert idx._connection is conn

    @pytest.mark.asyncio
    async def test_get_connection_idempotent(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        conn1 = await idx._get_connection()
        conn2 = await idx._get_connection()
        assert conn1 is conn2

    @pytest.mark.asyncio
    async def test_tables_created(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        for table_name in ("code_snippets", "snippet_tags"):
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            assert cursor.fetchone() is not None, f"Table {table_name} not created"

    @pytest.mark.asyncio
    async def test_indexes_created(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        conn = await idx._get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_snippet%'"
        )
        index_names = {row[0] for row in cursor.fetchall()}
        assert "idx_snippets_path_hash" in index_names
        assert "idx_snippets_name_type" in index_names
        assert "idx_snippets_language" in index_names
        assert "idx_snippet_tags_tag" in index_names


class TestCodeSnippetsIndexCalculateEntityQuality:
    """Tests for CodeSnippetsIndex._calculate_entity_quality method."""

    def _make_idx(self, tmp_path: Path) -> CodeSnippetsIndex:
        return CodeSnippetsIndex(config=_make_config(tmp_path))

    def test_quality_long_name_public_with_docs(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        entity = _make_entity(
            name="process_data",
            signature="def process_data(x):",
            docstring="Process data.",
        )
        quality = idx._calculate_entity_quality(entity)
        # name > 3 (+0.2), not private (+0.1), docstring (+0.3), signature (+0.2), 1 line sig (+0.2)
        assert quality == pytest.approx(1.0)

    def test_quality_short_name(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        entity = _make_entity(name="fn", signature=None, docstring=None)
        quality = idx._calculate_entity_quality(entity)
        # name <= 3 (+0), not private (+0.1), no docstring, no signature
        assert quality == pytest.approx(0.1)

    def test_quality_private_name(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        entity = _make_entity(name="_private_func", signature=None, docstring=None)
        quality = idx._calculate_entity_quality(entity)
        # name > 3 (+0.2), private (+0), no docstring, no signature
        assert quality == pytest.approx(0.2)

    def test_quality_with_docstring_only(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        entity = _make_entity(name="func", signature=None, docstring="Has docstring.")
        quality = idx._calculate_entity_quality(entity)
        # name > 3 (+0.2), not private (+0.1), docstring (+0.3), no signature
        assert quality == pytest.approx(0.6)

    def test_quality_capped_at_one(self, tmp_path: Path):
        idx = self._make_idx(tmp_path)
        entity = _make_entity(
            name="very_descriptive_function_name",
            signature="def very_descriptive_function_name(a, b, c):",
            docstring="Very well documented function.",
        )
        quality = idx._calculate_entity_quality(entity)
        assert quality <= 1.0


class TestCodeSnippetsIndexUpdate:
    """Tests for CodeSnippetsIndex.update method."""

    @pytest.mark.asyncio
    async def test_update_compute(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()

        item = PathAndCacheKey(path="/test/file.py", cache_key="abc123")
        results = _make_refresh_results(compute=[item])

        mock_entity = _make_entity()
        mock_processor = MagicMock()
        mock_processor.extract_entities.return_value = [mock_entity]
        mock_processor.analyze_dependencies.return_value = ["os"]
        mock_processor.calculate_complexity.return_value = 0.5

        with (
            patch(
                "pysearch.indexing.indexes.code_snippets_index.read_text_safely",
                return_value="def my_func(x: int) -> str:\n    pass\n",
            ),
            patch(
                "pysearch.indexing.indexes.code_snippets_index.language_registry",
            ) as mock_registry,
        ):
            mock_registry.get_processor.return_value = mock_processor

            updates = []
            async for update in idx.update(tag, results, mark_complete):
                updates.append(update)

        mark_complete.assert_called_once_with([item], "compute")
        assert any(u.status == "done" for u in updates)

        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM code_snippets")
        assert cursor.fetchone()[0] == 1

    @pytest.mark.asyncio
    async def test_update_compute_skips_empty(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/empty.py", cache_key="empty")
        results = _make_refresh_results(compute=[item])

        with patch(
            "pysearch.indexing.indexes.code_snippets_index.read_text_safely",
            return_value=None,
        ):
            async for _ in idx.update(tag, results, mark_complete):
                pass

        mark_complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_add_tag(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        new_tag = IndexTag(directory="/repo", branch="dev", artifact_id="enhanced_code_snippets")
        results = _make_refresh_results(add_tag=[item])

        async for _ in idx.update(new_tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "add_tag")

    @pytest.mark.asyncio
    async def test_update_remove_tag(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        results = _make_refresh_results(remove_tag=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "remove_tag")
        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM snippet_tags WHERE tag = ?", (tag.to_string(),))
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_delete(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        mark_complete = MagicMock()
        item = PathAndCacheKey(path="/test/file0.py", cache_key="hash_0")
        results = _make_refresh_results(delete=[item])

        async for _ in idx.update(tag, results, mark_complete):
            pass

        mark_complete.assert_called_once_with([item], "delete")
        conn = await idx._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM code_snippets")
        assert cursor.fetchone()[0] == 0

    @pytest.mark.asyncio
    async def test_update_empty_results(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        mark_complete = MagicMock()
        results = _make_refresh_results()

        updates = []
        async for update in idx.update(tag, results, mark_complete):
            updates.append(update)

        mark_complete.assert_not_called()
        assert updates[-1].status == "done"


class TestCodeSnippetsIndexRetrieve:
    """Tests for CodeSnippetsIndex.retrieve method."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("entity", tag)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_specific_term(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("entity_0", tag)
        assert len(results) >= 1
        assert any(r["name"] == "entity_0" for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_entity_type_filter(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("entity", tag, entity_type="function")
        assert all(r["entity_type"] == "function" for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_language_filter(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("entity", tag, language="python")
        assert len(results) == 3
        results = await idx.retrieve("entity", tag, language="rust")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_retrieve_with_min_quality(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("entity", tag, min_quality=0.7)
        assert all(r["quality_score"] >= 0.7 for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_with_limit(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=10)

        results = await idx.retrieve("entity", tag, limit=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_retrieve_result_structure(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        results = await idx.retrieve("entity", tag)
        assert len(results) == 1
        r = results[0]
        expected_keys = {
            "id",
            "path",
            "content_hash",
            "name",
            "entity_type",
            "signature",
            "docstring",
            "content",
            "language",
            "start_line",
            "end_line",
            "complexity_score",
            "quality_score",
            "dependencies",
            "metadata",
        }
        assert set(r.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_retrieve_no_match(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.retrieve("zzz_nonexistent_xyzzy", tag)
        assert len(results) == 0


class TestCodeSnippetsIndexGetEntityById:
    """Tests for CodeSnippetsIndex.get_entity_by_id method."""

    @pytest.mark.asyncio
    async def test_get_existing_entity(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        result = await idx.get_entity_by_id(1)
        assert result is not None
        assert result["name"] == "entity_0"
        assert "id" in result

    @pytest.mark.asyncio
    async def test_get_nonexistent_entity(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        await idx._get_connection()

        result = await idx.get_entity_by_id(99999)
        assert result is None


class TestCodeSnippetsIndexGetEntitiesByFile:
    """Tests for CodeSnippetsIndex.get_entities_by_file method."""

    @pytest.mark.asyncio
    async def test_get_entities_by_file(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=3)

        results = await idx.get_entities_by_file("/test/file0.py", tag)
        assert len(results) == 1
        assert results[0]["name"] == "entity_0"

    @pytest.mark.asyncio
    async def test_get_entities_by_file_empty(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()

        results = await idx.get_entities_by_file("/nonexistent.py", tag)
        assert results == []

    @pytest.mark.asyncio
    async def test_get_entities_by_file_result_structure(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        results = await idx.get_entities_by_file("/test/file0.py", tag)
        assert len(results) == 1
        expected_keys = {
            "id",
            "name",
            "entity_type",
            "signature",
            "start_line",
            "end_line",
            "complexity_score",
            "quality_score",
        }
        assert set(results[0].keys()) == expected_keys


class TestCodeSnippetsIndexGetStatistics:
    """Tests for CodeSnippetsIndex.get_statistics method."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await idx._get_connection()

        stats = await idx.get_statistics(tag)
        assert stats["total_entities"] == 0
        assert stats["entities_by_type"] == {}
        assert stats["entities_by_language"] == {}
        assert stats["average_quality"] == 0.0
        assert stats["average_complexity"] == 0.0

    @pytest.mark.asyncio
    async def test_get_statistics_with_data(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=3)

        stats = await idx.get_statistics(tag)
        assert stats["total_entities"] == 3
        assert "function" in stats["entities_by_type"]
        assert "python" in stats["entities_by_language"]
        assert stats["average_quality"] > 0
        assert stats["average_complexity"] > 0


class TestCodeSnippetsIndexSearchEntities:
    """Tests for CodeSnippetsIndex.search_entities method."""

    @pytest.mark.asyncio
    async def test_search_basic(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.search_entities("entity", tag)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_with_entity_types(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.search_entities("entity", tag, entity_types=["function"])
        assert all(r["entity_type"] == "function" for r in results)

    @pytest.mark.asyncio
    async def test_search_with_languages(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.search_entities("entity", tag, languages=["python"])
        assert len(results) == 3
        results = await idx.search_entities("entity", tag, languages=["java"])
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_with_min_quality(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.search_entities("entity", tag, min_quality=0.7)
        assert all(r["quality_score"] >= 0.7 for r in results)

    @pytest.mark.asyncio
    async def test_search_with_limit(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=10)

        results = await idx.search_entities("entity", tag, limit=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_empty_query(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag)

        results = await idx.search_entities("", tag)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_search_combined_filters(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=5)

        results = await idx.search_entities(
            "entity",
            tag,
            entity_types=["function"],
            languages=["python"],
            min_quality=0.5,
            limit=10,
        )
        assert all(r["entity_type"] == "function" for r in results)
        assert all(r["language"] == "python" for r in results)
        assert all(r["quality_score"] >= 0.5 for r in results)

    @pytest.mark.asyncio
    async def test_search_result_structure(self, tmp_path: Path):
        idx = CodeSnippetsIndex(config=_make_config(tmp_path))
        tag = _make_tag()
        await _seed_snippets(idx, tag, count=1)

        results = await idx.search_entities("entity", tag)
        assert len(results) == 1
        r = results[0]
        expected_keys = {
            "id",
            "path",
            "name",
            "entity_type",
            "signature",
            "docstring",
            "content",
            "language",
            "start_line",
            "end_line",
            "complexity_score",
            "quality_score",
            "dependencies",
            "metadata",
        }
        assert set(r.keys()) == expected_keys
