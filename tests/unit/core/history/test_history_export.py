"""Tests for pysearch.core.history.history_export module."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pysearch.core.config import SearchConfig
from pysearch.core.history.history_core import SearchCategory, SearchHistoryEntry
from pysearch.core.history.history_export import ExportFormat, HistoryExporter


def _make_entry(
    pattern: str = "test",
    category: SearchCategory = SearchCategory.GENERAL,
    items_count: int = 5,
    elapsed_ms: float = 50.0,
    success_score: float = 0.7,
    languages: set[str] | None = None,
    tags: set[str] | None = None,
    timestamp: float | None = None,
) -> SearchHistoryEntry:
    return SearchHistoryEntry(
        timestamp=timestamp or time.time(),
        query_pattern=pattern,
        use_regex=False,
        use_ast=False,
        context=2,
        files_matched=2,
        items_count=items_count,
        elapsed_ms=elapsed_ms,
        category=category,
        success_score=success_score,
        languages=languages,
        tags=tags,
    )


class TestExportFormat:
    """Tests for ExportFormat enum."""

    def test_values(self):
        assert ExportFormat.JSON == "json"
        assert ExportFormat.CSV == "csv"


class TestHistoryExporterExport:
    """Tests for export functionality."""

    def test_export_json_file(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        entries = [_make_entry("foo"), _make_entry("bar")]
        out = tmp_path / "export.json"

        count = exporter.export_history(entries, out, ExportFormat.JSON)

        assert count == 2
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_entries"] == 2
        assert len(data["entries"]) == 2
        assert data["entries"][0]["query_pattern"] == "foo"

    def test_export_csv_file(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        entries = [_make_entry("foo", languages={"python"}, tags={"tag1"})]
        out = tmp_path / "export.csv"

        count = exporter.export_history(entries, out, ExportFormat.CSV)

        assert count == 1
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "query_pattern" in content  # header
        assert "foo" in content

    def test_export_with_date_filter(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        now = time.time()
        entries = [
            _make_entry("old", timestamp=now - 100000),
            _make_entry("recent", timestamp=now),
        ]
        out = tmp_path / "filtered.json"

        count = exporter.export_history(
            entries,
            out,
            ExportFormat.JSON,
            start_time=now - 1000,
        )

        assert count == 1
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["entries"][0]["query_pattern"] == "recent"

    def test_export_with_end_time_filter(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        now = time.time()
        entries = [
            _make_entry("old", timestamp=now - 100000),
            _make_entry("recent", timestamp=now),
        ]
        out = tmp_path / "filtered.json"

        count = exporter.export_history(
            entries,
            out,
            ExportFormat.JSON,
            end_time=now - 50000,
        )

        assert count == 1
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["entries"][0]["query_pattern"] == "old"

    def test_export_to_string_json(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        entries = [_make_entry("hello")]

        result = exporter.export_history_to_string(entries, ExportFormat.JSON)

        data = json.loads(result)
        assert data["total_entries"] == 1
        assert data["entries"][0]["query_pattern"] == "hello"

    def test_export_to_string_csv(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        entries = [_make_entry("hello")]

        result = exporter.export_history_to_string(entries, ExportFormat.CSV)

        assert "query_pattern" in result
        assert "hello" in result

    def test_export_empty_list(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        out = tmp_path / "empty.json"

        count = exporter.export_history([], out, ExportFormat.JSON)

        assert count == 0
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_entries"] == 0

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        out = tmp_path / "sub" / "dir" / "export.json"

        exporter.export_history([_make_entry()], out, ExportFormat.JSON)

        assert out.exists()


class TestHistoryExporterImport:
    """Tests for import functionality."""

    def test_import_json(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        # First export
        entries = [_make_entry("foo"), _make_entry("bar")]
        out = tmp_path / "export.json"
        exporter.export_history(entries, out, ExportFormat.JSON)

        # Then import
        imported = exporter.import_history(out)

        assert len(imported) == 2
        assert imported[0].query_pattern == "foo"

    def test_import_merge(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        existing = [_make_entry("existing", timestamp=1000.0)]
        new_entries = [_make_entry("new", timestamp=2000.0)]
        out = tmp_path / "new.json"
        exporter.export_history(new_entries, out, ExportFormat.JSON)

        merged = exporter.import_history(out, merge=True, existing_entries=existing)

        assert len(merged) == 2
        patterns = {e.query_pattern for e in merged}
        assert "existing" in patterns
        assert "new" in patterns

    def test_import_merge_deduplicates(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        ts = 1000.0
        existing = [_make_entry("same", timestamp=ts)]
        same_entries = [_make_entry("same", timestamp=ts)]
        out = tmp_path / "same.json"
        exporter.export_history(same_entries, out, ExportFormat.JSON)

        merged = exporter.import_history(out, merge=True, existing_entries=existing)

        assert len(merged) == 1

    def test_import_replace(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        new_entries = [_make_entry("only_new")]
        out = tmp_path / "replace.json"
        exporter.export_history(new_entries, out, ExportFormat.JSON)

        replaced = exporter.import_history(out, merge=False)

        assert len(replaced) == 1
        assert replaced[0].query_pattern == "only_new"

    def test_import_from_string(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        entries = [_make_entry("from_string")]
        json_str = exporter.export_history_to_string(entries, ExportFormat.JSON)

        imported = exporter.import_history_from_string(json_str)

        assert len(imported) == 1
        assert imported[0].query_pattern == "from_string"

    def test_import_file_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        with pytest.raises(FileNotFoundError):
            exporter.import_history(tmp_path / "nonexistent.json")

    def test_import_invalid_format(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        bad_file = tmp_path / "bad.json"
        bad_file.write_text('{"no_entries": true}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'entries'"):
            exporter.import_history(bad_file)


class TestHistoryExporterBackupRestore:
    """Tests for backup and restore functionality."""

    def test_backup_creates_file(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)
        backup_file = tmp_path / "backup.json"

        counts = exporter.backup(backup_file)

        assert backup_file.exists()
        assert "history_entries" in counts

    def test_backup_includes_all_data(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=cache_dir)

        # Create some history data
        history_data = {
            "version": 1,
            "entries": [
                {
                    "timestamp": 1.0,
                    "query_pattern": "test",
                    "use_regex": False,
                    "use_ast": False,
                    "context": 0,
                    "files_matched": 0,
                    "items_count": 0,
                    "elapsed_ms": 0.0,
                    "category": "general",
                    "success_score": 0.0,
                }
            ],
        }
        (cache_dir / "search_history.json").write_text(json.dumps(history_data), encoding="utf-8")

        bookmarks_data = {
            "version": 1,
            "bookmarks": {
                "bm1": {
                    "timestamp": 1.0,
                    "query_pattern": "bm",
                    "use_regex": False,
                    "use_ast": False,
                    "context": 0,
                    "files_matched": 0,
                    "items_count": 0,
                    "elapsed_ms": 0.0,
                    "category": "general",
                    "success_score": 0.0,
                }
            },
        }
        (cache_dir / "bookmarks.json").write_text(json.dumps(bookmarks_data), encoding="utf-8")

        exporter = HistoryExporter(cfg)
        backup_file = tmp_path / "full_backup.json"
        counts = exporter.backup(backup_file)

        assert counts["history_entries"] == 1
        assert counts["bookmarks"] == 1

        data = json.loads(backup_file.read_text(encoding="utf-8"))
        assert "history" in data
        assert "bookmarks" in data
        assert "sessions" in data
        assert "bookmark_folders" in data

    def test_restore(self, tmp_path: Path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=cache_dir)

        # Create a backup
        backup_data = {
            "version": 1,
            "backup_time": time.time(),
            "history": {
                "version": 1,
                "entries": [
                    {
                        "timestamp": 1.0,
                        "query_pattern": "restored",
                        "use_regex": False,
                        "use_ast": False,
                        "context": 0,
                        "files_matched": 1,
                        "items_count": 2,
                        "elapsed_ms": 10.0,
                        "category": "general",
                        "success_score": 0.5,
                    }
                ],
            },
            "bookmarks": {"version": 1, "bookmarks": {}},
            "bookmark_folders": {"version": 1, "folders": {}},
            "sessions": {"version": 1, "sessions": {}},
        }
        backup_file = tmp_path / "restore_test.json"
        backup_file.write_text(json.dumps(backup_data), encoding="utf-8")

        exporter = HistoryExporter(cfg)
        counts = exporter.restore(backup_file)

        assert counts["history_entries"] == 1
        # Verify the history file was restored
        history_file = cache_dir / "search_history.json"
        assert history_file.exists()
        restored = json.loads(history_file.read_text(encoding="utf-8"))
        assert restored["entries"][0]["query_pattern"] == "restored"

    def test_restore_file_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        with pytest.raises(FileNotFoundError):
            exporter.restore(tmp_path / "nonexistent.json")

    def test_restore_invalid_format(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        bad_file = tmp_path / "bad_backup.json"
        bad_file.write_text('{"no_version": true}', encoding="utf-8")

        with pytest.raises(ValueError, match="missing 'version'"):
            exporter.restore(bad_file)

    def test_validate_backup_valid(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        backup_data = {
            "version": 1,
            "backup_time_iso": "2025-01-01T00:00:00",
            "history": {"entries": [{"a": 1}]},
            "bookmarks": {"bookmarks": {"b1": {}}},
            "sessions": {"sessions": {"s1": {}}},
        }
        backup_file = tmp_path / "valid.json"
        backup_file.write_text(json.dumps(backup_data), encoding="utf-8")

        result = exporter.validate_backup(backup_file)

        assert result["valid"] is True
        assert result["version"] == 1
        assert result["history_entries"] == 1
        assert result["bookmarks"] == 1
        assert result["sessions"] == 1

    def test_validate_backup_file_not_found(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        result = exporter.validate_backup(tmp_path / "missing.json")

        assert result["valid"] is False
        assert "not found" in result["error"].lower()

    def test_validate_backup_invalid_json(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        bad = tmp_path / "bad.json"
        bad.write_text("not json", encoding="utf-8")

        result = exporter.validate_backup(bad)

        assert result["valid"] is False
        assert "Invalid JSON" in result["error"]

    def test_validate_backup_missing_version(self, tmp_path: Path):
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=tmp_path / "cache")
        exporter = HistoryExporter(cfg)

        bad = tmp_path / "no_version.json"
        bad.write_text('{"data": 1}', encoding="utf-8")

        result = exporter.validate_backup(bad)

        assert result["valid"] is False

    def test_round_trip_backup_restore(self, tmp_path: Path):
        """Test full backup -> restore round trip."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cfg = SearchConfig(paths=[str(tmp_path)], cache_dir=cache_dir)

        # Create initial data
        history_data = {
            "version": 1,
            "entries": [
                {
                    "timestamp": 1.0,
                    "query_pattern": "roundtrip",
                    "use_regex": False,
                    "use_ast": False,
                    "context": 0,
                    "files_matched": 1,
                    "items_count": 3,
                    "elapsed_ms": 25.0,
                    "category": "general",
                    "success_score": 0.7,
                }
            ],
        }
        (cache_dir / "search_history.json").write_text(json.dumps(history_data), encoding="utf-8")

        exporter = HistoryExporter(cfg)

        # Backup
        backup_file = tmp_path / "roundtrip.json"
        backup_counts = exporter.backup(backup_file)

        # Clear cache
        (cache_dir / "search_history.json").unlink()

        # Restore
        restore_counts = exporter.restore(backup_file)

        assert restore_counts["history_entries"] == backup_counts["history_entries"]
        assert (cache_dir / "search_history.json").exists()
