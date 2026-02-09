"""
History export, import, backup, and restore functionality.

This module provides comprehensive data portability for search history,
including export to JSON/CSV formats, import from JSON, and full
backup/restore of the entire history state.

Classes:
    ExportFormat: Supported export formats
    HistoryExporter: Main export/import/backup/restore class

Key Features:
    - Export history to JSON or CSV with optional date range filter
    - Import history from JSON (merge or replace)
    - Full backup of history + bookmarks + sessions to a single JSON archive
    - Restore from backup archive
    - Format validation on import

Example:
    Export/Import usage:
        >>> from pysearch.core.history.history_export import HistoryExporter
        >>> from pysearch.core.config import SearchConfig
        >>>
        >>> config = SearchConfig()
        >>> exporter = HistoryExporter(config)
        >>>
        >>> # Export to JSON
        >>> exporter.export_history(entries, "history.json", ExportFormat.JSON)
        >>>
        >>> # Import from JSON
        >>> imported = exporter.import_history("history.json")
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..config import SearchConfig
from .history_core import SearchCategory, SearchHistoryEntry


class ExportFormat(str, Enum):
    """Supported export formats."""

    JSON = "json"
    CSV = "csv"


class HistoryExporter:
    """Export, import, backup, and restore for search history."""

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.resolve_cache_dir()

    # ── Export ────────────────────────────────────────────────────────────

    def export_history(
        self,
        entries: list[SearchHistoryEntry],
        output_path: str | Path,
        fmt: ExportFormat = ExportFormat.JSON,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> int:
        """
        Export history entries to a file.

        Args:
            entries: History entries to export
            output_path: Output file path
            fmt: Export format (json or csv)
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter

        Returns:
            Number of entries exported
        """
        filtered = self._filter_by_time(entries, start_time, end_time)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == ExportFormat.JSON:
            self._export_json(filtered, output_path)
        elif fmt == ExportFormat.CSV:
            self._export_csv(filtered, output_path)

        return len(filtered)

    def export_history_to_string(
        self,
        entries: list[SearchHistoryEntry],
        fmt: ExportFormat = ExportFormat.JSON,
        *,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> str:
        """
        Export history entries to a string.

        Args:
            entries: History entries to export
            fmt: Export format (json or csv)
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter

        Returns:
            Exported data as string
        """
        filtered = self._filter_by_time(entries, start_time, end_time)

        if fmt == ExportFormat.JSON:
            return self._entries_to_json_string(filtered)
        elif fmt == ExportFormat.CSV:
            return self._entries_to_csv_string(filtered)
        return ""

    def _filter_by_time(
        self,
        entries: list[SearchHistoryEntry],
        start_time: float | None,
        end_time: float | None,
    ) -> list[SearchHistoryEntry]:
        """Filter entries by time range."""
        filtered = entries
        if start_time is not None:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time is not None:
            filtered = [e for e in filtered if e.timestamp <= end_time]
        return filtered

    def _serialize_entry(self, entry: SearchHistoryEntry) -> dict[str, Any]:
        """Serialize a single entry to a JSON-safe dict."""
        d = asdict(entry)
        if isinstance(d.get("languages"), set):
            d["languages"] = sorted(d["languages"])
        if isinstance(d.get("tags"), set):
            d["tags"] = sorted(d["tags"])
        if hasattr(d.get("category"), "value"):
            d["category"] = d["category"].value
        # Add human-readable timestamp
        d["timestamp_iso"] = datetime.fromtimestamp(d["timestamp"]).isoformat()
        return d

    def _export_json(self, entries: list[SearchHistoryEntry], path: Path) -> None:
        """Export entries to JSON file."""
        content = self._entries_to_json_string(entries)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _entries_to_json_string(self, entries: list[SearchHistoryEntry]) -> str:
        """Convert entries to JSON string."""
        data = {
            "version": 1,
            "export_time": time.time(),
            "export_time_iso": datetime.now().isoformat(),
            "total_entries": len(entries),
            "entries": [self._serialize_entry(e) for e in entries],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _export_csv(self, entries: list[SearchHistoryEntry], path: Path) -> None:
        """Export entries to CSV file."""
        content = self._entries_to_csv_string(entries)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _entries_to_csv_string(self, entries: list[SearchHistoryEntry]) -> str:
        """Convert entries to CSV string."""
        output = io.StringIO()
        fieldnames = [
            "timestamp",
            "timestamp_iso",
            "query_pattern",
            "use_regex",
            "use_ast",
            "context",
            "files_matched",
            "items_count",
            "elapsed_ms",
            "filters",
            "session_id",
            "category",
            "languages",
            "paths",
            "success_score",
            "user_rating",
            "tags",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for entry in entries:
            row = self._serialize_entry(entry)
            # Flatten collections for CSV
            if row.get("languages"):
                row["languages"] = ";".join(row["languages"])
            if row.get("tags"):
                row["tags"] = ";".join(row["tags"])
            if row.get("paths"):
                row["paths"] = ";".join(str(p) for p in row["paths"])
            writer.writerow(row)

        return output.getvalue()

    # ── Import ───────────────────────────────────────────────────────────

    def import_history(
        self,
        input_path: str | Path,
        *,
        merge: bool = True,
        existing_entries: list[SearchHistoryEntry] | None = None,
    ) -> list[SearchHistoryEntry]:
        """
        Import history entries from a JSON file.

        Args:
            input_path: Path to JSON file to import
            merge: If True, merge with existing entries; if False, replace
            existing_entries: Current history entries (used when merge=True)

        Returns:
            List of all history entries after import

        Raises:
            ValueError: If the file format is invalid
            FileNotFoundError: If the file does not exist
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Import file not found: {input_path}")

        data = json.loads(input_path.read_text(encoding="utf-8"))
        imported = self._parse_import_data(data)

        if merge and existing_entries:
            # Merge: deduplicate by (timestamp, query_pattern)
            existing_keys = {(e.timestamp, e.query_pattern) for e in existing_entries}
            new_entries = [
                e for e in imported if (e.timestamp, e.query_pattern) not in existing_keys
            ]
            return list(existing_entries) + new_entries

        return imported

    def import_history_from_string(
        self,
        json_string: str,
        *,
        merge: bool = True,
        existing_entries: list[SearchHistoryEntry] | None = None,
    ) -> list[SearchHistoryEntry]:
        """
        Import history entries from a JSON string.

        Args:
            json_string: JSON string to import
            merge: If True, merge with existing entries; if False, replace
            existing_entries: Current history entries (used when merge=True)

        Returns:
            List of all history entries after import
        """
        data = json.loads(json_string)
        imported = self._parse_import_data(data)

        if merge and existing_entries:
            existing_keys = {(e.timestamp, e.query_pattern) for e in existing_entries}
            new_entries = [
                e for e in imported if (e.timestamp, e.query_pattern) not in existing_keys
            ]
            return list(existing_entries) + new_entries

        return imported

    def _parse_import_data(self, data: dict[str, Any]) -> list[SearchHistoryEntry]:
        """Parse imported JSON data into SearchHistoryEntry objects."""
        if "entries" not in data:
            raise ValueError("Invalid import format: missing 'entries' key")

        entries = []
        for entry_data in data["entries"]:
            # Remove extra fields not in the dataclass
            entry_data.pop("timestamp_iso", None)

            # Handle category
            if "category" not in entry_data:
                entry_data["category"] = SearchCategory.GENERAL
            elif isinstance(entry_data["category"], str):
                try:
                    entry_data["category"] = SearchCategory(entry_data["category"])
                except ValueError:
                    entry_data["category"] = SearchCategory.GENERAL

            # Handle sets
            if "languages" in entry_data and entry_data["languages"]:
                entry_data["languages"] = set(entry_data["languages"])
            if "tags" in entry_data and entry_data["tags"]:
                entry_data["tags"] = set(entry_data["tags"])

            entries.append(SearchHistoryEntry(**entry_data))

        return entries

    # ── Backup / Restore ─────────────────────────────────────────────────

    def backup(self, output_path: str | Path) -> dict[str, int]:
        """
        Create a full backup of history, bookmarks, and sessions.

        Args:
            output_path: Path to write the backup file

        Returns:
            Dictionary with counts of backed-up items
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        backup_data: dict[str, Any] = {
            "version": 1,
            "backup_time": time.time(),
            "backup_time_iso": datetime.now().isoformat(),
        }

        counts: dict[str, int] = {}

        # History
        history_file = self.cache_dir / "search_history.json"
        if history_file.exists():
            history_data = json.loads(history_file.read_text(encoding="utf-8"))
            backup_data["history"] = history_data
            counts["history_entries"] = len(history_data.get("entries", []))
        else:
            backup_data["history"] = {"entries": []}
            counts["history_entries"] = 0

        # Bookmarks
        bookmarks_file = self.cache_dir / "bookmarks.json"
        if bookmarks_file.exists():
            bookmarks_data = json.loads(bookmarks_file.read_text(encoding="utf-8"))
            backup_data["bookmarks"] = bookmarks_data
            counts["bookmarks"] = len(bookmarks_data.get("bookmarks", {}))
        else:
            backup_data["bookmarks"] = {"bookmarks": {}}
            counts["bookmarks"] = 0

        # Bookmark folders
        folders_file = self.cache_dir / "bookmark_folders.json"
        if folders_file.exists():
            folders_data = json.loads(folders_file.read_text(encoding="utf-8"))
            backup_data["bookmark_folders"] = folders_data
            counts["folders"] = len(folders_data.get("folders", {}))
        else:
            backup_data["bookmark_folders"] = {"folders": {}}
            counts["folders"] = 0

        # Sessions
        sessions_file = self.cache_dir / "search_sessions.json"
        if sessions_file.exists():
            sessions_data = json.loads(sessions_file.read_text(encoding="utf-8"))
            backup_data["sessions"] = sessions_data
            counts["sessions"] = len(sessions_data.get("sessions", {}))
        else:
            backup_data["sessions"] = {"sessions": {}}
            counts["sessions"] = 0

        # Write backup
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(backup_data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(output_path)

        return counts

    def restore(self, input_path: str | Path) -> dict[str, int]:
        """
        Restore history, bookmarks, and sessions from a backup file.

        Args:
            input_path: Path to the backup file

        Returns:
            Dictionary with counts of restored items

        Raises:
            FileNotFoundError: If the backup file does not exist
            ValueError: If the backup format is invalid
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Backup file not found: {input_path}")

        data = json.loads(input_path.read_text(encoding="utf-8"))

        if "version" not in data:
            raise ValueError("Invalid backup format: missing 'version' key")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}

        # Restore history
        if "history" in data:
            history_file = self.cache_dir / "search_history.json"
            tmp = history_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data["history"], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(history_file)
            counts["history_entries"] = len(data["history"].get("entries", []))

        # Restore bookmarks
        if "bookmarks" in data:
            bookmarks_file = self.cache_dir / "bookmarks.json"
            tmp = bookmarks_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data["bookmarks"], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(bookmarks_file)
            counts["bookmarks"] = len(data["bookmarks"].get("bookmarks", {}))

        # Restore bookmark folders
        if "bookmark_folders" in data:
            folders_file = self.cache_dir / "bookmark_folders.json"
            tmp = folders_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data["bookmark_folders"], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(folders_file)
            counts["folders"] = len(data["bookmark_folders"].get("folders", {}))

        # Restore sessions
        if "sessions" in data:
            sessions_file = self.cache_dir / "search_sessions.json"
            tmp = sessions_file.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(data["sessions"], ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(sessions_file)
            counts["sessions"] = len(data["sessions"].get("sessions", {}))

        return counts

    def validate_backup(self, input_path: str | Path) -> dict[str, Any]:
        """
        Validate a backup file without restoring.

        Args:
            input_path: Path to the backup file

        Returns:
            Dictionary with validation results
        """
        input_path = Path(input_path)
        if not input_path.exists():
            return {"valid": False, "error": "File not found"}

        try:
            data = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return {"valid": False, "error": f"Invalid JSON: {e}"}

        if "version" not in data:
            return {"valid": False, "error": "Missing version field"}

        result: dict[str, Any] = {
            "valid": True,
            "version": data.get("version"),
            "backup_time_iso": data.get("backup_time_iso", "unknown"),
        }

        if "history" in data:
            result["history_entries"] = len(data["history"].get("entries", []))
        if "bookmarks" in data:
            result["bookmarks"] = len(data["bookmarks"].get("bookmarks", {}))
        if "bookmark_folders" in data:
            result["folders"] = len(data["bookmark_folders"].get("folders", {}))
        if "sessions" in data:
            result["sessions"] = len(data["sessions"].get("sessions", {}))

        return result
