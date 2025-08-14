from __future__ import annotations

import re
import subprocess
from pathlib import Path

from ..core.types import FileMetadata, Language, MetadataFilters


def get_file_author(path: Path) -> str | None:
    """
    Get the author of a file using git blame or file system attributes.
    Returns the most recent author or None if unavailable.
    """
    try:
        # Try git blame first
        result = subprocess.run(
            ["git", "log", "-1", "--format=%an", str(path)],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=path.parent if path.parent.exists() else None,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, FileNotFoundError):
        pass

    # Fallback: try to get owner from file system (Unix-like systems)
    try:
        import pwd

        stat_info = path.stat()
        owner = pwd.getpwuid(stat_info.st_uid)
        return owner.pw_name
    except (ImportError, KeyError, OSError):
        pass

    return None


def apply_metadata_filters(metadata: FileMetadata, filters: MetadataFilters) -> bool:
    """
    Apply metadata filters to determine if a file should be included.

    Args:
        metadata: File metadata to check
        filters: Metadata filters to apply

    Returns:
        True if file passes all filters, False otherwise
    """
    # Size filters
    if filters.min_size is not None and metadata.size < filters.min_size:
        return False
    if filters.max_size is not None and metadata.size > filters.max_size:
        return False

    # Date filters
    if filters.modified_after is not None and metadata.mtime < filters.modified_after:
        return False
    if filters.modified_before is not None and metadata.mtime > filters.modified_before:
        return False

    if filters.created_after is not None:
        if metadata.created_date is None or metadata.created_date < filters.created_after:
            return False
    if filters.created_before is not None:
        if metadata.created_date is None or metadata.created_date > filters.created_before:
            return False

    # Line count filters
    if filters.min_lines is not None:
        if metadata.line_count is None or metadata.line_count < filters.min_lines:
            return False
    if filters.max_lines is not None:
        if metadata.line_count is None or metadata.line_count > filters.max_lines:
            return False

    # Author filter
    if filters.author_pattern is not None:
        if metadata.author is None:
            return False
        if not re.search(filters.author_pattern, metadata.author, re.IGNORECASE):
            return False

    # Encoding filter
    if filters.encoding_pattern is not None:
        if not re.search(filters.encoding_pattern, metadata.encoding, re.IGNORECASE):
            return False

    # Language filter
    if filters.languages is not None:
        if metadata.language not in filters.languages:
            return False

    return True


def create_size_filter(
    min_size: str | None = None, max_size: str | None = None
) -> tuple[int | None, int | None]:
    """
    Parse human-readable size strings into bytes.

    Args:
        min_size: Minimum size (e.g., "1KB", "5MB", "100")
        max_size: Maximum size (e.g., "10MB", "1GB")

    Returns:
        Tuple of (min_bytes, max_bytes)
    """

    def parse_size(size_str: str) -> int:
        if not size_str:
            return 0

        size_str = size_str.upper().strip()

        # Extract number and unit
        match = re.match(r"^(\d+(?:\.\d+)?)\s*([KMGT]?B?)?$", size_str)
        if not match:
            # Try just a number
            try:
                return int(size_str)
            except ValueError:
                raise ValueError(f"Invalid size format: {size_str}")

        number, unit = match.groups()
        number = float(number)

        if not unit or unit == "B":
            multiplier = 1
        elif unit in ("K", "KB"):
            multiplier = 1024
        elif unit in ("M", "MB"):
            multiplier = 1024 * 1024
        elif unit in ("G", "GB"):
            multiplier = 1024 * 1024 * 1024
        elif unit in ("T", "TB"):
            multiplier = 1024 * 1024 * 1024 * 1024
        else:
            raise ValueError(f"Unknown size unit: {unit}")

        return int(number * multiplier)

    min_bytes = parse_size(min_size) if min_size else None
    max_bytes = parse_size(max_size) if max_size else None

    return min_bytes, max_bytes


def create_date_filter(
    modified_after: str | None = None,
    modified_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
) -> tuple[float | None, float | None, float | None, float | None]:
    """
    Parse date strings into Unix timestamps.

    Supports formats:
    - ISO format: "2023-12-01", "2023-12-01T10:30:00"
    - Relative: "1d", "2w", "3m", "1y" (days, weeks, months, years ago)
    - Unix timestamp: "1701234567"

    Returns:
        Tuple of (modified_after, modified_before, created_after, created_before) timestamps
    """
    from datetime import datetime, timedelta

    def parse_date(date_str: str) -> float:
        if not date_str:
            return 0.0

        date_str = date_str.strip()

        # Try Unix timestamp first
        try:
            return float(date_str)
        except ValueError:
            pass

        # Try relative format (e.g., "1d", "2w", "3m", "1y")
        relative_match = re.match(r"^(\d+)([dwmy])$", date_str.lower())
        if relative_match:
            amount, unit = relative_match.groups()
            amount = int(amount)

            now = datetime.now()
            if unit == "d":
                target_date = now - timedelta(days=amount)
            elif unit == "w":
                target_date = now - timedelta(weeks=amount)
            elif unit == "m":
                target_date = now - timedelta(days=amount * 30)  # Approximate
            elif unit == "y":
                target_date = now - timedelta(days=amount * 365)  # Approximate
            else:
                raise ValueError(f"Unknown time unit: {unit}")

            return target_date.timestamp()

        # Try ISO format
        try:
            # Handle various ISO formats
            for fmt in ["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"]:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
        except ValueError:
            pass

        raise ValueError(f"Invalid date format: {date_str}")

    return (
        parse_date(modified_after) if modified_after else None,
        parse_date(modified_before) if modified_before else None,
        parse_date(created_after) if created_after else None,
        parse_date(created_before) if created_before else None,
    )


def create_metadata_filters(
    min_size: str | None = None,
    max_size: str | None = None,
    modified_after: str | None = None,
    modified_before: str | None = None,
    created_after: str | None = None,
    created_before: str | None = None,
    min_lines: int | None = None,
    max_lines: int | None = None,
    author_pattern: str | None = None,
    encoding_pattern: str | None = None,
    languages: set[Language] | None = None,
) -> MetadataFilters:
    """
    Create MetadataFilters from user-friendly parameters.

    Args:
        min_size, max_size: Human-readable size strings
        modified_after, modified_before: Date strings
        created_after, created_before: Date strings
        min_lines, max_lines: Line count limits
        author_pattern: Regex pattern for author names
        encoding_pattern: Regex pattern for file encoding
        languages: Set of languages to include

    Returns:
        MetadataFilters object
    """
    # Parse sizes
    min_bytes, max_bytes = create_size_filter(min_size, max_size)

    # Parse dates
    mod_after, mod_before, create_after, create_before = create_date_filter(
        modified_after, modified_before, created_after, created_before
    )

    return MetadataFilters(
        min_size=min_bytes,
        max_size=max_bytes,
        modified_after=mod_after,
        modified_before=mod_before,
        created_after=create_after,
        created_before=create_before,
        min_lines=min_lines,
        max_lines=max_lines,
        author_pattern=author_pattern,
        encoding_pattern=encoding_pattern,
        languages=languages,
    )
