"""Database migration management.

Provides a simple migration runner for applying and rolling back
schema changes in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable


@dataclass
class Migration:
    """Represents a single database migration."""

    version: str
    name: str
    up: Callable[[], None]
    down: Callable[[], None]
    applied_at: datetime | None = None

    @property
    def is_applied(self) -> bool:
        """Check if this migration has been applied."""
        return self.applied_at is not None


class MigrationRunner:
    """Manages and executes database migrations in order.

    Tracks applied migrations and supports both forward (up)
    and backward (down) migration execution.
    """

    def __init__(self) -> None:
        self._migrations: list[Migration] = []
        self._applied: set[str] = set()

    def register(self, version: str, name: str, up: Callable[[], None], down: Callable[[], None]) -> None:
        """Register a new migration.

        Args:
            version: Unique version identifier (e.g., "001").
            name: Human-readable migration name.
            up: Forward migration function.
            down: Rollback migration function.
        """
        migration = Migration(version=version, name=name, up=up, down=down)
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)

    def migrate_up(self, target_version: str | None = None) -> list[str]:
        """Apply pending migrations up to target version.

        Args:
            target_version: Optional target version to migrate to.
                If None, applies all pending migrations.

        Returns:
            List of applied migration versions.
        """
        applied = []
        for migration in self._migrations:
            if migration.version in self._applied:
                continue
            if target_version and migration.version > target_version:
                break

            migration.up()
            migration.applied_at = datetime.now(timezone.utc)
            self._applied.add(migration.version)
            applied.append(migration.version)

        return applied

    def migrate_down(self, target_version: str) -> list[str]:
        """Roll back migrations down to target version.

        Args:
            target_version: Version to roll back to (exclusive).

        Returns:
            List of rolled-back migration versions.
        """
        rolled_back = []
        for migration in reversed(self._migrations):
            if migration.version <= target_version:
                break
            if migration.version not in self._applied:
                continue

            migration.down()
            migration.applied_at = None
            self._applied.discard(migration.version)
            rolled_back.append(migration.version)

        return rolled_back

    @property
    def pending_migrations(self) -> list[Migration]:
        """Get list of unapplied migrations."""
        return [m for m in self._migrations if m.version not in self._applied]

    @property
    def applied_migrations(self) -> list[Migration]:
        """Get list of applied migrations."""
        return [m for m in self._migrations if m.version in self._applied]

    def get_status(self) -> dict[str, Any]:
        """Get migration status summary."""
        return {
            "total": len(self._migrations),
            "applied": len(self._applied),
            "pending": len(self._migrations) - len(self._applied),
            "latest_version": self._migrations[-1].version if self._migrations else None,
        }
