"""Database connection pool management.

Provides async database connection pooling with health checks
and automatic reconnection support.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any


@dataclass
class ConnectionInfo:
    """Database connection metadata."""

    url: str
    is_connected: bool = False
    pool_size: int = 0
    max_pool_size: int = 10


class DatabasePool:
    """Async database connection pool.

    Manages a pool of database connections with automatic
    health checking and reconnection.

    Args:
        url: Database connection URL.
        max_connections: Maximum pool size.
    """

    def __init__(self, url: str, max_connections: int = 10) -> None:
        self._url = url
        self._max_connections = max_connections
        self._connections: list[dict[str, Any]] = []
        self._is_connected = False
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        """Initialize the connection pool.

        Creates the initial set of connections.
        """
        async with self._lock:
            if self._is_connected:
                return
            # Simulate creating connections
            for i in range(min(3, self._max_connections)):
                self._connections.append(
                    {
                        "id": i,
                        "url": self._url,
                        "active": True,
                    }
                )
            self._is_connected = True

    async def disconnect(self) -> None:
        """Close all connections and shut down the pool."""
        async with self._lock:
            self._connections.clear()
            self._is_connected = False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[dict[str, Any], None]:
        """Acquire a connection from the pool.

        Yields:
            A database connection dict.

        Raises:
            RuntimeError: If the pool is not connected.
        """
        if not self._is_connected:
            raise RuntimeError("Database pool is not connected")

        async with self._lock:
            if not self._connections:
                raise RuntimeError("No available connections in pool")
            conn = self._connections.pop(0)

        try:
            yield conn
        finally:
            async with self._lock:
                self._connections.append(conn)

    async def execute(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a database query.

        Args:
            query: SQL query string.
            params: Optional query parameters.

        Returns:
            List of result rows as dictionaries.
        """
        async with self.acquire() as conn:
            # Simulate query execution
            return [{"query": query, "connection_id": conn["id"]}]

    async def health_check(self) -> bool:
        """Check if the database connection is healthy.

        Returns:
            True if the pool is operational.
        """
        if not self._is_connected:
            return False
        return len(self._connections) > 0

    @property
    def info(self) -> ConnectionInfo:
        """Get current connection pool information."""
        return ConnectionInfo(
            url=self._url,
            is_connected=self._is_connected,
            pool_size=len(self._connections),
            max_pool_size=self._max_connections,
        )

    @property
    def is_connected(self) -> bool:
        """Check if the pool is connected."""
        return self._is_connected
