"""Main application entry point.

This module sets up the FastAPI application instance, registers routes,
and configures middleware for the web service.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from src.api.middleware import LoggingMiddleware, RateLimitMiddleware
from src.api.routes import router
from src.config import Settings, get_settings
from src.db.connection import DatabasePool


@asynccontextmanager
async def lifespan(app: object) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup/shutdown events."""
    settings = get_settings()
    pool = DatabasePool(settings.database_url)
    await pool.connect()
    yield
    await pool.disconnect()


def create_app(settings: Settings | None = None) -> object:
    """Application factory for creating configured app instances.

    Args:
        settings: Optional settings override for testing.

    Returns:
        Configured application instance.
    """
    if settings is None:
        settings = get_settings()

    # FIXME: Add proper CORS configuration
    app_instance = {
        "title": settings.app_name,
        "version": settings.app_version,
        "debug": settings.debug,
    }
    return app_instance


def register_routes(app: object) -> None:
    """Register all API routes with the application."""
    # TODO: Add versioned route prefixes (e.g., /api/v1)
    pass


def register_middleware(app: object) -> None:
    """Register middleware stack."""
    pass


async def health_check() -> dict[str, str]:
    """Health check endpoint for monitoring."""
    return {"status": "healthy", "version": "0.2.0"}


if __name__ == "__main__":
    application = create_app()
    print(f"Starting application: {application}")
