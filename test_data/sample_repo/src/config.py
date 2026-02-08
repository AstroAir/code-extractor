"""Application configuration management.

Loads settings from environment variables with sensible defaults.
Supports multiple environments: development, staging, production.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any


class Environment(Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    app_name: str = "SampleWebApp"
    app_version: str = "0.2.0"
    debug: bool = False
    environment: Environment = Environment.DEVELOPMENT
    database_url: str = "sqlite:///./app.db"
    redis_url: str = "redis://localhost:6379/0"
    secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 30
    max_connections: int = 10
    request_timeout: int = 30
    allowed_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings from environment.

    Returns:
        Application settings instance.
    """
    env_name = os.getenv("APP_ENV", "development")
    environment = Environment(env_name)

    return Settings(
        debug=environment == Environment.DEVELOPMENT,
        environment=environment,
        database_url=os.getenv("DATABASE_URL", "sqlite:///./app.db"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        secret_key=os.getenv("SECRET_KEY", "change-me-in-production"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )


# Global constants
MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 20
SUPPORTED_LANGUAGES = ["en", "zh", "ja", "ko"]
API_PREFIX = "/api/v1"

# Feature flags
FEATURES: dict[str, bool] = {
    "enable_cache": True,
    "enable_rate_limit": True,
    "enable_websocket": False,  # TODO: Implement WebSocket support
    "enable_oauth": False,
}


def is_feature_enabled(feature_name: str) -> bool:
    """Check if a feature flag is enabled."""
    return FEATURES.get(feature_name, False)
