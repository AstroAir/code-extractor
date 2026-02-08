"""Base model classes and mixins.

Provides abstract base classes and reusable mixins for all data models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar
from uuid import uuid4

T = TypeVar("T", bound="BaseModel")


def generate_id() -> str:
    """Generate a unique identifier."""
    return uuid4().hex[:12]


@dataclass
class TimestampMixin:
    """Mixin providing created/updated timestamp tracking."""

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class BaseModel(TimestampMixin, ABC):
    """Abstract base model with common fields and behavior."""

    id: str = field(default_factory=generate_id)

    @abstractmethod
    def validate(self) -> bool:
        """Validate the model data.

        Returns:
            True if the model is valid.
        """
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        ...

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Deserialize a model from a dictionary.

        Args:
            data: Dictionary of field values.

        Returns:
            New model instance.
        """
        return cls(**data)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id!r}>"
