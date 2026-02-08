"""Post model for content management.

Defines the Post entity with status workflow,
tagging, and content validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.models.base import BaseModel


class PostStatus(Enum):
    """Post publication status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


@dataclass
class Post(BaseModel):
    """Blog post model with content and metadata."""

    title: str = ""
    content: str = ""
    author_id: str = ""
    status: PostStatus = PostStatus.DRAFT
    tags: list[str] = field(default_factory=list)
    view_count: int = 0
    like_count: int = 0

    def validate(self) -> bool:
        """Validate post data.

        Returns:
            True if title and content meet minimum requirements.
        """
        if not self.title or len(self.title) < 5:
            return False
        if not self.content or len(self.content) < 10:
            return False
        if not self.author_id:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize post to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "author_id": self.author_id,
            "status": self.status.value,
            "tags": self.tags,
            "view_count": self.view_count,
            "like_count": self.like_count,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def publish(self) -> bool:
        """Publish the post if it passes validation.

        Returns:
            True if the post was successfully published.
        """
        if not self.validate():
            return False
        self.status = PostStatus.PUBLISHED
        self.touch()
        return True

    def archive(self) -> None:
        """Archive the post."""
        self.status = PostStatus.ARCHIVED
        self.touch()

    def add_tag(self, tag: str) -> None:
        """Add a tag to the post if not already present."""
        normalized = tag.strip().lower()
        if normalized and normalized not in self.tags:
            self.tags.append(normalized)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the post."""
        normalized = tag.strip().lower()
        if normalized in self.tags:
            self.tags.remove(normalized)

    @property
    def summary(self) -> str:
        """Get a short summary of the post content."""
        if len(self.content) <= 200:
            return self.content
        return self.content[:197] + "..."

    @property
    def is_published(self) -> bool:
        """Check if the post is currently published."""
        return self.status == PostStatus.PUBLISHED

    def increment_views(self) -> int:
        """Increment and return the view count."""
        self.view_count += 1
        return self.view_count
