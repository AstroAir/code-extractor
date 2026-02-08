"""Post service for content management operations.

Provides CRUD operations for posts, search, and content moderation.
Imports from user_service to create a deliberate cross-dependency for testing.
"""

from __future__ import annotations

from typing import Any

from src.models.post import Post, PostStatus
from src.utils.cache import cached
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostService:
    """Service for blog post management."""

    def __init__(self) -> None:
        self._posts: dict[str, Post] = {}

    async def create_post(
        self,
        title: str,
        content: str,
        author_id: str,
        tags: list[str] | None = None,
    ) -> Post:
        """Create a new blog post.

        Args:
            title: Post title.
            content: Post body content.
            author_id: ID of the post author.
            tags: Optional list of tags.

        Returns:
            The newly created Post.

        Raises:
            ValueError: If post validation fails.
        """
        post = Post(
            title=title,
            content=content,
            author_id=author_id,
            tags=tags or [],
        )

        if not post.validate():
            raise ValueError("Invalid post data")

        self._posts[post.id] = post
        logger.info(f"Created post: {post.id} by author {author_id}")
        return post

    @cached(ttl=30)
    async def get_post(self, post_id: str) -> Post | None:
        """Get a post by its ID.

        Args:
            post_id: The post identifier.

        Returns:
            Post instance or None if not found.
        """
        return self._posts.get(post_id)

    async def update_post(
        self, post_id: str, updates: dict[str, Any]
    ) -> Post | None:
        """Update an existing post.

        Args:
            post_id: Target post ID.
            updates: Dictionary of fields to update.

        Returns:
            Updated post or None if not found.
        """
        post = self._posts.get(post_id)
        if post is None:
            return None

        if "title" in updates:
            post.title = updates["title"]
        if "content" in updates:
            post.content = updates["content"]
        if "tags" in updates:
            post.tags = updates["tags"]

        post.touch()
        logger.info(f"Updated post: {post_id}")
        return post

    async def delete_post(self, post_id: str) -> bool:
        """Delete a post by ID.

        Args:
            post_id: The post to delete.

        Returns:
            True if the post was found and deleted.
        """
        if post_id in self._posts:
            del self._posts[post_id]
            logger.info(f"Deleted post: {post_id}")
            return True
        return False

    async def delete_posts_by_author(self, author_id: str) -> int:
        """Delete all posts by a specific author.

        Args:
            author_id: The author whose posts to delete.

        Returns:
            Number of posts deleted.
        """
        to_delete = [
            pid for pid, post in self._posts.items() if post.author_id == author_id
        ]
        for pid in to_delete:
            del self._posts[pid]
        logger.info(f"Deleted {len(to_delete)} posts for author {author_id}")
        return len(to_delete)

    async def get_posts_by_author(
        self, author_id: str, status: PostStatus | None = None
    ) -> list[Post]:
        """Get all posts by a specific author.

        Args:
            author_id: The author ID to filter by.
            status: Optional status filter.

        Returns:
            List of matching posts.
        """
        posts = [p for p in self._posts.values() if p.author_id == author_id]
        if status is not None:
            posts = [p for p in posts if p.status == status]
        return sorted(posts, key=lambda p: p.created_at, reverse=True)

    async def search_posts(self, query: str) -> list[Post]:
        """Search posts by title and content.

        Args:
            query: Search term.

        Returns:
            List of matching posts.
        """
        query_lower = query.lower()
        return [
            p
            for p in self._posts.values()
            if query_lower in p.title.lower() or query_lower in p.content.lower()
        ]

    async def get_posts_by_tag(self, tag: str) -> list[Post]:
        """Get all posts with a specific tag.

        Args:
            tag: Tag to filter by.

        Returns:
            List of posts with the given tag.
        """
        normalized = tag.strip().lower()
        return [p for p in self._posts.values() if normalized in p.tags]

    async def publish_post(self, post_id: str) -> bool:
        """Publish a draft post.

        Args:
            post_id: The post to publish.

        Returns:
            True if the post was successfully published.
        """
        post = self._posts.get(post_id)
        if post is None:
            return False
        return post.publish()

    async def get_statistics(self) -> dict[str, Any]:
        """Get aggregate statistics about posts."""
        posts = list(self._posts.values())
        return {
            "total": len(posts),
            "published": sum(1 for p in posts if p.is_published),
            "drafts": sum(1 for p in posts if p.status == PostStatus.DRAFT),
            "archived": sum(1 for p in posts if p.status == PostStatus.ARCHIVED),
            "total_views": sum(p.view_count for p in posts),
            "total_likes": sum(p.like_count for p in posts),
        }
