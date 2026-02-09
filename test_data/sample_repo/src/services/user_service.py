"""User service for managing user lifecycle operations.

Provides CRUD operations, search, and user management.
Intentionally imports from post_service to create a cross-dependency.
"""

from __future__ import annotations

from typing import Any

from src.models.user import User, UserRole
from src.services.post_service import PostService
from src.utils.cache import cached
from src.utils.logger import get_logger

logger = get_logger(__name__)


class UserService:
    """Service for user management operations."""

    def __init__(self) -> None:
        self._users: dict[str, User] = {}
        self._post_service = PostService()

    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
    ) -> User:
        """Create a new user account.

        Args:
            username: Unique username.
            email: User email address.
            password: Plaintext password to hash.
            role: User role assignment.

        Returns:
            The newly created User.

        Raises:
            ValueError: If username already exists or validation fails.
        """
        if username in self._users:
            raise ValueError(f"Username '{username}' already exists")

        user = User(username=username, email=email, role=role)
        user.set_password(password)

        if not user.validate():
            raise ValueError("Invalid user data")

        self._users[username] = user
        logger.info(f"Created user: {username}")
        return user

    @cached(ttl=60)
    async def get_user(self, username: str) -> User | None:
        """Get a user by username.

        Args:
            username: The username to look up.

        Returns:
            User instance or None if not found.
        """
        return self._users.get(username)

    async def update_profile(self, username: str, profile_data: dict[str, Any]) -> User | None:
        """Update a user's profile information.

        Args:
            username: Target username.
            profile_data: Dictionary of profile fields to update.

        Returns:
            Updated user or None if user not found.
        """
        user = self._users.get(username)
        if user is None:
            return None

        profile = user.profile
        if "bio" in profile_data:
            profile.bio = profile_data["bio"]
        if "avatar_url" in profile_data:
            profile.avatar_url = profile_data["avatar_url"]
        if "location" in profile_data:
            profile.location = profile_data["location"]
        if "website" in profile_data:
            profile.website = profile_data["website"]

        user.touch()
        logger.info(f"Updated profile for user: {username}")
        return user

    async def delete_user(self, username: str) -> bool:
        """Delete a user and all their posts.

        Args:
            username: The username to delete.

        Returns:
            True if the user was found and deleted.
        """
        user = self._users.get(username)
        if user is None:
            return False

        # Delete user's posts via PostService
        await self._post_service.delete_posts_by_author(user.id)

        del self._users[username]
        logger.info(f"Deleted user: {username}")
        return True

    async def list_users(
        self,
        role: UserRole | None = None,
        active_only: bool = True,
    ) -> list[User]:
        """List users with optional filtering.

        Args:
            role: Filter by specific role.
            active_only: Only return active users.

        Returns:
            List of matching users.
        """
        users = list(self._users.values())
        if role is not None:
            users = [u for u in users if u.role == role]
        if active_only:
            users = [u for u in users if u.is_active]
        return users

    async def search_users(self, query: str) -> list[User]:
        """Search users by username or email.

        Args:
            query: Search term to match against.

        Returns:
            List of matching users.
        """
        query_lower = query.lower()
        return [
            u
            for u in self._users.values()
            if query_lower in u.username.lower() or query_lower in u.email.lower()
        ]

    async def get_user_with_posts(self, username: str) -> dict[str, Any] | None:
        """Get user data along with their posts.

        Args:
            username: The username to look up.

        Returns:
            Dictionary with user data and posts, or None.
        """
        user = self._users.get(username)
        if user is None:
            return None

        posts = await self._post_service.get_posts_by_author(user.id)
        return {
            "user": user.to_dict(),
            "posts": [p.to_dict() for p in posts],
            "post_count": len(posts),
        }
