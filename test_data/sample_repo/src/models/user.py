"""User model with authentication support.

Defines the User entity with role-based access control,
password hashing, and profile management.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.models.base import BaseModel


class UserRole(Enum):
    """User role enumeration for access control."""

    GUEST = "guest"
    USER = "user"
    MODERATOR = "moderator"
    ADMIN = "admin"


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


@dataclass
class UserProfile:
    """Extended user profile information."""

    bio: str = ""
    avatar_url: str = ""
    location: str = ""
    website: str = ""

    def is_complete(self) -> bool:
        """Check if the profile has all fields filled."""
        return all([self.bio, self.avatar_url, self.location])


@dataclass
class User(BaseModel):
    """User model with authentication and profile."""

    username: str = ""
    email: str = ""
    password_hash: str = ""
    role: UserRole = UserRole.USER
    is_active: bool = True
    profile: UserProfile = field(default_factory=UserProfile)
    followers_count: int = 0
    following_count: int = 0

    def validate(self) -> bool:
        """Validate user data.

        Returns:
            True if username and email are valid.
        """
        if not self.username or len(self.username) < 3:
            return False
        if not EMAIL_REGEX.match(self.email):
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize user to dictionary, excluding password hash."""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "is_active": self.is_active,
            "profile": {
                "bio": self.profile.bio,
                "avatar_url": self.profile.avatar_url,
                "location": self.profile.location,
                "website": self.profile.website,
            },
            "followers_count": self.followers_count,
            "following_count": self.following_count,
            "created_at": self.created_at.isoformat(),
        }

    def set_password(self, raw_password: str) -> None:
        """Hash and store a password.

        Args:
            raw_password: The plaintext password to hash.
        """
        self.password_hash = hashlib.sha256(raw_password.encode()).hexdigest()

    def check_password(self, raw_password: str) -> bool:
        """Verify a password against the stored hash."""
        candidate = hashlib.sha256(raw_password.encode()).hexdigest()
        return candidate == self.password_hash

    @property
    def display_name(self) -> str:
        """Get the user's display name."""
        return self.username.title()

    @property
    def is_admin(self) -> bool:
        """Check if the user has admin privileges."""
        return self.role == UserRole.ADMIN

    def promote(self, new_role: UserRole) -> None:
        """Promote user to a higher role."""
        role_order = [UserRole.GUEST, UserRole.USER, UserRole.MODERATOR, UserRole.ADMIN]
        current_idx = role_order.index(self.role)
        new_idx = role_order.index(new_role)
        if new_idx > current_idx:
            self.role = new_role
            self.touch()
