"""Shared test fixtures for the sample web application."""

from __future__ import annotations

import pytest

from src.models.post import Post
from src.models.user import User, UserRole


@pytest.fixture
def sample_user() -> User:
    """Create a sample user for testing."""
    user = User(
        username="testuser",
        email="test@example.com",
        role=UserRole.USER,
    )
    user.set_password("TestPassword123")
    return user


@pytest.fixture
def admin_user() -> User:
    """Create an admin user for testing."""
    user = User(
        username="admin",
        email="admin@example.com",
        role=UserRole.ADMIN,
    )
    user.set_password("AdminPass456")
    return user


@pytest.fixture
def sample_post(sample_user: User) -> Post:
    """Create a sample post for testing."""
    return Post(
        title="Test Blog Post Title",
        content="This is a test blog post with enough content to pass validation.",
        author_id=sample_user.id,
        tags=["test", "sample"],
    )
