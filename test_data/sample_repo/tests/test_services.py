"""Tests for business logic services."""

from __future__ import annotations

import pytest

from src.models.user import User
from src.services.auth import create_token, verify_token
from src.services.post_service import PostService


class TestAuthService:
    """Tests for AuthService."""

    def test_create_token(self) -> None:
        user = User(username="alice", email="alice@example.com")
        token = create_token(user)
        assert isinstance(token, str)
        assert len(token) > 0

    def test_verify_valid_token(self) -> None:
        user = User(username="alice", email="alice@example.com")
        token = create_token(user)
        payload = verify_token(token)
        assert payload is not None

    def test_verify_invalid_token(self) -> None:
        payload = verify_token("")
        assert payload is None


class TestPostService:
    """Tests for PostService."""

    @pytest.fixture
    def service(self) -> PostService:
        return PostService()

    @pytest.mark.asyncio
    async def test_create_and_get_post(self, service: PostService) -> None:
        post = await service.create_post(
            title="Test Post Title",
            content="This is a test post with sufficient content.",
            author_id="user1",
            tags=["test"],
        )
        retrieved = await service.get_post(post.id)
        assert retrieved is not None
        assert retrieved.title == "Test Post Title"

    @pytest.mark.asyncio
    async def test_search_posts(self, service: PostService) -> None:
        await service.create_post("Python Tips", "Learn Python programming basics.", "u1")
        await service.create_post("Java Guide", "Learn Java programming basics.", "u1")
        results = await service.search_posts("Python")
        assert len(results) == 1
        assert results[0].title == "Python Tips"
