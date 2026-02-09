"""Tests for data models."""

from __future__ import annotations

from src.models.post import Post, PostStatus
from src.models.user import User, UserRole


class TestUserModel:
    """Tests for the User model."""

    def test_create_user(self) -> None:
        user = User(username="alice", email="alice@example.com")
        assert user.username == "alice"
        assert user.role == UserRole.USER

    def test_validate_user(self) -> None:
        valid = User(username="alice", email="alice@example.com")
        assert valid.validate() is True

        invalid = User(username="ab", email="bad-email")
        assert invalid.validate() is False

    def test_password_hashing(self) -> None:
        user = User(username="alice", email="alice@example.com")
        user.set_password("secret123")
        assert user.check_password("secret123") is True
        assert user.check_password("wrong") is False

    def test_promote_user(self) -> None:
        user = User(username="alice", email="alice@example.com", role=UserRole.USER)
        user.promote(UserRole.MODERATOR)
        assert user.role == UserRole.MODERATOR

    def test_to_dict(self) -> None:
        user = User(username="alice", email="alice@example.com")
        data = user.to_dict()
        assert data["username"] == "alice"
        assert "password_hash" not in data


class TestPostModel:
    """Tests for the Post model."""

    def test_create_post(self) -> None:
        post = Post(
            title="Hello World", content="This is my first post content.", author_id="user1"
        )
        assert post.status == PostStatus.DRAFT

    def test_publish_post(self) -> None:
        post = Post(title="Valid Title", content="Valid content for the post body.", author_id="u1")
        assert post.publish() is True
        assert post.is_published is True

    def test_add_remove_tags(self) -> None:
        post = Post(title="Tagged", content="A post with tags for testing.", author_id="u1")
        post.add_tag("python")
        post.add_tag("Python")  # duplicate
        assert len(post.tags) == 1
        post.remove_tag("python")
        assert len(post.tags) == 0
