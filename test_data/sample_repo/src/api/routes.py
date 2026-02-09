"""API route definitions.

Defines all HTTP endpoints for the web application,
including user management, posts, and authentication.
"""

from __future__ import annotations

from typing import Any

from src.config import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from src.models.user import UserRole
from src.services.auth import AuthService, require_auth, require_role
from src.services.post_service import PostService
from src.services.user_service import UserService

router: dict[str, Any] = {}

auth_service = AuthService()
user_service = UserService()
post_service = PostService()


# --- Authentication endpoints ---


async def login(username: str, password: str) -> dict[str, Any]:
    """Handle user login request.

    Args:
        username: The username.
        password: The password.

    Returns:
        Token response or error.
    """
    token = await auth_service.login(username, password)
    if token is None:
        return {"error": "Invalid credentials", "status": 401}
    return {"token": token, "status": 200}


async def logout(token: str) -> dict[str, Any]:
    """Handle user logout request."""
    success = await auth_service.logout(token)
    return {"success": success, "status": 200}


# --- User endpoints ---


@require_auth
async def get_current_user(**kwargs: Any) -> dict[str, Any]:
    """Get the currently authenticated user's profile."""
    current_user = kwargs["current_user"]
    user = await user_service.get_user(current_user.username)
    if user is None:
        return {"error": "User not found", "status": 404}
    return {"user": user.to_dict(), "status": 200}


async def create_user(username: str, email: str, password: str) -> dict[str, Any]:
    """Register a new user account.

    Args:
        username: Desired username.
        email: User email address.
        password: Account password.

    Returns:
        Created user data or error.
    """
    try:
        user = await user_service.create_user(username, email, password)
        return {"user": user.to_dict(), "status": 201}
    except ValueError as e:
        return {"error": str(e), "status": 400}


async def list_users(
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    """List users with pagination.

    Args:
        page: Page number (1-indexed).
        page_size: Number of results per page.

    Returns:
        Paginated user list.
    """
    page_size = min(page_size, MAX_PAGE_SIZE)
    users = await user_service.list_users()
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "users": [u.to_dict() for u in users[start:end]],
        "total": len(users),
        "page": page,
        "page_size": page_size,
        "status": 200,
    }


# --- Post endpoints ---


@require_auth
async def create_post_handler(
    title: str, content: str, tags: list[str] | None = None, **kwargs: Any
) -> dict[str, Any]:
    """Create a new blog post.

    Args:
        title: Post title.
        content: Post content body.
        tags: Optional list of tags.

    Returns:
        Created post data or error.
    """
    current_user = kwargs["current_user"]
    try:
        post = await post_service.create_post(
            title=title,
            content=content,
            author_id=current_user.user_id,
            tags=tags,
        )
        return {"post": post.to_dict(), "status": 201}
    except ValueError as e:
        return {"error": str(e), "status": 400}


async def get_post_handler(post_id: str) -> dict[str, Any]:
    """Get a single post by ID."""
    post = await post_service.get_post(post_id)
    if post is None:
        return {"error": "Post not found", "status": 404}
    post.increment_views()
    return {"post": post.to_dict(), "status": 200}


async def search_posts_handler(
    query: str,
    tag: str | None = None,
) -> dict[str, Any]:
    """Search posts by query and optional tag filter.

    Args:
        query: Search term.
        tag: Optional tag filter.

    Returns:
        List of matching posts.
    """
    if tag:
        posts = await post_service.get_posts_by_tag(tag)
    else:
        posts = await post_service.search_posts(query)
    return {
        "posts": [p.to_dict() for p in posts],
        "count": len(posts),
        "status": 200,
    }


# --- Admin endpoints ---


@require_auth
@require_role(UserRole.ADMIN)
async def admin_dashboard(**kwargs: Any) -> dict[str, Any]:
    """Admin dashboard with statistics.

    Requires ADMIN role.
    """
    stats = await post_service.get_statistics()
    users = await user_service.list_users(active_only=False)
    return {
        "post_stats": stats,
        "user_count": len(users),
        "active_sessions": auth_service.active_session_count,
        "status": 200,
    }
