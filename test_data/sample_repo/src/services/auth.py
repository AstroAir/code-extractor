"""Authentication service with JWT token management.

Provides user authentication, token generation/validation,
and session management functionality.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from src.config import get_settings
from src.models.user import User, UserRole


@dataclass
class TokenPayload:
    """JWT token payload data."""

    user_id: str
    username: str
    role: str
    issued_at: float
    expires_at: float

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        return time.time() > self.expires_at


def create_token(user: User) -> str:
    """Create a JWT-like token for the given user.

    Args:
        user: The user to generate a token for.

    Returns:
        Encoded token string.
    """
    settings = get_settings()
    payload = f"{user.id}:{user.username}:{user.role.value}:{time.time()}"
    token = hashlib.sha256(f"{payload}:{settings.secret_key}".encode()).hexdigest()
    return token


def verify_token(token: str) -> TokenPayload | None:
    """Verify and decode a token.

    Args:
        token: The token string to verify.

    Returns:
        Decoded payload or None if invalid.
    """
    # Simplified verification for demonstration
    if not token or len(token) < 10:
        return None
    return TokenPayload(
        user_id="unknown",
        username="unknown",
        role="user",
        issued_at=time.time(),
        expires_at=time.time() + 3600,
    )


def require_auth(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to require authentication for a function.

    Args:
        func: The function to protect.

    Returns:
        Wrapped function with auth check.
    """

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = kwargs.get("token")
        if not token:
            raise PermissionError("Authentication required")
        payload = verify_token(token)
        if payload is None or payload.is_expired:
            raise PermissionError("Invalid or expired token")
        kwargs["current_user"] = payload
        return await func(*args, **kwargs)

    return wrapper


def require_role(role: UserRole) -> Callable[..., Any]:
    """Decorator factory to require a specific user role.

    Args:
        role: The minimum required role.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            current_user = kwargs.get("current_user")
            if current_user is None:
                raise PermissionError("Authentication required")
            role_hierarchy = {
                UserRole.GUEST: 0,
                UserRole.USER: 1,
                UserRole.MODERATOR: 2,
                UserRole.ADMIN: 3,
            }
            if role_hierarchy.get(UserRole(current_user.role), 0) < role_hierarchy[role]:
                raise PermissionError(f"Role {role.value} or higher required")
            return await func(*args, **kwargs)

        return wrapper

    return decorator


class AuthService:
    """Authentication service managing user sessions."""

    def __init__(self) -> None:
        self._active_sessions: dict[str, TokenPayload] = {}
        self._failed_attempts: dict[str, int] = {}
        self._max_failed_attempts = 5

    async def login(self, username: str, password: str) -> str | None:
        """Authenticate a user and return a token.

        Args:
            username: The username to authenticate.
            password: The plaintext password.

        Returns:
            Token string or None if authentication fails.
        """
        # Check for too many failed attempts
        if self._failed_attempts.get(username, 0) >= self._max_failed_attempts:
            return None

        # Simplified: in real app, look up user from database
        user = User(username=username)
        user.set_password(password)
        token = create_token(user)
        self._active_sessions[token] = TokenPayload(
            user_id=user.id,
            username=username,
            role=user.role.value,
            issued_at=time.time(),
            expires_at=time.time() + 1800,
        )
        return token

    async def logout(self, token: str) -> bool:
        """Invalidate a session token.

        Args:
            token: The token to invalidate.

        Returns:
            True if the token was found and removed.
        """
        if token in self._active_sessions:
            del self._active_sessions[token]
            return True
        return False

    def get_session(self, token: str) -> TokenPayload | None:
        """Get the session data for a token."""
        session = self._active_sessions.get(token)
        if session and session.is_expired:
            del self._active_sessions[token]
            return None
        return session

    @property
    def active_session_count(self) -> int:
        """Get the number of active sessions."""
        return len(self._active_sessions)
