"""Request validation utilities.

Provides validation functions and decorators for sanitizing
and verifying API request data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ValidationError:
    """Represents a single validation error."""

    field: str
    message: str
    code: str = "invalid"


class ValidationResult:
    """Collects validation errors for a request."""

    def __init__(self) -> None:
        self.errors: list[ValidationError] = []

    def add_error(self, field: str, message: str, code: str = "invalid") -> None:
        """Add a validation error."""
        self.errors.append(ValidationError(field=field, message=message, code=code))

    @property
    def is_valid(self) -> bool:
        """Check if validation passed with no errors."""
        return len(self.errors) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation errors."""
        return {
            "valid": self.is_valid,
            "errors": [
                {"field": e.field, "message": e.message, "code": e.code} for e in self.errors
            ],
        }


def validate_username(username: str) -> ValidationResult:
    """Validate a username string.

    Rules:
        - Must be 3-30 characters.
        - Only alphanumeric characters and underscores.
        - Must start with a letter.
    """
    result = ValidationResult()
    if not username:
        result.add_error("username", "Username is required", "required")
        return result

    if len(username) < 3:
        result.add_error("username", "Username must be at least 3 characters", "min_length")
    if len(username) > 30:
        result.add_error("username", "Username must be at most 30 characters", "max_length")
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9_]*$", username):
        result.add_error("username", "Invalid username format", "format")

    return result


def validate_email(email: str) -> ValidationResult:
    """Validate an email address."""
    result = ValidationResult()
    if not email:
        result.add_error("email", "Email is required", "required")
        return result

    pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(pattern, email):
        result.add_error("email", "Invalid email format", "format")

    return result


def validate_password(password: str) -> ValidationResult:
    """Validate a password string.

    Rules:
        - Must be at least 8 characters.
        - Must contain uppercase and lowercase letters.
        - Must contain at least one digit.
    """
    result = ValidationResult()
    if not password:
        result.add_error("password", "Password is required", "required")
        return result

    if len(password) < 8:
        result.add_error("password", "Password must be at least 8 characters", "min_length")
    if not re.search(r"[A-Z]", password):
        result.add_error("password", "Password must contain an uppercase letter", "complexity")
    if not re.search(r"[a-z]", password):
        result.add_error("password", "Password must contain a lowercase letter", "complexity")
    if not re.search(r"\d", password):
        result.add_error("password", "Password must contain a digit", "complexity")

    return result


def validate_post_data(title: str, content: str) -> ValidationResult:
    """Validate blog post data."""
    result = ValidationResult()

    if not title or len(title.strip()) < 5:
        result.add_error("title", "Title must be at least 5 characters", "min_length")
    if title and len(title) > 200:
        result.add_error("title", "Title must be at most 200 characters", "max_length")

    if not content or len(content.strip()) < 10:
        result.add_error("content", "Content must be at least 10 characters", "min_length")

    return result


def sanitize_html(text: str) -> str:
    """Remove potentially dangerous HTML tags from text.

    WARNING: This is a simplified sanitizer for demonstration.
    Use a proper library like bleach in production.
    """
    # FIXME: Replace with proper HTML sanitization library
    dangerous_tags = re.compile(
        r"<(script|iframe|object|embed|form)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
    )
    return dangerous_tags.sub("", text)
