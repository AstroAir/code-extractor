"""General-purpose helper functions.

Provides commonly used utility functions for string processing,
collection manipulation, and control flow.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable, Iterator
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def slugify(text: str) -> str:
    """Convert text to a URL-safe slug.

    Args:
        text: The input text to slugify.

    Returns:
        Lowercase hyphenated string.

    Examples:
        >>> slugify("Hello World!")
        'hello-world'
        >>> slugify("  Spaced   Out  ")
        'spaced-out'
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[-\s]+", "-", text)
    return text.strip("-")


def truncate(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to a maximum length.

    Args:
        text: The text to truncate.
        max_length: Maximum allowed length.
        suffix: String to append when truncated.

    Returns:
        Truncated text with suffix if needed.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def chunk_list(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Split a list into chunks of specified size.

    Args:
        items: The list to split.
        chunk_size: Size of each chunk.

    Yields:
        Chunks of the original list.

    Raises:
        ValueError: If chunk_size is less than 1.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least 1")
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def flatten(nested: Iterable[Iterable[T]]) -> list[T]:
    """Flatten a nested iterable into a single list.

    Args:
        nested: Nested iterable to flatten.

    Returns:
        Flat list of all elements.
    """
    return [item for sublist in nested for item in sublist]


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[..., Any]:
    """Decorator for retrying failed function calls with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts.
        delay: Initial delay between retries in seconds.
        backoff: Multiplier applied to delay after each failure.
        exceptions: Tuple of exception types to catch.

    Returns:
        Decorator function.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            current_delay = delay
            last_exception: Exception | None = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        time.sleep(current_delay)
                        current_delay *= backoff

            raise last_exception  # type: ignore[misc]

        return wrapper

    return decorator


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary.
        override: Dictionary with override values.

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string.

    Args:
        num_bytes: Number of bytes.

    Returns:
        Formatted string like '1.5 MB'.
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024
    return f"{num_bytes:.1f} PB"
