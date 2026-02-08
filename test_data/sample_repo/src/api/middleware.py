"""HTTP middleware implementations.

Provides logging, rate limiting, and error handling middleware
using async context manager patterns.
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Callable


@dataclass
class RequestContext:
    """Context object passed through middleware chain."""

    method: str = "GET"
    path: str = "/"
    headers: dict[str, str] = field(default_factory=dict)
    client_ip: str = "127.0.0.1"
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_ms(self) -> float:
        """Calculate request processing time in milliseconds."""
        return (time.time() - self.start_time) * 1000


class LoggingMiddleware:
    """Middleware that logs request/response information."""

    def __init__(self, log_headers: bool = False) -> None:
        self.log_headers = log_headers
        self._request_count = 0

    @asynccontextmanager
    async def __call__(
        self, context: RequestContext
    ) -> AsyncGenerator[RequestContext, None]:
        """Process request with logging.

        Args:
            context: The request context.

        Yields:
            The request context for downstream processing.
        """
        self._request_count += 1
        request_id = self._request_count

        # Log request start
        print(f"[{request_id}] {context.method} {context.path} - Start")

        try:
            yield context
        except Exception as e:
            print(f"[{request_id}] ERROR: {type(e).__name__}: {e}")
            raise
        finally:
            elapsed = context.elapsed_ms
            print(f"[{request_id}] {context.method} {context.path} - {elapsed:.1f}ms")

    @property
    def total_requests(self) -> int:
        """Get total number of processed requests."""
        return self._request_count


class RateLimitMiddleware:
    """Middleware that enforces per-client rate limits."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _cleanup_old_requests(self, client_ip: str) -> None:
        """Remove expired request timestamps."""
        cutoff = time.time() - self.window_seconds
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if t > cutoff
        ]

    def is_rate_limited(self, client_ip: str) -> bool:
        """Check if a client has exceeded the rate limit.

        Args:
            client_ip: The client's IP address.

        Returns:
            True if the client is rate limited.
        """
        self._cleanup_old_requests(client_ip)
        return len(self._requests[client_ip]) >= self.max_requests

    @asynccontextmanager
    async def __call__(
        self, context: RequestContext
    ) -> AsyncGenerator[RequestContext, None]:
        """Process request with rate limit enforcement.

        Args:
            context: The request context.

        Yields:
            The request context.

        Raises:
            RuntimeError: If the client has exceeded the rate limit.
        """
        if self.is_rate_limited(context.client_ip):
            raise RuntimeError(
                f"Rate limit exceeded for {context.client_ip}: "
                f"{self.max_requests} requests per {self.window_seconds}s"
            )

        self._requests[context.client_ip].append(time.time())
        yield context


class ErrorHandlerMiddleware:
    """Middleware that catches and formats errors."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self._error_count = 0

    @asynccontextmanager
    async def __call__(
        self, context: RequestContext
    ) -> AsyncGenerator[RequestContext, None]:
        """Process request with error handling.

        Args:
            context: The request context.

        Yields:
            The request context.
        """
        try:
            yield context
        except PermissionError:
            self._error_count += 1
            raise
        except ValueError:
            self._error_count += 1
            raise
        except Exception as e:
            self._error_count += 1
            if self.debug:
                raise
            raise RuntimeError(f"Internal server error: {type(e).__name__}") from e

    @property
    def error_count(self) -> int:
        """Get total number of errors handled."""
        return self._error_count
