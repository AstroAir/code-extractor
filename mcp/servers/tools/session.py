"""Session Management Tools — create and query search sessions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastmcp import FastMCP

    from ..engine import PySearchEngine


def register_session_tools(
    mcp: FastMCP,
    engine: PySearchEngine,
    _validate: Callable[..., dict[str, Any]],
) -> None:
    """Register session management tools on the MCP server."""
    from fastmcp.exceptions import ToolError

    @mcp.tool
    def create_session(
        user_id: str | None = None,
        priority: str = "normal",
    ) -> dict[str, Any]:
        """
        Create a new search session for context-aware search tracking.

        Sessions track search patterns, infer user intent, and provide
        contextual recommendations across multiple searches.

        Args:
            user_id: Optional user identifier for personalized experience
            priority: Session priority — "low", "normal", "high", "critical" (default: "normal")

        Returns:
            Session info including session_id, creation time, and initial state
        """
        from ...shared.session_manager import SessionPriority

        try:
            prio = SessionPriority(priority)
        except ValueError:
            prio = SessionPriority.NORMAL

        try:
            session = engine.session_manager.create_session(user_id=user_id, priority=prio)
            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "priority": session.priority.value,
                "status": "created",
            }
        except Exception as e:
            raise ToolError(f"Session creation failed: {e}") from e

    @mcp.tool
    def get_session_info(
        session_id: str,
    ) -> dict[str, Any]:
        """
        Get detailed information about a search session.

        Returns session state, search history within the session, inferred intent,
        contextual recommendations, and analytics.

        Args:
            session_id: The session ID to query

        Returns:
            Session details including context, intent, recommendations, and analytics
        """
        _validate(session_id=session_id)
        try:
            session = engine.session_manager.get_session(session_id)
            if session is None:
                raise ToolError(f"Session not found: {session_id}")

            recommendations = engine.session_manager.get_contextual_recommendations(session_id)
            context = session.get_current_context()
            suggestions = session.get_contextual_suggestions()

            return {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "created_at": session.created_at.isoformat(),
                "last_accessed": session.last_accessed.isoformat(),
                "priority": session.priority.value,
                "intent": session.intent.value,
                "total_searches": session.total_searches,
                "successful_searches": session.successful_searches,
                "avg_search_time": session.avg_search_time,
                "search_focus": session.search_focus,
                "recent_patterns": session.recent_patterns[-10:],
                "patterns_discovered_count": len(session.patterns_discovered),
                "current_files_count": len(session.current_files),
                "context": context,
                "suggestions": suggestions,
                "recommendations": recommendations,
            }
        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to get session info: {e}") from e
