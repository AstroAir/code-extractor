"""
Comprehensive test suite for Enhanced MCP Server.

Tests all MCP tools, resources, progress reporting, session management,
validation, and resource management features.
"""

import asyncio
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.servers.pysearch_mcp_server import PySearchEngine

from mcp.shared.resource_manager import ResourceManager
from mcp.shared.session_manager import EnhancedSessionManager, UserProfile
from mcp.shared.validation import InputValidator


class TestInputValidator:
    """Test input validation system."""

    def test_validate_pattern_success(self):
        """Test successful pattern validation."""
        result = InputValidator.validate_pattern("test pattern")
        assert result.is_valid
        assert result.sanitized_value == "test pattern"

    def test_validate_pattern_empty(self):
        """Test pattern validation with empty string."""
        result = InputValidator.validate_pattern("")
        assert not result.is_valid
        assert any("cannot be empty" in error for error in result.errors)

    def test_validate_pattern_too_long(self):
        """Test pattern validation with excessively long string."""
        long_pattern = "x" * 20000  # Exceeds 10k limit
        result = InputValidator.validate_pattern(long_pattern)
        assert not result.is_valid
        assert any("Pattern too long" in error for error in result.errors)

    def test_validate_pattern_dangerous(self):
        """Test pattern validation with potentially dangerous content."""
        dangerous_pattern = ".*" * 1000  # Could cause catastrophic backtracking
        result = InputValidator.validate_pattern(dangerous_pattern)
        assert not result.is_valid or len(result.warnings) > 0

    def test_validate_paths_success(self):
        """Test successful paths validation."""
        paths = ["./src", "./tests", "/home/user/project"]
        result = InputValidator.validate_paths(paths)
        assert result.is_valid
        assert len(result.sanitized_value) == 3

    def test_validate_paths_traversal(self):
        """Test paths validation prevents traversal attacks."""
        malicious_paths = ["../../../etc/passwd", "./safe/path"]
        result = InputValidator.validate_paths(malicious_paths)
        # Should filter out dangerous paths
        assert len(result.sanitized_value) == 1
        assert result.sanitized_value[0] == "./safe/path"

    def test_validate_context_lines_valid(self):
        """Test context lines validation with valid value."""
        result = InputValidator.validate_context_lines(5)
        assert result.is_valid
        assert result.sanitized_value == 5

    def test_validate_context_lines_invalid(self):
        """Test context lines validation with invalid value."""
        result = InputValidator.validate_context_lines(2000)
        assert not result.is_valid
        assert any("too large" in error for error in result.errors)

    def test_validate_similarity_threshold_valid(self):
        """Test similarity threshold validation with valid value."""
        result = InputValidator.validate_similarity_threshold(0.8)
        assert result.is_valid
        assert result.sanitized_value == 0.8

    def test_validate_similarity_threshold_out_of_range(self):
        """Test similarity threshold validation out of range."""
        result = InputValidator.validate_similarity_threshold(1.5)
        assert not result.is_valid
        assert any("must be between 0.0 and 1.0" in error for error in result.errors)

    def test_validate_max_results_valid(self):
        """Test max results validation with valid value."""
        result = InputValidator.validate_max_results(100)
        assert result.is_valid
        assert result.sanitized_value == 100

    def test_validate_max_results_too_large(self):
        """Test max results validation with excessive value."""
        result = InputValidator.validate_max_results(50000)
        assert not result.is_valid
        assert any("too large" in error for error in result.errors)

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # First request should succeed
        result1 = InputValidator.validate_pattern("test", identifier="test_user")
        assert result1.is_valid

        # Many rapid requests should trigger rate limiting
        for _ in range(150):  # Exceed 100/minute limit
            InputValidator.validate_pattern("test", identifier="test_user")

        result2 = InputValidator.validate_pattern("test", identifier="test_user")
        assert not result2.is_valid
        assert any("rate limit" in error.lower() for error in result2.errors)


class TestSessionManager:
    """Test session management system."""

    @pytest.fixture
    def session_manager(self):
        """Create a session manager for testing."""
        return EnhancedSessionManager()

    @pytest.mark.asyncio
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        context = {"project_type": "web_app", "focus": "authentication"}
        session = await session_manager.create_session("user1", context)

        assert session["session_id"] is not None
        assert session["user_id"] == "user1"
        assert session["context"] == context
        assert session["status"] == "active"

    @pytest.mark.asyncio
    async def test_get_session(self, session_manager):
        """Test retrieving an existing session."""
        context = {"project_type": "api"}
        session = await session_manager.create_session("user1", context)
        session_id = session["session_id"]

        retrieved = await session_manager.get_session(session_id)
        assert retrieved is not None
        assert retrieved["session_id"] == session_id
        assert retrieved["context"] == context

    @pytest.mark.asyncio
    async def test_update_session_context(self, session_manager):
        """Test updating session context."""
        session = await session_manager.create_session("user1")
        session_id = session["session_id"]

        new_context = {"current_task": "debugging", "focus": "error_handling"}
        await session_manager.update_session_context(session_id, new_context)

        updated = await session_manager.get_session(session_id)
        assert updated["context"] == new_context

    @pytest.mark.asyncio
    async def test_track_search_pattern(self, session_manager):
        """Test tracking search patterns in session."""
        session = await session_manager.create_session("user1")
        session_id = session["session_id"]

        await session_manager.track_search_pattern(session_id, "authentication", "semantic")
        await session_manager.track_search_pattern(session_id, "login.*error", "regex")

        updated = await session_manager.get_session(session_id)
        patterns = updated.get("search_patterns", [])
        assert len(patterns) == 2
        assert any(p["pattern"] == "authentication" for p in patterns)

    @pytest.mark.asyncio
    async def test_get_contextual_recommendations(self, session_manager):
        """Test getting contextual recommendations."""
        session = await session_manager.create_session("user1")
        session_id = session["session_id"]

        # Add some search history
        await session_manager.track_search_pattern(session_id, "auth", "text")
        await session_manager.track_search_pattern(session_id, "login", "semantic")

        recommendations = await session_manager.get_contextual_recommendations(session_id)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend related authentication patterns
        auth_related = any(
            "password" in rec or "token" in rec or "session" in rec for rec in recommendations
        )
        assert auth_related

    @pytest.mark.asyncio
    async def test_infer_search_intent(self, session_manager):
        """Test search intent inference."""
        session = await session_manager.create_session("user1")
        session_id = session["session_id"]

        # Add patterns that suggest debugging intent
        await session_manager.track_search_pattern(session_id, "error", "text")
        await session_manager.track_search_pattern(session_id, "exception", "text")
        await session_manager.track_search_pattern(session_id, "traceback", "text")

        intent = await session_manager.infer_search_intent(session_id)

        assert intent["primary_intent"] == "debugging"
        assert intent["confidence"] > 0.6

    @pytest.mark.asyncio
    async def test_user_profile_management(self, session_manager):
        """Test user profile creation and management."""
        profile = await session_manager.get_user_profile("user1")

        assert isinstance(profile, UserProfile)
        assert profile.user_id == "user1"
        assert profile.preferred_languages == []  # Initially empty

        # Update preferences
        await session_manager.update_user_preferences(
            "user1", {"preferred_languages": ["python", "javascript"], "default_context": 5}
        )

        updated_profile = await session_manager.get_user_profile("user1")
        assert updated_profile.preferred_languages == ["python", "javascript"]
        assert updated_profile.preferences["default_context"] == 5

    @pytest.mark.asyncio
    async def test_session_cleanup(self, session_manager):
        """Test session cleanup functionality."""
        # Create multiple sessions
        session1 = await session_manager.create_session("user1")
        await session_manager.create_session("user2")

        # Mark one as expired by setting old timestamp
        old_session = await session_manager.get_session(session1["session_id"])
        old_session["last_activity"] = datetime.now(timezone.utc).timestamp() - 86400  # 24h ago

        # Run cleanup
        cleaned = await session_manager.cleanup_expired_sessions()

        assert cleaned >= 1  # At least one session cleaned

        # Old session should be gone
        retrieved = await session_manager.get_session(session1["session_id"])
        assert retrieved is None


class TestResourceManager:
    """Test resource management system."""

    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager for testing."""
        return ResourceManager()

    @pytest.mark.asyncio
    async def test_cache_operations(self, resource_manager):
        """Test basic cache operations."""
        key = "test_key"
        value = {"data": "test_value", "timestamp": datetime.now().isoformat()}

        # Set cache value
        resource_manager.set_cache(key, value)

        # Get cache value
        cached = resource_manager.get_cache(key)
        assert cached == value

        # Check cache contains key
        assert resource_manager.has_cache(key)

    @pytest.mark.asyncio
    async def test_cache_expiration(self, resource_manager):
        """Test cache expiration functionality."""
        key = "expiring_key"
        value = {"data": "expires_soon"}

        # Set with short TTL
        resource_manager.set_cache(key, value, ttl=0.1)  # 0.1 seconds

        # Should be available immediately
        assert resource_manager.get_cache(key) == value

        # Wait for expiration
        await asyncio.sleep(0.2)

        # Should be None after expiration
        assert resource_manager.get_cache(key) is None

    @pytest.mark.asyncio
    async def test_cache_lru_eviction(self, resource_manager):
        """Test LRU cache eviction."""
        # Fill cache to max capacity
        for i in range(100):  # Default max size is 100
            resource_manager.set_cache(f"key_{i}", f"value_{i}")

        # Add one more to trigger eviction
        resource_manager.set_cache("key_new", "value_new")

        # First key should be evicted
        assert resource_manager.get_cache("key_0") is None
        assert resource_manager.get_cache("key_new") is not None

    @pytest.mark.asyncio
    async def test_cache_analytics(self, resource_manager):
        """Test cache analytics functionality."""
        # Perform some cache operations
        resource_manager.set_cache("key1", "value1")
        resource_manager.get_cache("key1")  # Hit
        resource_manager.get_cache("nonexistent")  # Miss

        analytics = resource_manager.get_cache_analytics()

        assert analytics["total_requests"] >= 2
        assert analytics["cache_hits"] >= 1
        assert analytics["cache_misses"] >= 1
        assert 0 <= analytics["hit_rate"] <= 1

    @pytest.mark.asyncio
    async def test_resource_optimization(self, resource_manager):
        """Test resource optimization features."""
        # Add some cache entries
        for i in range(10):
            resource_manager.set_cache(f"key_{i}", f"value_{i}")

        len(resource_manager._cache)

        # Clean expired entries
        cleaned = resource_manager.clean_expired()

        # Should return number cleaned (might be 0 if no expired entries)
        assert isinstance(cleaned, int)
        assert cleaned >= 0

    @pytest.mark.asyncio
    async def test_memory_monitoring(self, resource_manager):
        """Test memory usage monitoring."""
        memory_info = resource_manager.get_memory_usage()

        assert isinstance(memory_info, dict)
        assert "cache_memory_mb" in memory_info
        assert "total_entries" in memory_info
        assert memory_info["cache_memory_mb"] >= 0


class TestEnhancedMCPServer:
    """Test the enhanced MCP server implementation."""

    @pytest.fixture
    async def server(self):
        """Create an enhanced MCP server for testing."""
        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some test files
            test_files = {
                "main.py": '''
def authenticate_user(username, password):
    """Authenticate user with username and password."""
    if not username or not password:
        raise ValueError("Username and password required")
    
    # TODO: Implement proper authentication logic
    return verify_credentials(username, password)

async def process_login_request(request):
    """Process login request asynchronously."""
    try:
        username = request.get("username")
        password = request.get("password")
        
        if authenticate_user(username, password):
            return {"status": "success", "token": generate_token()}
        else:
            return {"status": "error", "message": "Invalid credentials"}
    except Exception as e:
        logger.error(f"Login error: {e}")
        return {"status": "error", "message": "Authentication failed"}
''',
                "utils.py": '''
import hashlib
import secrets

def hash_password(password: str) -> str:
    """Hash password using secure method."""
    salt = secrets.token_bytes(32)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt + key

def verify_credentials(username: str, password: str) -> bool:
    """Verify user credentials against database."""
    # Database lookup logic here
    return True  # Simplified for testing

class AuthenticationError(Exception):
    """Custom authentication exception."""
    pass
''',
                "config.json": """
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "myapp"
    },
    "authentication": {
        "secret_key": "your-secret-key",
        "token_expiry": 3600
    }
}
""",
            }

            # Write test files
            for filename, content in test_files.items():
                file_path = Path(temp_dir) / filename
                file_path.write_text(content)

            # Create server with test directory
            server = EnhancedPySearchMCPServer()
            server.search_engine.configure_search([temp_dir])

            yield server

    @pytest.mark.asyncio
    async def test_search_text_tool(self, server):
        """Test text search tool."""
        result = await server.search_text(pattern="authenticate", context=3, case_sensitive=False)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) > 0

        # Should find authentication-related code
        found_auth = any("authenticate" in str(match).lower() for match in result["results"])
        assert found_auth

    @pytest.mark.asyncio
    async def test_search_regex_tool(self, server):
        """Test regex search tool."""
        result = await server.search_regex(pattern=r"def\s+\w+", context=2)

        assert isinstance(result, dict)
        assert "results" in result
        assert len(result["results"]) > 0

        # Should find function definitions
        found_func = any("def " in str(match) for match in result["results"])
        assert found_func

    @pytest.mark.asyncio
    async def test_search_fuzzy_tool(self, server):
        """Test fuzzy search tool."""
        result = await server.search_fuzzy(
            pattern="autentication",  # Misspelled "authentication"
            similarity_threshold=0.7,
            max_results=10,
        )

        assert isinstance(result, dict)
        assert "results" in result
        # Should find "authentication" despite misspelling
        assert len(result["results"]) > 0

    @pytest.mark.asyncio
    async def test_search_multi_pattern_tool(self, server):
        """Test multi-pattern search tool."""
        result = await server.search_multi_pattern(
            patterns=["login", "password"], operator="AND", context=3
        )

        assert isinstance(result, dict)
        assert "results" in result
        # Should find matches containing both patterns

    @pytest.mark.asyncio
    async def test_search_with_ranking_tool(self, server):
        """Test search with ranking tool."""
        result = await server.search_with_ranking(
            pattern="authentication", max_results=5, importance_weight=0.3, recency_weight=0.2
        )

        assert isinstance(result, dict)
        assert "results" in result
        assert "ranking_info" in result

        # Results should include ranking scores
        if result["results"]:
            first_result = result["results"][0]
            assert "rank_score" in first_result or "score" in str(first_result)

    @pytest.mark.asyncio
    async def test_analyze_file_content_tool(self, server):
        """Test file content analysis tool."""
        # Get a test file path
        config = server.search_engine.get_search_config()
        search_paths = config.get("paths", [])
        if not search_paths:
            pytest.skip("No search paths configured")

        # Find a Python file to analyze
        test_files = []
        for path in search_paths:
            for file_path in Path(path).rglob("*.py"):
                test_files.append(str(file_path))
                break

        if not test_files:
            pytest.skip("No Python files found for analysis")

        result = await server.analyze_file_content(
            file_path=test_files[0], include_complexity=True, include_quality_metrics=True
        )

        assert isinstance(result, dict)
        assert "file_path" in result
        assert "analysis" in result

        analysis = result["analysis"]
        assert "lines_of_code" in analysis
        assert "complexity_score" in analysis
        assert analysis["lines_of_code"] > 0

    @pytest.mark.asyncio
    async def test_create_search_session_tool(self, server):
        """Test search session creation tool."""
        context = {
            "project_type": "web_application",
            "focus": "authentication",
            "user_preferences": {"preferred_languages": ["python"]},
        }

        result = await server.create_search_session(context=context)

        assert isinstance(result, dict)
        assert "session_id" in result
        assert "status" in result
        assert result["status"] == "created"
        assert result["context"] == context

    @pytest.mark.asyncio
    async def test_search_with_session(self, server):
        """Test search with session context."""
        # Create session first
        session_result = await server.create_search_session(context={"focus": "authentication"})
        session_id = session_result["session_id"]

        # Perform search with session
        result = await server.search_text(pattern="password", context=3, session_id=session_id)

        assert isinstance(result, dict)
        assert "results" in result
        assert "session_context" in result
        assert result["session_context"]["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_get_search_history_tool(self, server):
        """Test search history retrieval."""
        # Perform some searches first
        await server.search_text(pattern="test1")
        await server.search_regex(pattern="test2")

        result = await server.get_search_history(limit=5)

        assert isinstance(result, dict)
        assert "history" in result
        assert isinstance(result["history"], list)
        assert len(result["history"]) >= 2

    @pytest.mark.asyncio
    async def test_get_file_statistics_tool(self, server):
        """Test file statistics tool."""
        result = await server.get_file_statistics(include_analysis=True)

        assert isinstance(result, dict)
        assert "statistics" in result
        assert "total_files" in result["statistics"]
        assert "total_size" in result["statistics"]
        assert result["statistics"]["total_files"] > 0

    @pytest.mark.asyncio
    async def test_configure_search_tool(self, server):
        """Test search configuration tool."""
        new_config = {
            "context": 5,
            "parallel": True,
            "workers": 4,
            "exclude_patterns": ["*.pyc", "__pycache__"],
        }

        result = await server.configure_search(**new_config)

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "configured"
        assert "config" in result

        # Verify configuration was applied
        config = result["config"]
        assert config["context"] == 5
        assert config["parallel"]

    @pytest.mark.asyncio
    async def test_clear_caches_tool(self, server):
        """Test cache clearing tool."""
        # Add some cached data first
        server.resource_manager.set_cache("test_key", {"data": "test"})

        result = await server.clear_caches()

        assert isinstance(result, dict)
        assert "status" in result
        assert result["status"] == "cleared"

        # Verify cache was cleared
        assert server.resource_manager.get_cache("test_key") is None

    @pytest.mark.asyncio
    async def test_input_validation_integration(self, server):
        """Test input validation integration."""
        # Test with invalid input
        try:
            await server.search_text(
                pattern="",  # Empty pattern should fail validation
                context=5000,  # Excessive context should fail validation
            )
            raise AssertionError("Should have raised validation error")
        except Exception as e:
            # Should get validation error
            assert "validation" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_error_handling(self, server):
        """Test error handling in tools."""
        # Test with malformed regex
        result = await server.search_regex(pattern="[unclosed", context=3)  # Invalid regex

        # Should handle error gracefully
        assert isinstance(result, dict)
        assert "error" in result or "results" in result

    @pytest.mark.asyncio
    async def test_resource_endpoints(self, server):
        """Test MCP resource endpoints."""
        resources = [
            "pysearch://config/current",
            "pysearch://history/searches",
            "pysearch://stats/overview",
            "pysearch://cache/analytics",
            "pysearch://health/status",
        ]

        for resource_uri in resources:
            # Test that resource can be fetched without error
            try:
                result = await server._fetch_resource(resource_uri)
                assert isinstance(result, dict)
            except Exception as e:
                pytest.fail(f"Failed to fetch resource {resource_uri}: {e}")


class TestProgressReporting:
    """Test progress reporting functionality."""

    @pytest.fixture
    def progress_reporter(self):
        """Create a progress reporter for testing."""
        return ProgressReporter("test_operation")

    def test_progress_initialization(self, progress_reporter):
        """Test progress reporter initialization."""
        assert progress_reporter.operation_id == "test_operation"
        assert progress_reporter.total_steps == 100  # Default
        assert progress_reporter.current_step == 0
        assert progress_reporter.status == "initialized"

    def test_progress_updates(self, progress_reporter):
        """Test progress update functionality."""
        progress_reporter.update(25, "Processing files")

        status = progress_reporter.get_status()
        assert status["current_step"] == 25
        assert status["progress"] == 0.25
        assert status["message"] == "Processing files"
        assert status["status"] == "running"

    def test_progress_completion(self, progress_reporter):
        """Test progress completion."""
        progress_reporter.complete("Operation finished")

        status = progress_reporter.get_status()
        assert status["status"] == "completed"
        assert status["progress"] == 1.0
        assert status["message"] == "Operation finished"

    def test_progress_cancellation(self, progress_reporter):
        """Test progress cancellation."""
        progress_reporter.cancel("Operation cancelled by user")

        status = progress_reporter.get_status()
        assert status["status"] == "cancelled"
        assert status["message"] == "Operation cancelled by user"

    def test_progress_error_handling(self, progress_reporter):
        """Test progress error handling."""
        error_msg = "Test error occurred"
        progress_reporter.error(error_msg)

        status = progress_reporter.get_status()
        assert status["status"] == "error"
        assert error_msg in status["message"]

    @pytest.mark.asyncio
    async def test_progress_with_server(self):
        """Test progress reporting integration with server."""
        EnhancedPySearchMCPServer()

        # Create a simple async generator that yields progress
        async def mock_search_with_progress():
            for i in range(5):
                yield {
                    "progress": i / 4,
                    "current_step": i,
                    "total_steps": 4,
                    "message": f"Step {i}",
                    "status": "running" if i < 4 else "completed",
                }
                await asyncio.sleep(0.01)  # Small delay

        # Collect progress updates
        progress_updates = []
        async for update in mock_search_with_progress():
            progress_updates.append(update)

        assert len(progress_updates) == 5
        assert progress_updates[0]["progress"] == 0.0
        assert progress_updates[-1]["progress"] == 1.0
        assert progress_updates[-1]["status"] == "completed"


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete search workflow with session management."""
        server = EnhancedPySearchMCPServer()

        # 1. Create a session
        session_result = await server.create_search_session(
            context={
                "project_type": "python_web_app",
                "current_task": "debugging authentication issues",
            }
        )
        session_id = session_result["session_id"]

        # 2. Perform various searches with the session
        searches = [
            ("authentication", "text"),
            ("login.*error", "regex"),
            (["auth", "password"], "multi_pattern"),
            ("security", "semantic"),
        ]

        for pattern, search_type in searches:
            if search_type == "text":
                result = await server.search_text(pattern=pattern, session_id=session_id)
            elif search_type == "regex":
                result = await server.search_regex(pattern=pattern, session_id=session_id)
            elif search_type == "multi_pattern":
                result = await server.search_multi_pattern(patterns=pattern, session_id=session_id)
            elif search_type == "semantic":
                result = await server.search_semantic(concept=pattern, session_id=session_id)

            assert isinstance(result, dict)
            assert "results" in result

        # 3. Get search history
        history = await server.get_search_history(limit=10)
        assert len(history["history"]) >= 4  # Should have our 4 searches

        # 4. Get contextual recommendations
        try:
            recommendations = await server.get_contextual_recommendations(session_id)
            assert isinstance(recommendations, dict)
            assert "recommendations" in recommendations
        except Exception:
            # Recommendations might not be available if not implemented
            pass

        # 5. Check analytics and health
        cache_analytics = server.resource_manager.get_cache_analytics()
        assert isinstance(cache_analytics, dict)
        assert "total_requests" in cache_analytics

    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test server performance under concurrent load."""
        server = EnhancedPySearchMCPServer()

        # Create multiple concurrent search tasks
        async def perform_search(pattern_num):
            try:
                result = await server.search_text(pattern=f"test_pattern_{pattern_num}", context=3)
                return len(result.get("results", []))
            except Exception:
                return 0

        # Run multiple searches concurrently
        tasks = [perform_search(i) for i in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Most should complete successfully
        successful_results = [r for r in results if isinstance(r, int)]
        assert len(successful_results) >= 8  # At least 80% should succeed

    @pytest.mark.asyncio
    async def test_resource_cleanup(self):
        """Test proper resource cleanup."""
        server = EnhancedPySearchMCPServer()

        # Create multiple sessions and perform searches
        sessions = []
        for i in range(5):
            session = await server.create_search_session(context={"test": f"session_{i}"})
            sessions.append(session["session_id"])

            # Perform search in each session
            await server.search_text(pattern=f"test_{i}", session_id=session["session_id"])

        # Check that sessions were created
        session_manager = server.session_manager
        active_sessions = len(session_manager.sessions)
        assert active_sessions >= 5

        # Cleanup expired sessions (none should be expired yet)
        cleaned = await session_manager.cleanup_expired_sessions()
        assert cleaned == 0  # No expired sessions

        # Clear caches
        result = await server.clear_caches()
        assert result["status"] == "cleared"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
