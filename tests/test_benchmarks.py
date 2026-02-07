"""
Performance benchmark tests for Enhanced MCP Server.

Measures performance across different scenarios including concurrent usage,
large codebases, and resource utilization.
"""

import asyncio
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.servers.enhanced_fastmcp_server import EnhancedPySearchMCPServer


class BenchmarkRunner:
    """Helper class for running benchmark tests."""

    def __init__(self):
        self.results = {}

    async def time_async_operation(self, operation, name, *args, **kwargs):
        """Time an async operation and store results."""
        start_time = time.perf_counter()
        try:
            result = await operation(*args, **kwargs)
            end_time = time.perf_counter()
            duration = end_time - start_time

            if name not in self.results:
                self.results[name] = []
            self.results[name].append(duration)

            return result, duration
        except Exception:
            end_time = time.perf_counter()
            duration = end_time - start_time

            if f"{name}_errors" not in self.results:
                self.results[f"{name}_errors"] = []
            self.results[f"{name}_errors"].append(duration)

            return None, duration

    def get_stats(self, operation_name):
        """Get statistics for an operation."""
        if operation_name not in self.results:
            return None

        times = self.results[operation_name]
        if not times:
            return None

        return {
            "count": len(times),
            "min": min(times),
            "max": max(times),
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "total": sum(times),
        }


@pytest.fixture
def benchmark_runner():
    """Create a benchmark runner."""
    return BenchmarkRunner()


@pytest.fixture
async def large_codebase_server():
    """Create a server with a large synthetic codebase."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a synthetic large codebase
        files_created = 0

        # Create directory structure
        directories = [
            "src",
            "src/api",
            "src/models",
            "src/services",
            "src/utils",
            "tests",
            "tests/unit",
            "tests/integration",
            "tests/fixtures",
            "docs",
            "config",
            "scripts",
            "migrations",
        ]

        for dir_name in directories:
            (Path(temp_dir) / dir_name).mkdir(parents=True, exist_ok=True)

        # Generate files with realistic code patterns
        code_templates = {
            "service": '''
class {class_name}Service:
    """Service class for {domain} operations."""
    
    def __init__(self, db_connection, cache_client, logger):
        self.db = db_connection
        self.cache = cache_client
        self.logger = logger
        
    async def create_{entity}(self, data: dict) -> dict:
        """Create a new {entity}."""
        try:
            # Validation
            if not data.get("name"):
                raise ValueError("Name is required")
                
            # Check cache first
            cache_key = f"{entity}_{{data['name']}}"
            cached = await self.cache.get(cache_key)
            if cached:
                return cached
                
            # Create in database
            result = await self.db.create("{entity}", data)
            
            # Cache result
            await self.cache.set(cache_key, result, ttl=3600)
            
            self.logger.info(f"Created {entity}: {{result['id']}}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating {entity}: {{e}}")
            raise
    
    async def get_{entity}(self, {entity}_id: str) -> dict:
        """Get {entity} by ID."""
        cache_key = f"{entity}_{{entity_id}}"
        cached = await self.cache.get(cache_key)
        if cached:
            return cached
            
        result = await self.db.get("{entity}", {entity}_id)
        if result:
            await self.cache.set(cache_key, result, ttl=3600)
        return result
    
    async def update_{entity}(self, {entity}_id: str, data: dict) -> dict:
        """Update an existing {entity}."""
        try:
            result = await self.db.update("{entity}", {entity}_id, data)
            
            # Invalidate cache
            cache_key = f"{entity}_{{entity_id}}"
            await self.cache.delete(cache_key)
            
            self.logger.info(f"Updated {entity}: {{{entity}_id}}")
            return result
        except Exception as e:
            self.logger.error(f"Error updating {entity}: {{e}}")
            raise
    
    async def delete_{entity}(self, {entity}_id: str) -> bool:
        """Delete a {entity}."""
        try:
            await self.db.delete("{entity}", {entity}_id)
            
            # Invalidate cache
            cache_key = f"{entity}_{{entity_id}}"
            await self.cache.delete(cache_key)
            
            self.logger.info(f"Deleted {entity}: {{{entity}_id}}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting {entity}: {{e}}")
            return False
''',
            "model": '''
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class {class_name}:
    """Data model for {domain}."""
    
    id: Optional[str] = None
    name: str = ""
    description: Optional[str] = None
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {{
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }}
    
    @classmethod
    def from_dict(cls, data: dict) -> '{class_name}':
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            name=data.get("name", ""),
            description=data.get("description"),
            status=data.get("status", "active"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            updated_at=datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
        )
    
    def validate(self) -> bool:
        """Validate the model data."""
        if not self.name:
            return False
        if self.status not in ["active", "inactive", "pending"]:
            return False
        return True
    
    def update_timestamp(self):
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
''',
            "api": '''
from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional
from ..models.{domain} import {class_name}
from ..services.{domain}_service import {class_name}Service

router = APIRouter(prefix="/{domain}", tags=["{domain}"])

@router.get("/")
async def list_{domain}(
    skip: int = 0,
    limit: int = 100,
    service: {class_name}Service = Depends()
) -> List[dict]:
    """List all {domain} items."""
    try:
        items = await service.list_{domain}(skip=skip, limit=limit)
        return [item.to_dict() for item in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{{item_id}}")
async def get_{domain}_by_id(
    item_id: str,
    service: {class_name}Service = Depends()
) -> dict:
    """Get {domain} item by ID."""
    item = await service.get_{domain}(item_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"{class_name} not found")
    return item.to_dict()

@router.post("/")
async def create_{domain}(
    item_data: dict,
    service: {class_name}Service = Depends()
) -> dict:
    """Create new {domain} item."""
    try:
        # Validate input
        {domain}_item = {class_name}.from_dict(item_data)
        if not {domain}_item.validate():
            raise HTTPException(status_code=400, detail="Invalid {domain} data")
            
        result = await service.create_{domain}(item_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{{item_id}}")
async def update_{domain}(
    item_id: str,
    item_data: dict,
    service: {class_name}Service = Depends()
) -> dict:
    """Update {domain} item."""
    try:
        # Validate input
        {domain}_item = {class_name}.from_dict(item_data)
        if not {domain}_item.validate():
            raise HTTPException(status_code=400, detail="Invalid {domain} data")
            
        result = await service.update_{domain}(item_id, item_data)
        if not result:
            raise HTTPException(status_code=404, detail=f"{class_name} not found")
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{{item_id}}")
async def delete_{domain}(
    item_id: str,
    service: {class_name}Service = Depends()
) -> dict:
    """Delete {domain} item."""
    success = await service.delete_{domain}(item_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"{class_name} not found")
    return {{"message": f"{class_name} deleted successfully"}}
''',
            "test": '''
import pytest
from unittest.mock import AsyncMock, MagicMock
from {module_path} import {class_name}

class Test{class_name}:
    """Test suite for {class_name}."""
    
    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        return {{
            "db": AsyncMock(),
            "cache": AsyncMock(),
            "logger": MagicMock()
        }}
    
    @pytest.fixture
    def {domain}_service(self, mock_dependencies):
        """Create service instance with mocked dependencies."""
        return {class_name}Service(
            db_connection=mock_dependencies["db"],
            cache_client=mock_dependencies["cache"],
            logger=mock_dependencies["logger"]
        )
    
    @pytest.mark.asyncio
    async def test_create_{domain}_success(self, {domain}_service, mock_dependencies):
        """Test successful {domain} creation."""
        # Arrange
        test_data = {{
            "name": "Test {class_name}",
            "description": "Test description"
        }}
        expected_result = {{**test_data, "id": "test-id"}}
        
        mock_dependencies["cache"].get.return_value = None
        mock_dependencies["db"].create.return_value = expected_result
        
        # Act
        result = await {domain}_service.create_{domain}(test_data)
        
        # Assert
        assert result == expected_result
        mock_dependencies["db"].create.assert_called_once_with("{domain}", test_data)
        mock_dependencies["cache"].set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_{domain}_validation_error(self, {domain}_service):
        """Test {domain} creation with validation error."""
        # Arrange
        invalid_data = {{}}  # Missing required name
        
        # Act & Assert
        with pytest.raises(ValueError, match="Name is required"):
            await {domain}_service.create_{domain}(invalid_data)
    
    @pytest.mark.asyncio
    async def test_get_{domain}_from_cache(self, {domain}_service, mock_dependencies):
        """Test getting {domain} from cache."""
        # Arrange
        {domain}_id = "test-id"
        cached_result = {{"id": {domain}_id, "name": "Cached {class_name}"}}
        mock_dependencies["cache"].get.return_value = cached_result
        
        # Act
        result = await {domain}_service.get_{domain}({domain}_id)
        
        # Assert
        assert result == cached_result
        mock_dependencies["db"].get.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_get_{domain}_from_database(self, {domain}_service, mock_dependencies):
        """Test getting {domain} from database when not cached."""
        # Arrange
        {domain}_id = "test-id"
        db_result = {{"id": {domain}_id, "name": "DB {class_name}"}}
        
        mock_dependencies["cache"].get.return_value = None
        mock_dependencies["db"].get.return_value = db_result
        
        # Act
        result = await {domain}_service.get_{domain}({domain}_id)
        
        # Assert
        assert result == db_result
        mock_dependencies["db"].get.assert_called_once_with("{domain}", {domain}_id)
        mock_dependencies["cache"].set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_{domain}_success(self, {domain}_service, mock_dependencies):
        """Test successful {domain} update."""
        # Arrange
        {domain}_id = "test-id"
        update_data = {{"name": "Updated {class_name}"}}
        expected_result = {{**update_data, "id": {domain}_id}}
        
        mock_dependencies["db"].update.return_value = expected_result
        
        # Act
        result = await {domain}_service.update_{domain}({domain}_id, update_data)
        
        # Assert
        assert result == expected_result
        mock_dependencies["db"].update.assert_called_once_with("{domain}", {domain}_id, update_data)
        mock_dependencies["cache"].delete.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_{domain}_success(self, {domain}_service, mock_dependencies):
        """Test successful {domain} deletion."""
        # Arrange
        {domain}_id = "test-id"
        
        # Act
        result = await {domain}_service.delete_{domain}({domain}_id)
        
        # Assert
        assert result is True
        mock_dependencies["db"].delete.assert_called_once_with("{domain}", {domain}_id)
        mock_dependencies["cache"].delete.assert_called_once()
''',
        }

        # Create files for different domains
        domains = ["users", "products", "orders", "payments", "notifications", "reports"]

        for domain in domains:
            class_name = domain.capitalize()[:-1]  # Remove 's' and capitalize

            # Create service file
            service_content = code_templates["service"].format(
                class_name=class_name, domain=domain, entity=domain[:-1]  # Remove 's'
            )
            service_file = Path(temp_dir) / "src" / "services" / f"{domain}_service.py"
            service_file.write_text(service_content)
            files_created += 1

            # Create model file
            model_content = code_templates["model"].format(class_name=class_name, domain=domain)
            model_file = Path(temp_dir) / "src" / "models" / f"{domain[:-1]}.py"
            model_file.write_text(model_content)
            files_created += 1

            # Create API file
            api_content = code_templates["api"].format(
                class_name=class_name, domain=domain[:-1]  # Remove 's'
            )
            api_file = Path(temp_dir) / "src" / "api" / f"{domain}_api.py"
            api_file.write_text(api_content)
            files_created += 1

            # Create test files
            test_content = code_templates["test"].format(
                class_name=class_name,
                domain=domain[:-1],
                module_path=f"src.services.{domain}_service",
            )
            test_file = Path(temp_dir) / "tests" / "unit" / f"test_{domain}_service.py"
            test_file.write_text(test_content)
            files_created += 1

        # Create additional utility and configuration files
        additional_files = {
            "src/utils/database.py": '''
import asyncpg
from typing import Optional

class DatabaseManager:
    """Database connection manager."""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None
    
    async def connect(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(self.connection_string)
    
    async def disconnect(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def execute(self, query: str, *args):
        """Execute query."""
        async with self.pool.acquire() as connection:
            return await connection.execute(query, *args)
    
    async def fetch(self, query: str, *args):
        """Fetch query results."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)
    
    async def fetchrow(self, query: str, *args):
        """Fetch single row."""
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
''',
            "src/utils/cache.py": '''
import redis.asyncio as redis
from typing import Any, Optional
import json

class CacheManager:
    """Redis cache manager."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        self.client = redis.from_url(self.redis_url)
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.client:
            await self.client.close()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.client:
            return None
        
        value = await self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        if not self.client:
            return
        
        await self.client.setex(key, ttl, json.dumps(value))
    
    async def delete(self, key: str):
        """Delete key from cache."""
        if not self.client:
            return
        
        await self.client.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.client:
            return False
        
        return await self.client.exists(key)
''',
            "config/settings.py": '''
import os
from typing import Optional

class Settings:
    """Application settings."""
    
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql://localhost/testdb")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # API settings
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8000"))
        
        # Authentication settings
        self.jwt_algorithm = "HS256"
        self.jwt_expiration = 3600  # 1 hour
        
        # Cache settings
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))
        self.cache_prefix = os.getenv("CACHE_PREFIX", "app")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database_url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL.""" 
        return self.redis_url
    
    def is_debug(self) -> bool:
        """Check if debug mode is enabled."""
        return self.debug

settings = Settings()
''',
            "src/main.py": '''
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import users_api, products_api, orders_api, payments_api, notifications_api, reports_api
from .utils.database import DatabaseManager
from .utils.cache import CacheManager
from config.settings import settings
import logging

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Benchmark API",
    description="API for benchmark testing",
    version="1.0.0",
    debug=settings.is_debug()
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database and cache managers
db_manager = DatabaseManager(settings.get_database_url())
cache_manager = CacheManager(settings.get_redis_url())

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup."""
    await db_manager.connect()
    await cache_manager.connect()
    logger.info("Application started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown."""
    await db_manager.disconnect()
    await cache_manager.disconnect()
    logger.info("Application shut down successfully")

# Include API routers
app.include_router(users_api.router)
app.include_router(products_api.router)
app.include_router(orders_api.router)
app.include_router(payments_api.router)
app.include_router(notifications_api.router)
app.include_router(reports_api.router)

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Benchmark API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": "1.0.0"
    }
''',
        }

        for file_path, content in additional_files.items():
            file_full_path = Path(temp_dir) / file_path
            file_full_path.parent.mkdir(parents=True, exist_ok=True)
            file_full_path.write_text(content)
            files_created += 1

        # Create server with the synthetic codebase
        server = EnhancedPySearchMCPServer()
        server.search_engine.configure_search([temp_dir])

        print(f"Created synthetic codebase with {files_created} files in {temp_dir}")

        yield server


class TestSearchPerformance:
    """Test search performance across different scenarios."""

    @pytest.mark.asyncio
    async def test_text_search_performance(self, large_codebase_server, benchmark_runner):
        """Test text search performance."""
        server = large_codebase_server

        # Test various search patterns
        patterns = [
            "class",
            "async def",
            "import",
            "Exception",
            "return",
            "authentication",
            "database",
            "cache",
        ]

        for pattern in patterns:
            result, duration = await benchmark_runner.time_async_operation(
                server.search_text, "text_search", pattern=pattern, context=3
            )

            assert result is not None
            print(f"Text search '{pattern}': {duration:.4f}s")

        # Get statistics
        stats = benchmark_runner.get_stats("text_search")
        print(f"Text search stats: {stats}")

        # Performance assertions
        assert stats["mean"] < 2.0  # Average should be under 2 seconds
        assert stats["max"] < 5.0  # Max should be under 5 seconds

    @pytest.mark.asyncio
    async def test_regex_search_performance(self, large_codebase_server, benchmark_runner):
        """Test regex search performance."""
        server = large_codebase_server

        # Test regex patterns
        patterns = [
            r"class\s+\w+",
            r"def\s+\w+\(",
            r"import\s+\w+",
            r"async\s+def\s+\w+",
            r"@\w+",
            r"raise\s+\w+Error",
            r"return\s+\w+",
            r"await\s+\w+",
        ]

        for pattern in patterns:
            result, duration = await benchmark_runner.time_async_operation(
                server.search_regex, "regex_search", pattern=pattern, context=3
            )

            print(f"Regex search '{pattern}': {duration:.4f}s")

        stats = benchmark_runner.get_stats("regex_search")
        print(f"Regex search stats: {stats}")

        # Regex should be slightly slower than text search
        assert stats["mean"] < 3.0
        assert stats["max"] < 7.0

    @pytest.mark.asyncio
    async def test_fuzzy_search_performance(self, large_codebase_server, benchmark_runner):
        """Test fuzzy search performance."""
        server = large_codebase_server

        # Test fuzzy patterns (with intentional typos)
        patterns = [
            ("databse", "database"),
            ("authentcation", "authentication"),
            ("conection", "connection"),
            ("excpetion", "exception"),
            ("sevice", "service"),
            ("respose", "response"),
        ]

        for typo_pattern, _ in patterns:
            result, duration = await benchmark_runner.time_async_operation(
                server.search_fuzzy,
                "fuzzy_search",
                pattern=typo_pattern,
                similarity_threshold=0.7,
                max_results=50,
            )

            print(f"Fuzzy search '{typo_pattern}': {duration:.4f}s")

        stats = benchmark_runner.get_stats("fuzzy_search")
        print(f"Fuzzy search stats: {stats}")

        # Fuzzy search should be slower due to similarity calculations
        assert stats["mean"] < 5.0
        assert stats["max"] < 10.0

    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, large_codebase_server, benchmark_runner):
        """Test performance under concurrent load."""
        server = large_codebase_server

        async def perform_concurrent_search(search_id):
            """Perform a search operation."""
            patterns = ["class", "def", "import", "async", "return"]
            pattern = patterns[search_id % len(patterns)]

            start_time = time.perf_counter()
            try:
                result = await server.search_text(pattern=pattern, context=3)
                end_time = time.perf_counter()
                return end_time - start_time, len(result.get("results", []))
            except Exception:
                end_time = time.perf_counter()
                return end_time - start_time, 0

        # Run concurrent searches
        concurrent_count = 10
        tasks = [perform_concurrent_search(i) for i in range(concurrent_count)]

        start_time = time.perf_counter()
        results = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        total_time = end_time - start_time
        durations = [r[0] for r in results]
        result_counts = [r[1] for r in results]

        print(f"Concurrent searches completed in: {total_time:.4f}s")
        print(f"Individual search times: {durations}")
        print(f"Result counts: {result_counts}")

        # Performance assertions
        assert total_time < 15.0  # All searches should complete within 15 seconds
        assert statistics.mean(durations) < 5.0  # Average individual time under 5 seconds
        assert sum(result_counts) > 0  # Should find some results

    @pytest.mark.asyncio
    async def test_memory_usage_performance(self, large_codebase_server):
        """Test memory usage during operations."""
        server = large_codebase_server

        # Get initial memory usage
        initial_memory = server.resource_manager.get_memory_usage()
        print(f"Initial memory usage: {initial_memory}")

        # Perform multiple operations
        operations = [
            ("text_search", server.search_text, {"pattern": "class", "context": 5}),
            ("regex_search", server.search_regex, {"pattern": r"def\s+\w+", "context": 5}),
            (
                "fuzzy_search",
                server.search_fuzzy,
                {"pattern": "service", "similarity_threshold": 0.8},
            ),
            ("file_stats", server.get_file_statistics, {"include_analysis": True}),
        ]

        memory_usage = []

        for op_name, operation, kwargs in operations:
            # Perform operation
            await operation(**kwargs)

            # Measure memory usage
            current_memory = server.resource_manager.get_memory_usage()
            memory_usage.append((op_name, current_memory))

            print(f"Memory after {op_name}: {current_memory}")

        # Check that memory usage is reasonable
        final_memory = memory_usage[-1][1]
        memory_increase = final_memory["cache_memory_mb"] - initial_memory["cache_memory_mb"]

        print(f"Total memory increase: {memory_increase:.2f} MB")

        # Memory increase should be reasonable (under 100MB for this test)
        assert memory_increase < 100.0


class TestResourceManagementPerformance:
    """Test resource management performance."""

    @pytest.mark.asyncio
    async def test_cache_performance(self, benchmark_runner):
        """Test cache operation performance."""
        from mcp.shared.resource_manager import ResourceManager

        resource_manager = ResourceManager()

        # Test cache set operations
        for i in range(100):
            key = f"test_key_{i}"
            value = {"id": i, "data": f"test_data_{i}", "timestamp": datetime.now().isoformat()}

            start_time = time.perf_counter()
            resource_manager.set_cache(key, value)
            end_time = time.perf_counter()

            benchmark_runner.results.setdefault("cache_set", []).append(end_time - start_time)

        # Test cache get operations
        for i in range(100):
            key = f"test_key_{i}"

            start_time = time.perf_counter()
            value = resource_manager.get_cache(key)
            end_time = time.perf_counter()

            assert value is not None
            benchmark_runner.results.setdefault("cache_get", []).append(end_time - start_time)

        # Test cache analytics
        start_time = time.perf_counter()
        analytics = resource_manager.get_cache_analytics()
        end_time = time.perf_counter()

        print(f"Cache analytics time: {end_time - start_time:.6f}s")
        print(f"Cache analytics: {analytics}")

        # Get statistics
        set_stats = benchmark_runner.get_stats("cache_set")
        get_stats = benchmark_runner.get_stats("cache_get")

        print(f"Cache set stats: {set_stats}")
        print(f"Cache get stats: {get_stats}")

        # Performance assertions
        assert set_stats["mean"] < 0.001  # Set should be under 1ms
        assert get_stats["mean"] < 0.001  # Get should be under 1ms
        assert analytics["hit_rate"] > 0.5  # Should have reasonable hit rate

    @pytest.mark.asyncio
    async def test_session_management_performance(self, benchmark_runner):
        """Test session management performance."""
        from mcp.shared.session_manager import SessionManager

        session_manager = SessionManager()

        # Test session creation
        session_ids = []
        for i in range(50):
            context = {"user_id": f"user_{i}", "project_type": "test_project", "session_number": i}

            start_time = time.perf_counter()
            session = await session_manager.create_session(f"user_{i}", context)
            end_time = time.perf_counter()

            session_ids.append(session["session_id"])
            benchmark_runner.results.setdefault("session_create", []).append(end_time - start_time)

        # Test session retrieval
        for session_id in session_ids:
            start_time = time.perf_counter()
            session = await session_manager.get_session(session_id)
            end_time = time.perf_counter()

            assert session is not None
            benchmark_runner.results.setdefault("session_get", []).append(end_time - start_time)

        # Test session updates
        for i, session_id in enumerate(session_ids[:10]):  # Test first 10
            new_context = {"updated": True, "iteration": i}

            start_time = time.perf_counter()
            await session_manager.update_session_context(session_id, new_context)
            end_time = time.perf_counter()

            benchmark_runner.results.setdefault("session_update", []).append(end_time - start_time)

        # Get statistics
        create_stats = benchmark_runner.get_stats("session_create")
        get_stats = benchmark_runner.get_stats("session_get")
        update_stats = benchmark_runner.get_stats("session_update")

        print(f"Session create stats: {create_stats}")
        print(f"Session get stats: {get_stats}")
        print(f"Session update stats: {update_stats}")

        # Performance assertions
        assert create_stats["mean"] < 0.01  # Create should be under 10ms
        assert get_stats["mean"] < 0.005  # Get should be under 5ms
        assert update_stats["mean"] < 0.01  # Update should be under 10ms


class TestScalabilityBenchmarks:
    """Test scalability with increasing load."""

    @pytest.mark.asyncio
    async def test_increasing_file_count_performance(self):
        """Test performance as file count increases."""
        results = {}

        # Test with different numbers of files
        file_counts = [10, 50, 100, 200]

        for count in file_counts:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create specified number of files
                for i in range(count):
                    file_content = f'''
def function_{i}():
    """Function number {i}."""
    result = "test_result_{i}"
    return result

class Class_{i}:
    """Class number {i}."""
    
    def __init__(self):
        self.value = {i}
        
    def method_{i}(self):
        """Method {i}."""
        return self.value * {i}
'''
                    file_path = Path(temp_dir) / f"file_{i}.py"
                    file_path.write_text(file_content)

                # Create server and measure search performance
                server = EnhancedPySearchMCPServer()
                server.search_engine.configure_search([temp_dir])

                start_time = time.perf_counter()
                result = await server.search_text(pattern="def function", context=3)
                end_time = time.perf_counter()

                duration = end_time - start_time
                result_count = len(result.get("results", []))

                results[count] = {
                    "duration": duration,
                    "results_found": result_count,
                    "results_per_second": result_count / duration if duration > 0 else 0,
                }

                print(f"Files: {count}, Duration: {duration:.4f}s, Results: {result_count}")

        # Analyze scalability
        print("\nScalability analysis:")
        for count, data in results.items():
            print(
                f"  {count} files: {data['duration']:.4f}s, "
                f"{data['results_found']} results, "
                f"{data['results_per_second']:.1f} results/s"
            )

        # Performance should scale reasonably
        max_duration = max(data["duration"] for data in results.values())
        assert max_duration < 10.0  # Even with 200 files, should complete under 10s

    @pytest.mark.asyncio
    async def test_concurrent_user_performance(self):
        """Test performance with multiple concurrent users."""
        server = EnhancedPySearchMCPServer()

        async def simulate_user_session(user_id: int):
            """Simulate a user session with multiple searches."""
            # Create session
            session = await server.create_search_session(
                context={"user_id": user_id, "session_type": "benchmark"}
            )
            session_id = session["session_id"]

            # Perform multiple searches
            search_patterns = ["class", "def", "import", "async", "return"]
            search_times = []

            for pattern in search_patterns:
                start_time = time.perf_counter()
                await server.search_text(pattern=pattern, context=3, session_id=session_id)
                end_time = time.perf_counter()

                search_times.append(end_time - start_time)

            return {
                "user_id": user_id,
                "session_id": session_id,
                "total_searches": len(search_patterns),
                "total_time": sum(search_times),
                "average_time": statistics.mean(search_times),
                "search_times": search_times,
            }

        # Test with increasing number of concurrent users
        user_counts = [1, 5, 10, 20]

        for user_count in user_counts:
            print(f"\nTesting with {user_count} concurrent users...")

            start_time = time.perf_counter()

            # Run concurrent user sessions
            tasks = [simulate_user_session(i) for i in range(user_count)]
            user_results = await asyncio.gather(*tasks)

            end_time = time.perf_counter()
            total_time = end_time - start_time

            # Analyze results
            total_searches = sum(r["total_searches"] for r in user_results)
            average_search_time = statistics.mean([r["average_time"] for r in user_results])

            print(f"  Total time: {total_time:.4f}s")
            print(f"  Total searches: {total_searches}")
            print(f"  Average search time: {average_search_time:.4f}s")
            print(f"  Searches per second: {total_searches / total_time:.1f}")

            # Performance assertions
            assert total_time < 60.0  # Should complete within 60 seconds
            assert average_search_time < 5.0  # Average search under 5 seconds


class TestMemoryEfficiency:
    """Test memory efficiency and garbage collection."""

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations."""
        server = EnhancedPySearchMCPServer()

        # Get initial memory
        initial_memory = server.resource_manager.get_memory_usage()
        memory_samples = [initial_memory["cache_memory_mb"]]

        # Perform many operations
        for i in range(100):
            # Create and destroy session
            session = await server.create_search_session(context={"iteration": i})

            # Perform search
            await server.search_text(
                pattern=f"test_{i}", context=3, session_id=session["session_id"]
            )

            # Sample memory every 10 iterations
            if i % 10 == 0:
                current_memory = server.resource_manager.get_memory_usage()
                memory_samples.append(current_memory["cache_memory_mb"])

        print(f"Memory samples: {memory_samples}")

        # Check memory growth
        memory_growth = memory_samples[-1] - memory_samples[0]
        print(f"Memory growth: {memory_growth:.2f} MB")

        # Memory growth should be reasonable (under 50MB)
        assert memory_growth < 50.0

        # Memory should not grow linearly (indicating proper cleanup)
        if len(memory_samples) > 5:
            # Check that memory growth rate decreases over time
            early_growth = memory_samples[2] - memory_samples[0]
            late_growth = memory_samples[-1] - memory_samples[-3]

            # Late growth should not be significantly higher than early growth
            assert late_growth <= early_growth * 2.0

    @pytest.mark.asyncio
    async def test_cache_eviction_performance(self):
        """Test performance of cache eviction mechanisms."""
        from mcp.shared.resource_manager import ResourceManager

        resource_manager = ResourceManager()

        # Fill cache beyond capacity
        start_time = time.perf_counter()

        for i in range(150):  # Exceeds default capacity of 100
            key = f"test_key_{i}"
            value = {"id": i, "data": "x" * 1000}  # 1KB per entry
            resource_manager.set_cache(key, value)

        fill_time = time.perf_counter() - start_time

        # Check that cache size is limited
        cache_size = len(resource_manager._cache)
        print(f"Cache size after filling: {cache_size}")
        print(f"Fill time: {fill_time:.4f}s")

        # Test cache cleanup performance
        start_time = time.perf_counter()
        cleaned = resource_manager.clean_expired()
        cleanup_time = time.perf_counter() - start_time

        print(f"Cleaned {cleaned} expired entries in {cleanup_time:.4f}s")

        # Performance assertions
        assert cache_size <= 100  # Should not exceed capacity
        assert fill_time < 1.0  # Should fill quickly
        assert cleanup_time < 0.1  # Cleanup should be fast


def print_benchmark_summary(benchmark_runner: BenchmarkRunner):
    """Print a summary of all benchmark results."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    for operation_name in sorted(benchmark_runner.results.keys()):
        if operation_name.endswith("_errors"):
            continue

        stats = benchmark_runner.get_stats(operation_name)
        if stats:
            print(f"\n{operation_name.upper()}:")
            print(f"  Count: {stats['count']}")
            print(f"  Mean:  {stats['mean']:.4f}s")
            print(f"  Min:   {stats['min']:.4f}s")
            print(f"  Max:   {stats['max']:.4f}s")
            print(f"  StdDev: {stats['stdev']:.4f}s")

            # Check for errors
            error_key = f"{operation_name}_errors"
            if error_key in benchmark_runner.results:
                error_count = len(benchmark_runner.results[error_key])
                print(f"  Errors: {error_count}")


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])
