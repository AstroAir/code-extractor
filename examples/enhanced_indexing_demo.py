"""
Enhanced Code Indexing Engine Demo

This example demonstrates the key features of the enhanced code indexing engine,
showing how to use the various components and capabilities.

Features Demonstrated:
- Content-addressed caching and tag-based indexing
- Multi-language support with tree-sitter parsing
- Advanced chunking strategies
- Vector database integration for semantic search
- Performance monitoring and optimization
- Error handling and recovery
- Distributed indexing for large codebases

Usage:
    python examples/enhanced_indexing_demo.py
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path

# Import enhanced indexing components
from src.pysearch.advanced_chunking import ChunkingConfig, ChunkingEngine, ChunkingStrategy
from src.pysearch.config import SearchConfig
from src.pysearch.content_addressing import IndexTag
from src.pysearch.enhanced_indexing_engine import EnhancedIndexingEngine, IndexCoordinator
from src.pysearch.enhanced_vector_db import EmbeddingConfig
from src.pysearch.indexes import EnhancedCodeSnippetsIndex, EnhancedFullTextIndex
from src.pysearch.performance_monitoring import PerformanceMonitor


async def create_sample_codebase(base_dir: Path) -> Path:
    """Create a sample codebase for demonstration."""
    print("Creating sample codebase...")
    
    # Python files
    python_dir = base_dir / "python_project"
    python_dir.mkdir(exist_ok=True)
    
    # Main application file
    (python_dir / "main.py").write_text('''
import json
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class User:
    """Represents a user in the system."""
    id: int
    name: str
    email: str
    active: bool = True

class UserManager:
    """Manages user operations and data persistence."""
    
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.users: Dict[int, User] = {}
    
    async def load_users(self) -> None:
        """Load users from storage."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                for user_data in data:
                    user = User(**user_data)
                    self.users[user.id] = user
        except FileNotFoundError:
            print("No existing user data found")
    
    async def save_users(self) -> None:
        """Save users to storage."""
        data = [user.__dict__ for user in self.users.values()]
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def add_user(self, user: User) -> bool:
        """Add a new user."""
        if user.id in self.users:
            return False
        self.users[user.id] = user
        return True
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self.users.get(user_id)
    
    def get_active_users(self) -> List[User]:
        """Get all active users."""
        return [user for user in self.users.values() if user.active]

async def main():
    """Main application entry point."""
    manager = UserManager("users.json")
    await manager.load_users()
    
    # Add sample users
    users = [
        User(1, "Alice Johnson", "alice@example.com"),
        User(2, "Bob Smith", "bob@example.com"),
        User(3, "Carol Davis", "carol@example.com", False),
    ]
    
    for user in users:
        manager.add_user(user)
    
    await manager.save_users()
    
    # Display active users
    active_users = manager.get_active_users()
    print(f"Active users: {len(active_users)}")
    for user in active_users:
        print(f"  {user.name} ({user.email})")

if __name__ == "__main__":
    asyncio.run(main())
''')
    
    # Utility functions
    (python_dir / "utils.py").write_text('''
import hashlib
import re
from typing import Any, List, Union

def calculate_hash(data: Union[str, bytes]) -> str:
    """Calculate SHA256 hash of data."""
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem usage."""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename

class ConfigManager:
    """Manages application configuration."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = {}
    
    def load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = self._get_default_config()
        return self.config
    
    def save_config(self) -> None:
        """Save configuration to file."""
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            "database_url": "sqlite:///app.db",
            "debug": False,
            "max_users": 1000,
            "cache_timeout": 3600,
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
''')
    
    # JavaScript files
    js_dir = base_dir / "js_project"
    js_dir.mkdir(exist_ok=True)
    
    (js_dir / "app.js").write_text('''
const express = require('express');
const cors = require('cors');
const { UserService } = require('./services/userService');

/**
 * Main application class for the web server.
 */
class Application {
    constructor(port = 3000) {
        this.port = port;
        this.app = express();
        this.userService = new UserService();
        this.setupMiddleware();
        this.setupRoutes();
    }
    
    /**
     * Setup Express middleware.
     */
    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json());
        this.app.use(express.static('public'));
    }
    
    /**
     * Setup API routes.
     */
    setupRoutes() {
        // User routes
        this.app.get('/api/users', async (req, res) => {
            try {
                const users = await this.userService.getAllUsers();
                res.json(users);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        this.app.get('/api/users/:id', async (req, res) => {
            try {
                const user = await this.userService.getUserById(req.params.id);
                if (user) {
                    res.json(user);
                } else {
                    res.status(404).json({ error: 'User not found' });
                }
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
        
        this.app.post('/api/users', async (req, res) => {
            try {
                const user = await this.userService.createUser(req.body);
                res.status(201).json(user);
            } catch (error) {
                res.status(400).json({ error: error.message });
            }
        });
    }
    
    /**
     * Start the server.
     */
    async start() {
        return new Promise((resolve) => {
            this.server = this.app.listen(this.port, () => {
                console.log(`Server running on port ${this.port}`);
                resolve();
            });
        });
    }
    
    /**
     * Stop the server.
     */
    async stop() {
        if (this.server) {
            this.server.close();
        }
    }
}

module.exports = { Application };
''')
    
    print(f"Sample codebase created at: {base_dir}")
    return base_dir


async def demo_basic_indexing(codebase_dir: Path):
    """Demonstrate basic enhanced indexing."""
    print("\n=== Demo 1: Basic Enhanced Indexing ===")
    
    # Create configuration
    config = SearchConfig(
        paths=[str(codebase_dir)],
        cache_dir=str(codebase_dir / "cache"),
        enable_enhanced_indexing=True,
    )
    
    # Initialize enhanced indexing engine
    engine = EnhancedIndexingEngine(config)
    await engine.initialize()
    
    # Perform indexing with progress tracking
    print("Indexing codebase...")
    start_time = time.time()
    
    async for progress in engine.refresh_index():
        print(f"[{progress.progress:.1%}] {progress.description}")
        
        if progress.warnings:
            for warning in progress.warnings:
                print(f"  Warning: {warning}")
    
    duration = time.time() - start_time
    print(f"Indexing completed in {duration:.2f} seconds")
    
    # Get index statistics
    stats = await engine.coordinator.get_index_stats()
    print(f"Total indexes: {stats['total_indexes']}")
    print(f"Index types: {', '.join(stats['index_types'])}")


async def demo_advanced_chunking(codebase_dir: Path):
    """Demonstrate advanced chunking capabilities."""
    print("\n=== Demo 2: Advanced Code Chunking ===")
    
    # Test different chunking strategies
    strategies = [
        ChunkingStrategy.STRUCTURAL,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.HYBRID,
    ]
    
    test_file = codebase_dir / "python_project" / "main.py"
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value} chunking:")
        
        config = ChunkingConfig(
            strategy=strategy,
            max_chunk_size=800,
            min_chunk_size=100,
        )
        
        engine = ChunkingEngine(config)
        chunks = await engine.chunk_file(str(test_file))
        
        print(f"  Generated {len(chunks)} chunks")
        
        # Show chunk statistics
        stats = engine.get_chunking_stats(chunks)
        print(f"  Average chunk size: {stats['average_size']:.0f} characters")
        print(f"  Average quality score: {stats['average_quality']:.2f}")
        print(f"  Chunk types: {', '.join(stats['chunk_types'].keys())}")


async def demo_semantic_search(codebase_dir: Path):
    """Demonstrate semantic search capabilities."""
    print("\n=== Demo 3: Semantic Search ===")
    
    # Note: This demo requires OpenAI API key for embeddings
    # For demonstration, we'll show the setup without actual API calls
    
    try:
        from src.pysearch.indexes.vector_index import EnhancedVectorIndex
        
        config = SearchConfig(
            paths=[str(codebase_dir)],
            cache_dir=str(codebase_dir / "cache"),
            embedding_provider="openai",
            # embedding_api_key="your-api-key-here",  # Uncomment with real key
        )
        
        # Create vector index
        vector_index = EnhancedVectorIndex(config)
        tag = IndexTag(str(codebase_dir), "main", "enhanced_vectors")
        
        print("Vector index configured (requires API key for actual embeddings)")
        print("Example semantic search queries:")
        print("  - 'user management and database operations'")
        print("  - 'error handling and validation'")
        print("  - 'configuration and settings management'")
        
        # Show what the search would return (structure)
        sample_results = [
            {
                "chunk_id": "main.py:15:45",
                "content": "class UserManager: ...",
                "file_path": "python_project/main.py",
                "similarity_score": 0.85,
                "entity_name": "UserManager",
            }
        ]
        
        print(f"Sample search result structure:")
        print(json.dumps(sample_results[0], indent=2))
        
    except ImportError as e:
        print(f"Vector indexing not available: {e}")
        print("Install with: pip install lancedb openai")


async def demo_performance_monitoring(codebase_dir: Path):
    """Demonstrate performance monitoring."""
    print("\n=== Demo 4: Performance Monitoring ===")
    
    config = SearchConfig(
        paths=[str(codebase_dir)],
        cache_dir=str(codebase_dir / "cache"),
    )
    
    # Create performance monitor
    monitor = PerformanceMonitor(config, codebase_dir / "cache")
    await monitor.start_monitoring()
    
    try:
        # Simulate indexing operation
        print("Starting performance monitoring...")
        
        # Monitor for a few seconds
        for i in range(5):
            await asyncio.sleep(1)
            print(f"Monitoring... {i+1}/5")
        
        # Get performance report
        report = await monitor.get_performance_report()
        
        print("\nPerformance Report:")
        print(f"  System Health Score: {report['health_score']:.2f}")
        print(f"  CPU Usage: {report['system']['cpu_usage_percent']:.1f}%")
        print(f"  Memory Usage: {report['system']['memory_usage_percent']:.1f}%")
        
        if report['optimizations']:
            print("\nOptimization Suggestions:")
            for opt in report['optimizations']:
                print(f"  - {opt['description']}")
        
    finally:
        await monitor.stop_monitoring()


async def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n=== Demo 5: Error Handling and Recovery ===")
    
    from src.pysearch.enhanced_error_handling import ErrorCollector, ErrorSeverity, RecoveryManager
    
    # Create error collector
    error_collector = ErrorCollector()
    
    # Simulate various errors
    await error_collector.add_error(
        "test_file.py",
        "File not found",
        ErrorSeverity.ERROR
    )
    
    await error_collector.add_error(
        "network_operation",
        "Connection timeout",
        ErrorSeverity.WARNING
    )
    
    await error_collector.add_error(
        "memory_operation",
        "Out of memory",
        ErrorSeverity.CRITICAL
    )
    
    # Get error summary
    summary = error_collector.get_error_summary()
    print(f"Total errors collected: {summary['total_errors']}")
    print(f"Error categories: {', '.join(summary['category_counts'].keys())}")
    print(f"Average impact score: {summary['average_impact_score']:.2f}")
    
    # Demonstrate recovery
    config = SearchConfig()
    recovery_manager = RecoveryManager(config)
    
    print("\nRecovery capabilities:")
    print("  - File access errors: Retry with different encodings")
    print("  - Network errors: Exponential backoff and connectivity testing")
    print("  - Memory errors: Reduce batch sizes and force garbage collection")
    print("  - Timeout errors: Increase timeout values")


async def demo_distributed_indexing(codebase_dir: Path):
    """Demonstrate distributed indexing for large codebases."""
    print("\n=== Demo 6: Distributed Indexing ===")
    
    try:
        from src.pysearch.distributed_indexing import DistributedIndexingEngine
        
        config = SearchConfig(
            paths=[str(codebase_dir)],
            cache_dir=str(codebase_dir / "cache"),
        )
        
        # Create distributed engine with multiple workers
        engine = DistributedIndexingEngine(
            config,
            num_workers=4,  # Use 4 workers for demonstration
        )
        
        print("Starting distributed indexing with 4 workers...")
        start_time = time.time()
        
        async for progress in engine.index_codebase([str(codebase_dir)]):
            print(f"[{progress.progress:.1%}] {progress.description}")
            
            if progress.debug_info:
                print(f"  Debug: {progress.debug_info}")
        
        duration = time.time() - start_time
        print(f"Distributed indexing completed in {duration:.2f} seconds")
        
        # Get worker statistics
        worker_stats = await engine.get_worker_stats()
        print(f"Workers used: {len(worker_stats)}")
        
        total_processed = sum(w.items_processed for w in worker_stats)
        print(f"Total items processed: {total_processed}")
        
    except ImportError as e:
        print(f"Distributed indexing not available: {e}")
        print("This feature requires additional dependencies")


async def demo_index_coordination(codebase_dir: Path):
    """Demonstrate index coordination and management."""
    print("\n=== Demo 7: Index Coordination ===")
    
    config = SearchConfig(
        paths=[str(codebase_dir)],
        cache_dir=str(codebase_dir / "cache"),
    )
    
    # Create coordinator and add specific indexes
    coordinator = IndexCoordinator(config)
    coordinator.add_index(EnhancedCodeSnippetsIndex(config))
    coordinator.add_index(EnhancedFullTextIndex(config))
    
    print(f"Added {len(coordinator.indexes)} indexes:")
    for index in coordinator.indexes:
        print(f"  - {index.artifact_id} (relative time: {index.relative_expected_time})")
    
    # Demonstrate search across different indexes
    tag = IndexTag(str(codebase_dir), "main", "enhanced_code_snippets")
    
    # Search code snippets
    snippets_index = coordinator.get_index("enhanced_code_snippets")
    if snippets_index:
        entities = await snippets_index.retrieve("UserManager", tag, limit=5)
        print(f"\nCode snippets search for 'UserManager': {len(entities)} results")
        
        for entity in entities[:2]:  # Show first 2 results
            print(f"  - {entity['name']} ({entity['entity_type']}) in {Path(entity['path']).name}")
    
    # Search full-text
    fulltext_index = coordinator.get_index("enhanced_full_text")
    if fulltext_index:
        files = await fulltext_index.retrieve("async def", tag, limit=5)
        print(f"\nFull-text search for 'async def': {len(files)} results")
        
        for file_result in files[:2]:  # Show first 2 results
            print(f"  - {Path(file_result['path']).name} ({file_result['language']})")


async def main():
    """Run all demonstrations."""
    print("Enhanced Code Indexing Engine Demo")
    print("=" * 50)
    
    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create sample codebase
        codebase_dir = await create_sample_codebase(temp_path)
        
        # Run demonstrations
        await demo_basic_indexing(codebase_dir)
        await demo_advanced_chunking(codebase_dir)
        await demo_semantic_search(codebase_dir)
        await demo_performance_monitoring(codebase_dir)
        await demo_error_handling()
        await demo_distributed_indexing(codebase_dir)
        await demo_index_coordination(codebase_dir)
        
        print("\n" + "=" * 50)
        print("Demo completed! Check the generated cache directory for index files.")
        print(f"Cache location: {codebase_dir / 'cache'}")
        
        # Show cache contents
        cache_dir = codebase_dir / "cache"
        if cache_dir.exists():
            print("\nGenerated cache files:")
            for cache_file in cache_dir.rglob("*"):
                if cache_file.is_file():
                    size_kb = cache_file.stat().st_size / 1024
                    print(f"  - {cache_file.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
