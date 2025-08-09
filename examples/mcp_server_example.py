#!/usr/bin/env python3
"""
PySearch MCP Server Usage Examples

This script demonstrates how to use the PySearch MCP server functionality
both directly and through the MCP protocol.
"""

import asyncio

from mcp.servers.mcp_server import PySearchMCPServer


async def demonstrate_search_functionality():
    """Demonstrate the core search functionality of the MCP server."""

    print("PySearch MCP Server Examples")
    print("=" * 50)

    # Create the server instance
    server = PySearchMCPServer()

    # Example 1: Basic text search
    print("\n1. Basic Text Search")
    print("-" * 30)

    try:
        result = await server.search_text(pattern="def main", paths=["./src"], context=2)

        print(f"Found {result.total_matches} matches in {result.execution_time_ms:.2f}ms")
        print(f"Searched {result.stats['files_scanned']} files")

        # Show first few results
        for i, item in enumerate(result.items[:3]):
            print(f"\nMatch {i+1}: {item['file']} (lines {item['start_line']}-{item['end_line']})")
            for line in item["lines"][:3]:  # Show first 3 lines
                print(f"  {line}")

    except Exception as e:
        print(f"Text search failed: {e}")

    # Example 2: Regex search
    print("\n\n2. Regex Pattern Search")
    print("-" * 30)

    try:
        result = await server.search_regex(pattern=r"class \w+Test", context=1)

        print(f"Found {result.total_matches} test classes")

        for i, item in enumerate(result.items[:2]):
            print(f"\nTest class {i+1}: {item['file']}")
            for line in item["lines"]:
                print(f"  {line}")

    except Exception as e:
        print(f"Regex search failed: {e}")

    # Example 3: AST-based search
    print("\n\n3. AST-based Structural Search")
    print("-" * 30)

    try:
        result = await server.search_ast(pattern="def", func_name=".*_handler$", context=1)

        print(f"Found {result.total_matches} handler functions")

        for i, item in enumerate(result.items[:2]):
            print(f"\nHandler {i+1}: {item['file']}")
            for line in item["lines"]:
                print(f"  {line}")

    except Exception as e:
        print(f"AST search failed: {e}")

    # Example 4: Semantic search
    print("\n\n4. Semantic Concept Search")
    print("-" * 30)

    try:
        result = await server.search_semantic(concept="database", context=1)

        print(f"Found {result.total_matches} database-related code sections")

        for i, item in enumerate(result.items[:2]):
            print(f"\nDatabase code {i+1}: {item['file']}")
            for line in item["lines"]:
                print(f"  {line}")

    except Exception as e:
        print(f"Semantic search failed: {e}")

    # Example 5: Configuration management
    print("\n\n5. Configuration Management")
    print("-" * 30)

    try:
        # Get current configuration
        config = await server.get_search_config()
        print("Current configuration:")
        print(f"  Paths: {config.paths}")
        print(f"  Include patterns: {config.include_patterns}")
        print(f"  Context lines: {config.context_lines}")
        print(f"  Parallel: {config.parallel}")
        print(f"  Workers: {config.workers}")

        # Update configuration
        new_config = await server.configure_search(
            paths=["./src", "./examples"], context=5, workers=2
        )

        print("\nUpdated configuration:")
        print(f"  Paths: {new_config.paths}")
        print(f"  Context lines: {new_config.context_lines}")
        print(f"  Workers: {new_config.workers}")

    except Exception as e:
        print(f"Configuration management failed: {e}")

    # Example 6: Utility functions
    print("\n\n6. Utility Functions")
    print("-" * 30)

    try:
        # Get supported languages
        languages = await server.get_supported_languages()
        print(f"Supported languages ({len(languages)}): {', '.join(languages[:10])}...")

        # Clear caches
        cache_result = await server.clear_caches()
        print(f"Cache status: {cache_result['status']}")

        # Get search history
        history = await server.get_search_history(limit=3)
        print(f"\nRecent searches ({len(history)}):")
        for i, search in enumerate(history):
            print(
                f"  {i+1}. Pattern: '{search['query']['pattern']}' "
                f"({search['result_count']} results in {search['execution_time_ms']:.2f}ms)"
            )

    except Exception as e:
        print(f"Utility functions failed: {e}")


def demonstrate_mcp_integration():
    """Show how the server would integrate with MCP clients."""

    print("\n\nMCP Integration Example")
    print("=" * 50)

    print(
        """
When integrated with an MCP client (like Claude Desktop), the tools would be available as:

1. search_text(pattern, paths?, context?, case_sensitive?)
   - Search for literal text patterns
   
2. search_regex(pattern, paths?, context?, case_sensitive?)
   - Search using regular expressions
   
3. search_ast(pattern, func_name?, class_name?, decorator?, imported?, paths?, context?)
   - Structural search using AST analysis
   
4. search_semantic(concept, paths?, context?)
   - Semantic concept search
   
5. configure_search(paths?, include_patterns?, exclude_patterns?, context?, parallel?, workers?, languages?)
   - Update search configuration
   
6. get_search_config()
   - Get current configuration
   
7. get_supported_languages()
   - List supported programming languages
   
8. clear_caches()
   - Clear search caches
   
9. get_search_history(limit?)
   - Get recent search history

Example MCP client usage:
- "Search for all database connection functions in the src directory"
- "Find test classes that use pytest fixtures"
- "Show me error handling patterns in the codebase"
- "Configure the search to include TypeScript files"
"""
    )


async def main():
    """Main example runner."""
    await demonstrate_search_functionality()
    demonstrate_mcp_integration()

    print("\n\nTo use this as an MCP server:")
    print("1. Install FastMCP: pip install fastmcp")
    print("2. Uncomment FastMCP code in pysearch_mcp_server.py")
    print("3. Run: python pysearch_mcp_server.py")
    print("4. Configure your MCP client to use the server")


if __name__ == "__main__":
    asyncio.run(main())
