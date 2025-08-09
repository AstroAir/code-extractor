#!/usr/bin/env python3
"""
Tutorial 05: Search Types - Text, Regex, AST, and Semantic

This tutorial explores the different types of searches available in pysearch
and when to use each one for maximum effectiveness.

What you'll learn:
- Text search for simple string matching
- Regex search for pattern matching
- AST search for code structure analysis
- Semantic search for conceptual matching
- Performance trade-offs between search types
- Best practices for choosing search types

Prerequisites:
- Completed Tutorial 01 (Getting Started)
- Basic understanding of regular expressions (helpful)
- Familiarity with Python syntax concepts

Estimated time: 20 minutes
"""

from pathlib import Path
from pysearch import PySearch, SearchConfig
from pysearch.types import ASTFilters, Query, OutputFormat
import time


def lesson_1_text_search():
    """
    Lesson 1: Text Search - Simple and Fast
    
    Text search is the simplest and fastest search type. It looks for exact
    string matches in file contents.
    """
    print("=== Lesson 1: Text Search ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(
        paths=[str(project_root / "src")],
        include=["**/*.py"],
        context=2
    )
    engine = PySearch(config)
    
    print("Text search looks for exact string matches...")
    
    # Example 1: Simple text search
    print("\n🔍 Example 1: Finding 'def main'")
    start_time = time.time()
    results = engine.search("def main")  # No regex flag = text search
    elapsed = time.time() - start_time
    
    print(f"   Found {len(results.items)} matches in {elapsed*1000:.1f}ms")
    
    if results.items:
        item = results.items[0]
        print(f"   First match in {item.file.name}:")
        for i, line in enumerate(item.lines):
            line_num = item.start_line + i
            marker = ">>>" if "def main" in line else "   "
            print(f"   {marker} {line_num:3d}: {line.rstrip()}")
    
    # Example 2: Case sensitivity
    print("\n🔍 Example 2: Case sensitivity")
    upper_results = engine.search("DEF MAIN")  # Different case
    print(f"   'DEF MAIN' (uppercase): {len(upper_results.items)} matches")
    print(f"   'def main' (lowercase): {len(results.items)} matches")
    print("   → Text search is case-sensitive by default")
    
    # Example 3: Partial matches
    print("\n🔍 Example 3: Partial vs exact matches")
    def_results = engine.search("def")  # Just "def"
    def_space_results = engine.search("def ")  # "def" with space
    
    print(f"   'def': {len(def_results.items)} matches")
    print(f"   'def ': {len(def_space_results.items)} matches")
    print("   → Adding space makes search more specific")
    
    print("\n💡 Text search is best for:")
    print("   ✅ Exact string matches")
    print("   ✅ Fast searches")
    print("   ✅ Simple patterns")
    print("   ❌ Complex patterns")
    print("   ❌ Case-insensitive searches")


def lesson_2_regex_search():
    """
    Lesson 2: Regex Search - Powerful Pattern Matching
    
    Regex search uses regular expressions for complex pattern matching.
    It's more flexible but slower than text search.
    """
    print("\n=== Lesson 2: Regex Search ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    print("Regex search uses regular expressions for pattern matching...")
    
    # Example 1: Basic regex patterns
    print("\n🔍 Example 1: Function definitions with any name")
    start_time = time.time()
    results = engine.search(r"def \w+", regex=True)  # regex=True enables regex
    elapsed = time.time() - start_time
    
    print(f"   Pattern 'def \\w+': {len(results.items)} matches in {elapsed*1000:.1f}ms")
    
    if results.items:
        # Show some function names
        function_names = []
        for item in results.items[:5]:
            for line in item.lines:
                if "def " in line:
                    try:
                        import re
                        match = re.search(r'def (\w+)', line)
                        if match:
                            function_names.append(match.group(1))
                    except:
                        pass
        
        print(f"   Sample functions found: {', '.join(function_names[:5])}")
    
    # Example 2: Case-insensitive search
    print("\n🔍 Example 2: Case-insensitive search")
    case_insensitive = engine.search(r"(?i)class", regex=True)  # (?i) = case insensitive
    case_sensitive = engine.search(r"class", regex=True)
    
    print(f"   'class' (case sensitive): {len(case_sensitive.items)} matches")
    print(f"   '(?i)class' (case insensitive): {len(case_insensitive.items)} matches")
    
    # Example 3: Complex patterns
    print("\n🔍 Example 3: Complex patterns")
    
    # Find async functions
    async_functions = engine.search(r"async def \w+", regex=True)
    print(f"   Async functions: {len(async_functions.items)} matches")
    
    # Find functions with specific naming patterns
    test_functions = engine.search(r"def test_\w+", regex=True)
    print(f"   Test functions: {len(test_functions.items)} matches")
    
    # Find import statements
    import_statements = engine.search(r"^import \w+|^from \w+", regex=True)
    print(f"   Import statements: {len(import_statements.items)} matches")
    
    # Example 4: Performance comparison
    print("\n⚡ Performance comparison:")
    
    # Text search
    start = time.time()
    text_results = engine.search("def main")
    text_time = time.time() - start
    
    # Regex search
    start = time.time()
    regex_results = engine.search(r"def main", regex=True)
    regex_time = time.time() - start
    
    print(f"   Text search: {text_time*1000:.1f}ms")
    print(f"   Regex search: {regex_time*1000:.1f}ms")
    print(f"   Regex is {regex_time/text_time:.1f}x slower")
    
    print("\n💡 Regex search is best for:")
    print("   ✅ Pattern matching")
    print("   ✅ Case-insensitive searches")
    print("   ✅ Complex text patterns")
    print("   ✅ Flexible matching")
    print("   ❌ Simple exact matches (use text instead)")


def lesson_3_ast_search():
    """
    Lesson 3: AST Search - Code Structure Analysis
    
    AST (Abstract Syntax Tree) search understands Python code structure
    and can find specific code elements like functions, classes, and decorators.
    """
    print("\n=== Lesson 3: AST Search ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    print("AST search understands Python code structure...")
    
    # Example 1: Find all functions
    print("\n🔍 Example 1: Finding all function definitions")
    start_time = time.time()
    
    # AST search with function filter
    ast_filters = ASTFilters(func_name=".*")  # Any function name
    query = Query(
        pattern="def",  # Pattern is less important for AST
        use_ast=True,
        ast_filters=ast_filters
    )
    results = engine.run(query)
    elapsed = time.time() - start_time
    
    print(f"   Found {len(results.items)} functions in {elapsed*1000:.1f}ms")
    
    # Example 2: Find functions with specific names
    print("\n🔍 Example 2: Functions with specific naming patterns")
    
    # Find test functions
    test_filters = ASTFilters(func_name=r"test_.*")
    test_query = Query(pattern="def", use_ast=True, ast_filters=test_filters)
    test_results = engine.run(test_query)
    print(f"   Test functions (test_*): {len(test_results.items)} matches")
    
    # Find private functions
    private_filters = ASTFilters(func_name=r"_.*")
    private_query = Query(pattern="def", use_ast=True, ast_filters=private_filters)
    private_results = engine.run(private_query)
    print(f"   Private functions (_*): {len(private_results.items)} matches")
    
    # Find main functions
    main_filters = ASTFilters(func_name=r".*main.*")
    main_query = Query(pattern="def", use_ast=True, ast_filters=main_filters)
    main_results = engine.run(main_query)
    print(f"   Main functions (*main*): {len(main_results.items)} matches")
    
    # Example 3: Find classes
    print("\n🔍 Example 3: Finding classes")
    
    class_filters = ASTFilters(class_name=".*")  # Any class
    class_query = Query(pattern="class", use_ast=True, ast_filters=class_filters)
    class_results = engine.run(class_query)
    print(f"   All classes: {len(class_results.items)} matches")
    
    # Find specific class patterns
    config_classes = ASTFilters(class_name=r".*Config.*")
    config_query = Query(pattern="class", use_ast=True, ast_filters=config_classes)
    config_results = engine.run(config_query)
    print(f"   Config classes: {len(config_results.items)} matches")
    
    # Example 4: Find decorators
    print("\n🔍 Example 4: Finding decorators")
    
    decorator_filters = ASTFilters(decorator=".*")  # Any decorator
    decorator_query = Query(pattern="@", use_ast=True, ast_filters=decorator_filters)
    decorator_results = engine.run(decorator_query)
    print(f"   Functions with decorators: {len(decorator_results.items)} matches")
    
    # Find specific decorators
    property_filters = ASTFilters(decorator="property")
    property_query = Query(pattern="@", use_ast=True, ast_filters=property_filters)
    property_results = engine.run(property_query)
    print(f"   @property decorators: {len(property_results.items)} matches")
    
    # Example 5: Accuracy comparison
    print("\n🎯 Accuracy comparison:")
    
    # Text search for functions (less accurate)
    text_functions = engine.search("def ")
    
    # AST search for functions (more accurate)
    ast_function_filters = ASTFilters(func_name=".*")
    ast_function_query = Query(pattern="def", use_ast=True, ast_filters=ast_function_filters)
    ast_functions = engine.run(ast_function_query)
    
    print(f"   Text search 'def ': {len(text_functions.items)} matches")
    print(f"   AST function search: {len(ast_functions.items)} matches")
    print("   → AST search is more accurate for code structures")
    
    print("\n💡 AST search is best for:")
    print("   ✅ Code structure analysis")
    print("   ✅ Finding specific functions/classes")
    print("   ✅ Decorator-based searches")
    print("   ✅ Accurate code element detection")
    print("   ❌ Non-Python files")
    print("   ❌ Simple text patterns")


def lesson_4_semantic_search():
    """
    Lesson 4: Semantic Search - Conceptual Matching
    
    Semantic search finds code based on meaning and concepts,
    not just exact text matches.
    """
    print("\n=== Lesson 4: Semantic Search ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    print("Semantic search finds conceptually related code...")
    
    # Example 1: Conceptual search
    print("\n🔍 Example 1: Finding file-related code")
    
    try:
        # Search for file-related concepts
        file_concepts = engine.search("file operations", use_semantic=True)
        print(f"   'file operations': {len(file_concepts.items)} conceptual matches")
        
        # Search for error-related concepts
        error_concepts = engine.search("error handling", use_semantic=True)
        print(f"   'error handling': {len(error_concepts.items)} conceptual matches")
        
        # Search for configuration concepts
        config_concepts = engine.search("configuration settings", use_semantic=True)
        print(f"   'configuration settings': {len(config_concepts.items)} conceptual matches")
        
    except Exception as e:
        print(f"   ⚠️  Semantic search may not be fully available: {e}")
        print("   This is normal - semantic search is an advanced feature")
    
    print("\n💡 Semantic search is best for:")
    print("   ✅ Conceptual code discovery")
    print("   ✅ Finding related functionality")
    print("   ✅ Exploring unfamiliar codebases")
    print("   ✅ Cross-language concept matching")
    print("   ❌ Exact matches (use text/regex instead)")
    print("   ❌ Performance-critical searches")


def lesson_5_choosing_search_types():
    """
    Lesson 5: Choosing the Right Search Type
    
    Learn when to use each search type for optimal results.
    """
    print("\n=== Lesson 5: Choosing the Right Search Type ===")
    
    print("Decision guide for choosing search types:\n")
    
    scenarios = [
        {
            "scenario": "Find all TODO comments",
            "best": "Text search",
            "pattern": "TODO",
            "reason": "Simple exact match, fast"
        },
        {
            "scenario": "Find functions ending with '_handler'",
            "best": "Regex search",
            "pattern": r"def \w+_handler",
            "reason": "Pattern matching needed"
        },
        {
            "scenario": "Find all async functions",
            "best": "AST search",
            "pattern": "async def with AST filters",
            "reason": "Code structure analysis"
        },
        {
            "scenario": "Find error handling code",
            "best": "Semantic search",
            "pattern": "error handling concepts",
            "reason": "Conceptual matching"
        },
        {
            "scenario": "Find specific import statement",
            "best": "Text search",
            "pattern": "from pysearch import",
            "reason": "Exact match, fast"
        },
        {
            "scenario": "Find all class definitions",
            "best": "AST search",
            "pattern": "class with AST filters",
            "reason": "Accurate structure detection"
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"📋 Scenario {i}: {scenario['scenario']}")
        print(f"   Best choice: {scenario['best']}")
        print(f"   Pattern: {scenario['pattern']}")
        print(f"   Why: {scenario['reason']}\n")
    
    print("🎯 Performance vs Accuracy Trade-offs:")
    print("   Text Search:     ⚡⚡⚡ Fast    🎯🎯   Accurate")
    print("   Regex Search:    ⚡⚡  Medium  🎯🎯🎯 Accurate")
    print("   AST Search:      ⚡    Slower  🎯🎯🎯🎯 Very Accurate")
    print("   Semantic Search: ⚡    Slower  🎯🎯🎯 Conceptual")


def exercise_search_comparison():
    """
    Exercise: Compare different search types on the same query
    """
    print("\n=== Exercise: Search Type Comparison ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    print("Let's compare different approaches to find function definitions:\n")
    
    # 1. Text search
    print("1️⃣ Text search for 'def ':")
    start = time.time()
    text_results = engine.search("def ")
    text_time = time.time() - start
    print(f"   Results: {len(text_results.items)} matches in {text_time*1000:.1f}ms")
    
    # 2. Regex search
    print("\n2️⃣ Regex search for function pattern:")
    start = time.time()
    regex_results = engine.search(r"def \w+\(", regex=True)
    regex_time = time.time() - start
    print(f"   Results: {len(regex_results.items)} matches in {regex_time*1000:.1f}ms")
    
    # 3. AST search
    print("\n3️⃣ AST search for functions:")
    start = time.time()
    ast_filters = ASTFilters(func_name=".*")
    ast_query = Query(pattern="def", use_ast=True, ast_filters=ast_filters)
    ast_results = engine.run(ast_query)
    ast_time = time.time() - start
    print(f"   Results: {len(ast_results.items)} matches in {ast_time*1000:.1f}ms")
    
    print("\n📊 Comparison Summary:")
    print(f"   Text:  {len(text_results.items):3d} results, {text_time*1000:5.1f}ms")
    print(f"   Regex: {len(regex_results.items):3d} results, {regex_time*1000:5.1f}ms")
    print(f"   AST:   {len(ast_results.items):3d} results, {ast_time*1000:5.1f}ms")
    
    print("\n💭 Analysis:")
    print("   - Text search is fastest but may include false positives")
    print("   - Regex search is more precise than text")
    print("   - AST search is most accurate for actual function definitions")
    print("   - Choose based on your accuracy vs speed requirements")


def main():
    """
    Main tutorial runner
    """
    print("🚀 Welcome to pysearch Tutorial 05: Search Types!")
    print("=" * 60)
    print("Learn when and how to use different search types effectively.\n")
    
    try:
        lesson_1_text_search()
        lesson_2_regex_search()
        lesson_3_ast_search()
        lesson_4_semantic_search()
        lesson_5_choosing_search_types()
        exercise_search_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 Congratulations! You've mastered search types!")
        
        print("\nWhat you learned:")
        print("✅ Text search for exact matches")
        print("✅ Regex search for patterns")
        print("✅ AST search for code structures")
        print("✅ Semantic search for concepts")
        print("✅ How to choose the right search type")
        
        print("\n🔜 Next steps:")
        print("- Try Tutorial 06: AST Filtering")
        print("- Experiment with complex regex patterns")
        print("- Practice choosing search types for different scenarios")
        print("- Explore performance optimization techniques")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure you're running this from the correct directory")


if __name__ == "__main__":
    main()
