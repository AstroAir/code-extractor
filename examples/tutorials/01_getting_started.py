#!/usr/bin/env python3
"""
Tutorial 01: Getting Started with pysearch

This tutorial introduces you to pysearch and walks you through your first search.
You'll learn the basics of setting up and using the search engine.

What you'll learn:
- How to import and use pysearch
- Perform your first search
- Understand basic search results
- Navigate the output

Prerequisites:
- Python 3.10+ installed
- pysearch installed (pip install pysearch)
- Basic Python knowledge

Estimated time: 10 minutes
"""

from pathlib import Path
from pysearch import PySearch, SearchConfig


def lesson_1_your_first_search():
    """
    Lesson 1: Perform your very first search with pysearch
    
    We'll start with the simplest possible search to find Python function definitions.
    """
    print("=== Lesson 1: Your First Search ===")
    
    # Step 1: Create a basic search configuration
    # We'll search in the current project's source directory
    project_root = Path(__file__).parent.parent.parent  # Go up to project root
    src_path = project_root / "src"
    
    print(f"Searching in: {src_path}")
    
    # Step 2: Configure the search engine
    config = SearchConfig(
        paths=[str(src_path)],  # Where to search
        include=["**/*.py"],    # Only Python files
    )
    
    # Step 3: Create the search engine
    engine = PySearch(config)
    
    # Step 4: Perform a simple search
    print("\nSearching for 'def main'...")
    results = engine.search("def main")
    
    # Step 5: Examine the results
    print(f"✅ Search completed!")
    print(f"   Found {len(results.items)} matches")
    print(f"   Scanned {results.stats.files_scanned} files")
    print(f"   Search took {results.stats.elapsed_ms:.1f} milliseconds")
    
    # Step 6: Look at the first result (if any)
    if results.items:
        first_result = results.items[0]
        print(f"\n📄 First match found in: {first_result.file}")
        print(f"   Lines {first_result.start_line}-{first_result.end_line}:")
        
        for i, line in enumerate(first_result.lines):
            line_number = first_result.start_line + i
            print(f"   {line_number:3d}: {line}")
    else:
        print("\n🔍 No matches found. Try searching for 'import' instead!")


def lesson_2_understanding_configuration():
    """
    Lesson 2: Understanding search configuration
    
    Learn how to customize your search with different configuration options.
    """
    print("\n=== Lesson 2: Understanding Configuration ===")
    
    # Let's create a more detailed configuration
    project_root = Path(__file__).parent.parent.parent
    
    config = SearchConfig(
        paths=[str(project_root / "src")],           # Search paths
        include=["**/*.py"],                         # Include patterns
        exclude=["**/__pycache__/**", "**/.git/**"], # Exclude patterns
        context=3,                                   # Lines of context around matches
        parallel=True,                               # Enable parallel processing
        enable_docstrings=True,                      # Search in docstrings
        enable_comments=True,                        # Search in comments
        enable_strings=False,                        # Skip string literals
    )
    
    engine = PySearch(config)
    
    # Search for imports to see configuration in action
    print("Searching for 'import' with 3 lines of context...")
    results = engine.search("import")
    
    print(f"✅ Found {len(results.items)} matches")
    
    # Show one result with context
    if results.items:
        result = results.items[0]
        print(f"\n📄 Example with context from: {result.file}")
        print(f"   Lines {result.start_line}-{result.end_line}:")
        
        for i, line in enumerate(result.lines):
            line_number = result.start_line + i
            # Highlight the middle line (where the match is)
            middle_line = len(result.lines) // 2
            marker = ">>>" if i == middle_line else "   "
            print(f"{marker} {line_number:3d}: {line}")


def lesson_3_different_search_patterns():
    """
    Lesson 3: Trying different search patterns
    
    Explore how different patterns return different results.
    """
    print("\n=== Lesson 3: Different Search Patterns ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    # Try several different search patterns
    patterns = [
        ("class", "Find class definitions"),
        ("def ", "Find function definitions"),
        ("import", "Find import statements"),
        ("TODO", "Find TODO comments"),
        ("return", "Find return statements"),
    ]
    
    print("Comparing different search patterns:\n")
    
    for pattern, description in patterns:
        results = engine.search(pattern)
        print(f"🔍 '{pattern}' ({description})")
        print(f"   Results: {len(results.items)} matches in {results.stats.files_matched} files")
        
        # Show the file with the most matches
        if results.items:
            files_with_matches = {}
            for item in results.items:
                file_path = str(item.file)
                files_with_matches[file_path] = files_with_matches.get(file_path, 0) + 1
            
            most_matches_file = max(files_with_matches, key=files_with_matches.get)
            match_count = files_with_matches[most_matches_file]
            print(f"   Most matches: {match_count} in {Path(most_matches_file).name}")
        
        print()


def lesson_4_working_with_results():
    """
    Lesson 4: Working with search results
    
    Learn how to process and analyze search results programmatically.
    """
    print("=== Lesson 4: Working with Results ===")
    
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    # Search for function definitions
    results = engine.search("def ")
    
    print(f"Analyzing {len(results.items)} function definitions...\n")
    
    # Group results by file
    files_with_functions = {}
    for item in results.items:
        file_path = str(item.file)
        if file_path not in files_with_functions:
            files_with_functions[file_path] = []
        files_with_functions[file_path].append(item)
    
    # Show summary statistics
    print("📊 Summary Statistics:")
    print(f"   Total functions found: {len(results.items)}")
    print(f"   Files with functions: {len(files_with_functions)}")
    print(f"   Average functions per file: {len(results.items) / len(files_with_functions):.1f}")
    
    # Show top files by function count
    print("\n📈 Files with most functions:")
    sorted_files = sorted(
        files_with_functions.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )
    
    for file_path, functions in sorted_files[:5]:  # Top 5 files
        file_name = Path(file_path).name
        print(f"   {file_name}: {len(functions)} functions")
    
    # Show some function names (extract from the matched lines)
    print("\n🔧 Sample function names found:")
    function_names = []
    for item in results.items[:10]:  # First 10 results
        for line in item.lines:
            if "def " in line:
                # Simple extraction of function name
                try:
                    def_part = line.split("def ")[1]
                    func_name = def_part.split("(")[0].strip()
                    if func_name and func_name not in function_names:
                        function_names.append(func_name)
                except (IndexError, AttributeError):
                    continue
    
    for name in function_names[:8]:  # Show first 8 unique names
        print(f"   - {name}()")


def exercise_1():
    """
    Exercise 1: Your turn to explore!
    
    Try modifying the searches to learn more about the codebase.
    """
    print("\n=== Exercise 1: Your Turn! ===")
    print("Try these challenges:")
    print("1. Search for 'class' and count how many classes are in the codebase")
    print("2. Search for 'import' and see which modules are imported most")
    print("3. Search for 'TODO' or 'FIXME' to find areas needing work")
    print("4. Search for 'test' to find test-related code")
    print("\nModify the code below to try these searches:")
    
    # Your code here - try the challenges above!
    project_root = Path(__file__).parent.parent.parent
    config = SearchConfig(paths=[str(project_root / "src")])
    engine = PySearch(config)
    
    # Challenge 1: Count classes
    print("\n🎯 Challenge 1: Counting classes")
    class_results = engine.search("class ")
    print(f"   Found {len(class_results.items)} class definitions")
    
    # Challenge 2: Most common imports (simplified)
    print("\n🎯 Challenge 2: Common imports")
    import_results = engine.search("import ")
    print(f"   Found {len(import_results.items)} import statements")
    
    # You can extend this to analyze which modules are imported most!
    
    print("\n💡 Tip: Try modifying the search patterns and see what you discover!")


def main():
    """
    Main tutorial runner
    """
    print("🚀 Welcome to pysearch Tutorial 01: Getting Started!")
    print("=" * 60)
    print("This tutorial will teach you the basics of using pysearch.")
    print("Follow along and try the exercises at the end.\n")
    
    try:
        # Run all lessons
        lesson_1_your_first_search()
        lesson_2_understanding_configuration()
        lesson_3_different_search_patterns()
        lesson_4_working_with_results()
        exercise_1()
        
        print("\n" + "=" * 60)
        print("🎉 Congratulations! You've completed Tutorial 01!")
        print("\nWhat you learned:")
        print("✅ How to create a PySearch engine")
        print("✅ How to configure search parameters")
        print("✅ How to perform basic searches")
        print("✅ How to analyze search results")
        
        print("\n🔜 Next steps:")
        print("- Try Tutorial 02: Basic Configuration")
        print("- Experiment with different search patterns")
        print("- Explore the examples/ directory")
        print("- Read the documentation in docs/")
        
    except ImportError as e:
        print(f"❌ Error: {e}")
        print("\nMake sure pysearch is installed:")
        print("pip install pysearch")
        print("\nOr if you're in development mode:")
        print("pip install -e .")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check that you're running this from the correct directory")
        print("and that the source files exist.")


if __name__ == "__main__":
    main()
