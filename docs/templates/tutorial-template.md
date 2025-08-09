# Tutorial: Getting Started with pysearch

This tutorial provides a hands-on introduction to the core features of pysearch. You will learn how to perform basic and advanced searches, configure the search engine, and build a simple code analysis tool.

## Learning Objectives

By the end of this tutorial, you will be able to:

- [ ] **Perform basic searches** for text and patterns in your codebase.
- [ ] **Configure the search engine** to include or exclude specific files and directories.
- [ ] **Use advanced search features** like regular expressions and AST-based queries.
- [ ] **Build a practical tool** to analyze your code and generate a report.

## Prerequisites

### Required Knowledge

- **Basic Python**: Understanding of functions, classes, and imports.
- **Command line**: Basic familiarity with your terminal or command prompt.

### Required Setup

- **Python 3.10+** installed on your system.
- **pysearch** installed (`pip install pysearch`).
- A **text editor** or IDE of your choice.
- A **sample codebase** to practice with (we'll provide instructions to create one).

## Estimated Time

⏱️ **Total time**: 25-35 minutes

- Setup: 5 minutes
- Lessons: 15-20 minutes
- Final Project: 5-10 minutes

---

## Setup

### 1. Verify Installation

First, let's make sure everything is properly installed:

```bash
# Check your Python version
python --version

# Check your pysearch installation
pysearch --version

# Test that pysearch is importable
python -c "from pysearch import PySearch; print('✅ pysearch is ready!')"
```

### 2. Create a Sample Project

For this tutorial, we'll create a small sample project to search through:

```bash
mkdir tutorial-workspace
cd tutorial-workspace

mkdir src
touch src/main.py
touch src/utils.py

mkdir tests
touch tests/test_main.py
```

Now, add some content to the files:

**`src/main.py`**
```python
import utils

# TODO: Add more comprehensive error handling
def main():
    """Main function to run the application."""
    print("Starting application...")
    utils.helper_function()

class MainApp:
    def __init__(self):
        self.name = "My App"

    def start(self):
        main()
```

**`src/utils.py`**
```python
def helper_function():
    """A helper function for the main application."""
    print("Executing helper function.")

# FIXME: This is a temporary implementation
class Utility:
    pass
```

**`tests/test_main.py`**
```python
from src import main

def test_main_function():
    """Test the main function."""
    # This is a placeholder test
    assert True
```

---

## Lesson 1: Basic Searching

### What You'll Learn

In this lesson, you'll learn how to perform a simple text search and examine the results.

### Step-by-Step Instructions

Create a new file named `tutorial.py` in your `tutorial-workspace` directory.

#### Step 1: Import and Configure pysearch

```python
# tutorial.py

from pysearch import PySearch, SearchConfig

# Configure pysearch to search in the 'src' directory
config = SearchConfig(paths=["./src"])
engine = PySearch(config)

print("✅ Engine configured.")
```

**What's happening here:**

- We import the necessary classes from the `pysearch` library.
- We create a `SearchConfig` object to specify which directories to search.
- We initialize the `PySearch` engine with our configuration.

#### Step 2: Perform a Search

Now, let's search for the word "import" in our source files.

```python
# Add to tutorial.py

results = engine.search("import")
print(f"Found {len(results.items)} match(es).")
```

#### Step 3: Examine the Results

Let's look at the details of the first match.

```python
# Add to tutorial.py

if results.items:
    first_result = results.items[0]
    print(f"\nFirst match found in: {first_result.file}")
    print(f"On line: {first_result.start_line}")
    print(f"Content: {first_result.lines[0].strip()}")
```

### Run Your Code

Execute your `tutorial.py` script from the command line:

```bash
python tutorial.py
```

**Expected Output:**

```
✅ Engine configured.
Found 1 match(es).

First match found in: src/main.py
On line: 1
Content: import utils
```

### Try It Yourself

1.  **Change the search pattern**: Try searching for "def" or "class".
2.  **Search a different directory**: Change the `paths` in your `SearchConfig` to `["./tests"]`.

---

## Lesson 2: Advanced Searching and Filtering

### What You'll Learn

Now, let's explore more advanced features like regular expressions and file filtering.

### Step-by-Step Instructions

Modify your `tutorial.py` file for this lesson.

#### Step 1: Use Regular Expressions

Let's find all `TODO` or `FIXME` comments in our code using a regular expression.

```python
# Modify tutorial.py

config = SearchConfig(paths=["./src"])
engine = PySearch(config)

# Search for TODO or FIXME using a regex pattern
results = engine.search("TODO|FIXME", regex=True)

print(f"Found {len(results.items)} TODO/FIXME comments:")
for item in results.items:
    print(f"- {item.file}:{item.start_line} -> {item.lines[0].strip()}")
```

**What's happening here:**

- We set `regex=True` in the `search` method to enable regular expression matching.
- The pattern `"TODO|FIXME"` matches lines containing either "TODO" or "FIXME".

#### Step 2: Filter Files

Let's search for all function definitions, but only in files named `main.py`.

```python
# Modify tutorial.py

config = SearchConfig(
    paths=["./src", "./tests"],  # Search in both directories
    include=["**/main.py"]      # Only include files named main.py
)
engine = PySearch(config)

results = engine.search("def ", regex=False)

print(f"\nFound {len(results.items)} function definitions in main.py:")
for item in results.items:
    print(f"- {item.file}:{item.start_line} -> {item.lines[0].strip()}")
```

**What's happening here:**

- We added an `include` pattern to our `SearchConfig` to restrict the search to specific files.

### Run Your Code

Execute your modified `tutorial.py` script.

**Expected Output:**

```
Found 2 TODO/FIXME comments:
- src/main.py:3 -> # TODO: Add more comprehensive error handling
- src/utils.py:5 -> # FIXME: This is a temporary implementation

Found 1 function definitions in main.py:
- src/main.py:4 -> def main():
```

---

## Putting It All Together: A Simple Code Analyzer

Let's use what we've learned to build a tool that generates a report about our codebase.

### Final Project Code

Replace the content of `tutorial.py` with the following code:

```python
#!/usr/bin/env python3
"""
Final Tutorial Project: A Simple Code Analysis Tool
"""

from pysearch import PySearch, SearchConfig

def analyze_codebase(path):
    """Analyzes a codebase and generates a report."""
    
    print(f"🔍 Analyzing codebase at: {path}")
    
    config = SearchConfig(
        paths=[path],
        include=["**/*.py"],
        exclude=["**/.venv/**", "**/__pycache__/**"]
    )
    engine = PySearch(config)
    
    # Analysis 1: Count functions
    function_results = engine.search("def ", regex=False)
    function_count = len(function_results.items)
    
    # Analysis 2: Count classes
    class_results = engine.search("class ", regex=False)
    class_count = len(class_results.items)
    
    # Analysis 3: Find TODO items
    todo_results = engine.search("TODO|FIXME", regex=True)
    todo_count = len(todo_results.items)
    
    # Generate report
    print("\n📊 Codebase Analysis Report")
    print("=" * 30)
    print(f"Functions found: {function_count}")
    print(f"Classes found: {class_count}")
    print(f"TODO/FIXME items: {todo_count}")
    
    if todo_count > 0:
        print("\n📝 TODO/FIXME Items:")
        for item in todo_results.items:
            print(f"  - {item.file}:{item.start_line} -> {item.lines[0].strip()}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    analyze_codebase("./")
```

### Running Your Final Project

Execute the script from your `tutorial-workspace` directory:

```bash
python tutorial.py
```

**Expected Output:**

```
🔍 Analyzing codebase at: ./

📊 Codebase Analysis Report
==============================
Functions found: 2
Classes found: 2
TODO/FIXME items: 2

📝 TODO/FIXME Items:
  - src/main.py:3 -> # TODO: Add more comprehensive error handling
  - src/utils.py:5 -> # FIXME: This is a temporary implementation

✅ Analysis complete!
```

---

## What You've Learned

Congratulations! You've completed the tutorial. You now know how to:

- [x] **Perform basic searches** using `pysearch`.
- [x] **Configure searches** with paths and filters.
- [x] **Use regular expressions** for advanced pattern matching.
- [x] **Build a simple but useful tool** to analyze your code.

## Next Steps

- **Experiment**: Modify the code analyzer to count imports or find other patterns.
- **Explore**: Read the [API Reference](api-reference.md) to discover more features.
- **Integrate**: Try using `pysearch` in your own projects.

### Additional Example: Find all function calls

```python
# Example: Find all function calls in the src directory
config = SearchConfig(paths=["./src"])
engine = PySearch(config)

# This regex is a simplified example and may not catch all edge cases
results = engine.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', regex=True)

print(f"\nFound {len(results.items)} potential function calls:")
for item in results.items:
    print(f"- {item.file}:{item.start_line} -> {item.lines[0].strip()}")
```

```