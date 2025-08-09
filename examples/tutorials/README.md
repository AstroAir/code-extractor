# pysearch Tutorials

This directory contains step-by-step tutorials for learning pysearch, from beginner to advanced usage.

## Tutorial Structure

### 🎯 Beginner Tutorials

Perfect for users new to pysearch:

- **[Getting Started](01_getting_started.py)** - Your first search with pysearch
- **[Basic Configuration](02_basic_configuration.py)** - Setting up search parameters
- **[Understanding Results](03_understanding_results.py)** - Working with search results
- **[File Patterns](04_file_patterns.py)** - Include/exclude patterns and file filtering

### 🚀 Intermediate Tutorials

For users comfortable with basics:

- **[Search Types](05_search_types.py)** - Text, regex, AST, and semantic search
- **[AST Filtering](06_ast_filtering.py)** - Advanced code structure filtering
- **[Performance Optimization](07_performance_optimization.py)** - Making searches faster
- **[Output Formats](08_output_formats.py)** - JSON, text, and highlighted output

### 🔧 Advanced Tutorials

For power users and integrators:

- **[Custom Workflows](09_custom_workflows.py)** - Building search-based tools
- **[Caching and Watching](10_caching_and_watching.py)** - Advanced performance features
- **[Multi-Repository Search](11_multi_repo_search.py)** - Searching across projects
- **[Integration Patterns](12_integration_patterns.py)** - Integrating with other tools

### 🎨 Specialized Tutorials

Domain-specific use cases:

- **[Code Analysis](13_code_analysis.py)** - Analyzing code patterns and metrics
- **[Refactoring Assistant](14_refactoring_assistant.py)** - Search-driven refactoring
- **[Documentation Generation](15_documentation_generation.py)** - Extracting docs from code
- **[Testing and QA](16_testing_and_qa.py)** - Quality assurance workflows

---

## How to Use These Tutorials

### Prerequisites

Before starting, ensure you have:

1. **Python 3.10+** installed
2. **pysearch** installed (`pip install pysearch` or development setup)
3. **Basic Python knowledge** (variables, functions, imports)
4. **Command line familiarity** (optional for CLI tutorials)

### Running Tutorials

Each tutorial is a standalone Python script:

```bash
# Run from the project root
python examples/tutorials/01_getting_started.py

# Or from the tutorials directory
cd examples/tutorials
python 01_getting_started.py
```

### Tutorial Format

Each tutorial follows this structure:

1. **Introduction** - What you'll learn
2. **Concepts** - Key concepts explained
3. **Code Examples** - Practical, runnable code
4. **Exercises** - Try-it-yourself challenges
5. **Next Steps** - What to learn next

### Learning Path

**Recommended learning sequence:**

```
Beginner Path:
01 → 02 → 03 → 04 → 05

Intermediate Path:
05 → 06 → 07 → 08 → 09

Advanced Path:
09 → 10 → 11 → 12

Specialized Paths:
- Code Analysis: 01 → 05 → 13 → 14
- Tool Integration: 01 → 08 → 12 → 16
- Performance Focus: 01 → 07 → 10 → 11
```

---

## Tutorial Descriptions

### Beginner Level

#### 01. Getting Started
**What you'll learn:**
- Install and verify pysearch
- Perform your first search
- Understand basic output

**Key concepts:**
- PySearch engine
- Basic search patterns
- Result interpretation

#### 02. Basic Configuration
**What you'll learn:**
- Configure search paths
- Set basic options
- Understand configuration hierarchy

**Key concepts:**
- SearchConfig class
- Path specification
- Option precedence

#### 03. Understanding Results
**What you'll learn:**
- Navigate search results
- Extract useful information
- Handle different result types

**Key concepts:**
- SearchResult structure
- SearchItem details
- Match spans and context

#### 04. File Patterns
**What you'll learn:**
- Include/exclude file patterns
- Language-specific filtering
- Directory exclusions

**Key concepts:**
- Glob patterns
- Language detection
- Performance implications

### Intermediate Level

#### 05. Search Types
**What you'll learn:**
- Text vs. regex vs. AST search
- When to use each type
- Performance trade-offs

**Key concepts:**
- Search strategies
- Pattern complexity
- Accuracy vs. speed

#### 06. AST Filtering
**What you'll learn:**
- Filter by code structures
- Function/class/decorator filters
- Complex AST queries

**Key concepts:**
- Abstract Syntax Trees
- Structural patterns
- Code understanding

#### 07. Performance Optimization
**What you'll learn:**
- Identify performance bottlenecks
- Optimize configuration
- Monitor resource usage

**Key concepts:**
- Parallel processing
- Memory management
- I/O optimization

#### 08. Output Formats
**What you'll learn:**
- JSON for automation
- Highlighted terminal output
- Custom result processing

**Key concepts:**
- Output formats
- Data serialization
- Tool integration

### Advanced Level

#### 09. Custom Workflows
**What you'll learn:**
- Build search-based tools
- Combine multiple searches
- Error handling patterns

**Key concepts:**
- Workflow design
- Error recovery
- Tool composition

#### 10. Caching and Watching
**What you'll learn:**
- Enable result caching
- Set up file watching
- Optimize for development

**Key concepts:**
- Cache strategies
- File system monitoring
- Development workflows

#### 11. Multi-Repository Search
**What you'll learn:**
- Search across projects
- Repository prioritization
- Cross-project analysis

**Key concepts:**
- Multi-repo architecture
- Repository management
- Aggregated results

#### 12. Integration Patterns
**What you'll learn:**
- Integrate with IDEs
- Build CLI tools
- Create web interfaces

**Key concepts:**
- API design
- Protocol integration
- User interfaces

### Specialized Level

#### 13. Code Analysis
**What you'll learn:**
- Extract code metrics
- Identify patterns
- Generate reports

**Key concepts:**
- Static analysis
- Pattern recognition
- Metric calculation

#### 14. Refactoring Assistant
**What you'll learn:**
- Find refactoring candidates
- Analyze dependencies
- Suggest improvements

**Key concepts:**
- Code quality
- Dependency analysis
- Refactoring strategies

#### 15. Documentation Generation
**What you'll learn:**
- Extract API documentation
- Generate code examples
- Create reference docs

**Key concepts:**
- Documentation extraction
- API discovery
- Content generation

#### 16. Testing and QA
**What you'll learn:**
- Find test coverage gaps
- Identify code smells
- Automate quality checks

**Key concepts:**
- Quality assurance
- Test analysis
- Automation patterns

---

## Interactive Learning

### Exercises and Challenges

Each tutorial includes:

- **Guided exercises** with step-by-step instructions
- **Challenge problems** to test your understanding
- **Real-world scenarios** based on common use cases
- **Extension activities** for deeper exploration

### Example Exercise Format

```python
# Exercise: Find all async functions in your codebase
# 1. Configure pysearch for your project
# 2. Use regex search to find async functions
# 3. Filter results by function name patterns
# 4. Generate a summary report

# Your code here:
# config = SearchConfig(...)
# engine = PySearch(config)
# results = engine.search(...)

# Solution provided at the end of each tutorial
```

### Practice Projects

Apply your learning with these practice projects:

1. **Code Inventory Tool** - Catalog functions and classes
2. **Dependency Analyzer** - Map import relationships
3. **Style Checker** - Find style inconsistencies
4. **Documentation Validator** - Check docstring coverage
5. **Migration Helper** - Find deprecated API usage

---

## Getting Help

### Tutorial Support

If you get stuck:

1. **Check the solution** at the end of each tutorial
2. **Review prerequisites** to ensure you have the basics
3. **Run the example code** exactly as shown first
4. **Check the main documentation** for detailed API info

### Common Issues

- **Import errors**: Ensure pysearch is properly installed
- **Path issues**: Use absolute paths or check current directory
- **No results**: Verify your search patterns and file paths
- **Performance issues**: Start with small directories first

### Community Resources

- **GitHub Discussions**: Ask questions and share solutions
- **Example Gallery**: See community-contributed examples
- **Issue Tracker**: Report bugs or request features
- **Documentation**: Comprehensive API and usage docs

---

## Contributing to Tutorials

### Adding New Tutorials

We welcome contributions! To add a tutorial:

1. **Follow the naming convention**: `NN_descriptive_name.py`
2. **Include comprehensive docstrings** and comments
3. **Add exercises and solutions**
4. **Test with different Python versions**
5. **Update this README** with the new tutorial

### Improving Existing Tutorials

Help us improve by:

- **Fixing errors** or unclear explanations
- **Adding more examples** or use cases
- **Improving code comments** and documentation
- **Suggesting better exercises** or challenges

### Tutorial Guidelines

When creating tutorials:

1. **Start simple** and build complexity gradually
2. **Use real-world examples** that users can relate to
3. **Include error handling** and edge cases
4. **Provide multiple approaches** when appropriate
5. **Test thoroughly** before submitting

---

## Next Steps

After completing these tutorials:

1. **Explore the main examples** in the parent directory
2. **Read the comprehensive documentation** in `docs/`
3. **Try building your own tools** using pysearch
4. **Contribute back** by sharing your use cases
5. **Join the community** to help others learn

Happy learning! 🚀
