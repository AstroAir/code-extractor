# Documentation Style Guide

This guide establishes consistent standards for all pysearch documentation to ensure clarity, accessibility, and maintainability.

## Table of Contents

- [General Principles](#general-principles)
- [Document Structure](#document-structure)
- [Writing Style](#writing-style)
- [Formatting Standards](#formatting-standards)
- [Code Examples](#code-examples)
- [API Documentation](#api-documentation)
- [Tutorial Guidelines](#tutorial-guidelines)
- [Templates](#templates)

---

## General Principles

### Clarity First

- **Write for your audience**: Consider the reader's experience level, from beginner to advanced.
- **Be concise**: Remove unnecessary words without losing meaning.
- **Use active voice**: "Configure the search" instead of "The search should be configured."
- **Avoid jargon**: Explain technical terms when first introduced.

### Consistency

- **Follow established patterns** throughout all documentation.
- **Use consistent terminology** (e.g., always "search engine" not "search tool").
- **Maintain uniform formatting** across all documents.
- **Apply the same structure** to similar document types.

### Accessibility

- **Use clear headings** for easy navigation.
- **Include a table of contents** for longer documents.
- **Provide alternative text** for images and diagrams.
- **Use descriptive link text** instead of "click here."

### Maintainability

- **Keep examples current** and test them regularly.
- **Use relative links** for internal references.
- **Include modification dates** for time-sensitive content.
- **Write modular content** that can be easily updated.

---

## Document Structure

### Standard Document Layout

Every documentation file should follow this structure:

```markdown
# Document Title

Brief description of what this document covers (1-2 sentences).

## Table of Contents

- [Section 1](#section-1)
- [Section 2](#section-2)
- [Subsection 2.1](#subsection-21)

---

## Section 1

Content here...

### Subsection 1.1

Content here...

---

## See Also

- [Related Document 1](link1.md)
- [Related Document 2](link2.md)
```

### Heading Hierarchy

Use consistent heading levels:

- **H1 (`#`)**: Document title (only one per document).
- **H2 (`##`)**: Major sections.
- **H3 (`###`)**: Subsections.
- **H4 (`####`)**: Sub-subsections (use sparingly).

### Table of Contents

Include a table of contents for documents longer than 3 sections:

```markdown
## Table of Contents

- [Getting Started](#getting-started)
- [Basic Usage](#basic-usage)
  - [Configuration](#configuration)
  - [Search Types](#search-types)
- [Advanced Topics](#advanced-topics)
```

---

## Writing Style

### Tone and Voice

- **Professional but approachable**: Friendly without being casual.
- **Confident**: Use definitive statements when appropriate.
- **Helpful**: Anticipate user questions and provide solutions.
- **Inclusive**: Use gender-neutral language and avoid assumptions.

### Grammar and Usage

#### Preferred Constructions

✅ **Good Examples:**

- "Configure pysearch to search your codebase."
- "The search returns results in JSON format."
- "You can filter results by file type."

❌ **Avoid:**

- "pysearch can be configured to search your codebase" (passive voice).
- "The search will return results in JSON format" (unnecessary future tense).
- "Results can be filtered by file type" (passive voice).

#### Technical Terms

- **Define terms** on first use: "AST (Abstract Syntax Tree)."
- **Use consistent terminology** throughout all documentation.
- **Prefer simple terms** when possible: "use" instead of "utilize."

#### Common Terms and Conventions

| Preferred | Avoid |
|-----------|-------|
| pysearch | PySearch (except in code) |
| search engine | search tool, searcher |
| configuration | config (in prose) |
| command line | command-line (as adjective) |
| file system | filesystem |
| codebase | code base |

---

## Formatting Standards

### Text Formatting

- **Bold** (`**text**`): For emphasis, UI elements, important terms.
- *Italic* (`*text*`): For file names, variables, first use of terms.
- `Code` (`` `text` ``): For code snippets, commands, file paths.
- **Bold code** (`**`code`**`): For important code elements.

### Lists

#### Unordered Lists

Use `-` for consistency:

```markdown
- First item
- Second item
  - Nested item
  - Another nested item
- Third item
```

#### Ordered Lists

Use numbers with periods:

```markdown
1. First step
2. Second step
   1. Sub-step
   2. Another sub-step
3. Third step
```

### Links

#### Internal Links

Use relative paths:

```markdown
- [Usage Guide](usage.md)
- [API Reference](api-reference.md)
- [Examples](../examples/README.md)
```

#### External Links

Include descriptive text:

```markdown
- [Python Documentation](https://docs.python.org/)
- [Regular Expressions Guide](https://regexr.com/)
```

### Tables

Use consistent formatting:

```markdown
| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |
```

### Admonitions

Use for important information:

```markdown
!!! note
    This is a note with important information.

!!! warning
    This is a warning about potential issues.

!!! tip
    This is a helpful tip for users.
```

---

## Code Examples

### Code Block Standards

#### Language Specification

Always specify the language:

```markdown
```python
from pysearch import PySearch
engine = PySearch()
```

```bash
pysearch find --pattern "def main"
```

```json
{
  "results": [],
  "stats": {}
}
```

```

#### Complete Examples
Provide complete, runnable examples:

```python
# Good: Complete example
from pysearch import PySearch, SearchConfig

config = SearchConfig(paths=["./src"])
engine = PySearch(config)
results = engine.search("def main")
print(f"Found {len(results.items)} matches")
```

```python
# Avoid: Incomplete example
results = engine.search("def main")  # Where does 'engine' come from?
```

#### Comments and Explanations

Add helpful comments:

```python
# Configure search for Python files only
config = SearchConfig(
    paths=["./src"],           # Search in source directory
    include=["**/*.py"],       # Only Python files
    exclude=["**/.venv/**"]    # Skip virtual environment
)

# Create search engine
engine = PySearch(config)

# Perform search
results = engine.search("def main")
```

### Command Line Examples

Show complete commands with context:

```bash
# Basic search in current directory
pysearch find --pattern "TODO"

# Search with specific paths and patterns
pysearch find \
  --pattern "def.*handler" \
  --regex \
  --path ./src \
  --path ./tests \
  --context 3
```

### Output Examples

Show expected output when helpful:

```bash
$ pysearch find --pattern "def main" --stats
Found 3 matches in 2 files
Scanned 45 files in 125ms
```

---

## API Documentation

### Function Documentation

Use this template for functions:

```python
def search_files(pattern: str, paths: list[str], regex: bool = False) -> SearchResult:
    """Search for pattern in specified files.
    
    Args:
        pattern: The search pattern to match.
        paths: List of file paths to search.
        regex: Whether to treat pattern as regex (default: False).
        
    Returns:
        SearchResult containing matches and metadata.
        
    Raises:
        SearchError: If search operation fails.
        FileNotFoundError: If specified paths don't exist.
        
    Example:
        >>> result = search_files("def main", ["./src"])
        >>> print(f"Found {len(result.items)} matches")
        Found 3 matches
    """
```

### Class Documentation

Use this template for classes:

```python
class SearchConfig:
    """Configuration for search operations.
    
    This class manages all configuration options for pysearch,
    including search paths, file patterns, and performance settings.
    
    Attributes:
        paths: List of directories to search.
        include: File patterns to include.
        exclude: File patterns to exclude.
        
    Example:
        >>> config = SearchConfig(paths=["./src"])
        >>> config.include = ["**/*.py"]
        >>> engine = PySearch(config)
    """
```

---

## Tutorial Guidelines

### Tutorial Structure

Every tutorial should follow this structure:

1. **Introduction**: What will be learned
2. **Prerequisites**: Required knowledge/setup
3. **Step-by-step content**: Numbered lessons
4. **Exercises**: Hands-on practice
5. **Summary**: What was covered
6. **Next steps**: Where to go next

### Writing Tutorials

#### Use Progressive Disclosure

Start simple and build complexity:

```markdown
## Lesson 1: Basic Search
Learn to perform simple text searches.

## Lesson 2: Adding Filters
Add file type and path filters to your searches.

## Lesson 3: Advanced Patterns
Use regular expressions for complex patterns.
```

#### Include Exercises

Provide hands-on practice:

```markdown
### Exercise 1: Your Turn
Try searching for these patterns in your own codebase:
1. Find all TODO comments
2. Locate test functions
3. Search for import statements

### Solution
Here's how to approach each exercise...
```

#### Show Expected Results

Help users verify their progress:

```markdown
When you run this command:
```bash
pysearch find --pattern "def test_"
```

You should see output similar to:

```
Found 15 matches in 8 files
./tests/test_api.py:10: def test_basic_search():
./tests/test_config.py:25: def test_configuration():
...
```

```

---

## Templates

### Document Templates

#### User Guide Template

See [user-guide-template.md](./templates/user-guide-template.md) for the full template.

#### API Reference Template

See [api-reference-template.md](./templates/api-reference-template.md) for the full template.

### Checklist Templates

#### Documentation Review Checklist

- [ ] **Structure**: Follows standard document layout
- [ ] **Headings**: Uses consistent heading hierarchy
- [ ] **TOC**: Includes table of contents (if needed)
- [ ] **Writing**: Uses active voice and clear language
- [ ] **Code**: All examples are complete and tested
- [ ] **Links**: All internal links work correctly
- [ ] **Formatting**: Consistent with style guide
- [ ] **Grammar**: Proofread for errors
- [ ] **Accessibility**: Includes alt text for images

#### Tutorial Review Checklist

- [ ] **Learning objectives**: Clearly stated
- [ ] **Prerequisites**: Listed and accurate
- [ ] **Progressive structure**: Builds complexity gradually
- [ ] **Code examples**: Complete and runnable
- [ ] **Exercises**: Provide meaningful practice
- [ ] **Solutions**: Include example solutions
- [ ] **Next steps**: Guide to further learning

---

## Maintenance

### Regular Reviews

- **Monthly**: Check for broken links and outdated examples.
- **Quarterly**: Review and update screenshots and UI references.
- **Per release**: Update version-specific information.
- **Annually**: Comprehensive style and structure review.

### Version Control

- **Track changes**: Use meaningful commit messages for documentation.
- **Review process**: All documentation changes should be reviewed.
- **Testing**: Verify all code examples work with current version.

---

## Tools and Resources

### Recommended Tools

- **Markdown editor**: Typora, Mark Text, or VS Code
- **Link checker**: markdown-link-check
- **Spell checker**: aspell or VS Code extensions
- **Grammar checker**: Grammarly or LanguageTool

### Style Resources

- [Google Developer Documentation Style Guide](https://developers.google.com/style)
- [Microsoft Writing Style Guide](https://docs.microsoft.com/en-us/style-guide/)
- [Plain Language Guidelines](https://www.plainlanguage.gov/guidelines/)

---

This style guide is a living document. Please suggest improvements and updates as the project evolves.
