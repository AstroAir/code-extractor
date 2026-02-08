# Quick Start

Get up and running with pysearch in under 5 minutes.

## Prerequisites

- Python 3.10 or higher
- pip installed

## Install

```bash
pip install -e .
```

Verify the installation:

```bash
pysearch --version
```

---

## 30-Second CLI Example

```bash
# Find all function definitions in Python files
pysearch find --pattern "def " --path ./src --include "**/*.py"

# Search with regex for handler functions
pysearch find --pattern "def.*handler" --regex --context 3

# Find class definitions with AST filtering
pysearch find --pattern "class" --ast --filter-class-name ".*Test"
```

## First Python Script

```python
from pysearch import PySearch, SearchConfig

# Create search engine
config = SearchConfig(paths=["./src"], include=["**/*.py"])
engine = PySearch(config)

# Perform search
results = engine.search("def main")

# Display results
for item in results.items:
    print(f"{item.file}: lines {item.start_line}-{item.end_line}")
```

---

## Search Modes at a Glance

| Mode           | Flag         | Example                                                          |
| -------------- | ------------ | ---------------------------------------------------------------- |
| Text (default) | —            | `pysearch find --pattern "TODO"`                                 |
| Regex          | `--regex`    | `pysearch find --pattern "def.*handler" --regex`                 |
| AST            | `--ast`      | `pysearch find --pattern "def" --ast --filter-func-name "main"` |
| Semantic       | `--semantic` | `pysearch find --pattern "database connection" --semantic`       |
| Fuzzy          | `--fuzzy`    | `pysearch find --pattern "authetication" --fuzzy`                |
| Boolean        | `--logic`    | `pysearch find --pattern "(async AND handler) NOT test" --logic` |
| GraphRAG       | subcommand   | `pysearch graphrag --query "auth handler" --path ./src`          |

---

## Next Steps

- **[Usage Guide](../guide/usage.md)** — Full CLI & Python API reference with all options
- **[Configuration](../guide/configuration.md)** — Customize pysearch for your project
- **[Installation](installation.md)** — Detailed installation options (dev mode, extras, etc.)
