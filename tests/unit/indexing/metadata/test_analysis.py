"""Tests for pysearch.indexing.metadata.analysis module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.core.types import Language
from pysearch.indexing.metadata.analysis import (
    calculate_entity_complexity,
    calculate_file_complexity,
    extract_imports,
    extract_dependencies,
    create_entity_text,
)


class TestCalculateFileComplexity:
    """Tests for calculate_file_complexity function."""

    def test_empty_content(self):
        score = calculate_file_complexity("", Language.PYTHON)
        assert score == 0.0

    def test_simple_code(self):
        code = "x = 1\ny = 2\n"
        score = calculate_file_complexity(code, Language.PYTHON)
        assert score > 0

    def test_complex_code(self):
        code = "if x:\n  for i in r:\n    while True:\n      try:\n        pass\n      except:\n        pass\n"
        score = calculate_file_complexity(code, Language.PYTHON)
        assert score > 1.0

    def test_capped_at_100(self):
        # Very long file
        code = "\n".join(["if x:" for _ in range(500)])
        score = calculate_file_complexity(code, Language.PYTHON)
        assert score <= 100.0


class TestExtractImports:
    """Tests for extract_imports function."""

    def test_python_imports(self):
        code = "import os\nimport sys\nfrom pathlib import Path\n"
        imports = extract_imports(code, Language.PYTHON)
        assert isinstance(imports, list)
        assert len(imports) >= 1

    def test_no_imports(self):
        code = "x = 1\ny = 2\n"
        imports = extract_imports(code, Language.PYTHON)
        assert isinstance(imports, list)

    def test_empty_content(self):
        imports = extract_imports("", Language.PYTHON)
        assert imports == []


class TestExtractDependencies:
    """Tests for extract_dependencies function."""

    def test_basic(self):
        code = "import os\nfrom pathlib import Path\n"
        deps = extract_dependencies(code, Language.PYTHON)
        assert isinstance(deps, list)

    def test_empty(self):
        deps = extract_dependencies("", Language.PYTHON)
        assert isinstance(deps, list)


class TestCreateEntityText:
    """Tests for create_entity_text function."""

    def test_basic(self):
        from pysearch.core.types import CodeEntity, EntityType
        entity = CodeEntity(
            id="e1", name="main", entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"), start_line=1, end_line=5,
            signature="def main():",
        )
        text = create_entity_text(entity)
        assert isinstance(text, str)
        assert "main" in text
