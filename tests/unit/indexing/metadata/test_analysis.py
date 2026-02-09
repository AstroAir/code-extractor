"""Tests for pysearch.indexing.metadata.analysis module."""

from __future__ import annotations

from pathlib import Path

from pysearch.core.types import Language
from pysearch.indexing.metadata.analysis import (
    calculate_entity_complexity,
    calculate_file_complexity,
    create_entity_text,
    extract_dependencies,
    extract_imports,
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


class TestCalculateEntityComplexity:
    """Tests for calculate_entity_complexity function."""

    def test_basic(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e1",
            name="process",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            signature="def process():",
        )
        content = "def process():\n    if x:\n        for i in r:\n            pass\n    return\n"
        score = calculate_entity_complexity(entity, content)
        assert score > 0

    def test_no_signature_returns_zero(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e2",
            name="x",
            entity_type=EntityType.VARIABLE,
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            signature=None,
        )
        score = calculate_entity_complexity(entity, "x = 1\n")
        assert score == 0.0

    def test_capped_at_50(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e3",
            name="huge",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=300,
            signature="def huge():",
        )
        content = "\n".join(["if x:" for _ in range(300)])
        score = calculate_entity_complexity(entity, content)
        assert score <= 50.0


class TestCreateEntityText:
    """Tests for create_entity_text function."""

    def test_basic(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e1",
            name="main",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            signature="def main():",
        )
        text = create_entity_text(entity)
        assert isinstance(text, str)
        assert "main" in text

    def test_with_docstring(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e2",
            name="helper",
            entity_type=EntityType.FUNCTION,
            file_path=Path("test.py"),
            start_line=1,
            end_line=5,
            signature="def helper():",
            docstring="Helps with things.",
        )
        text = create_entity_text(entity)
        assert "Helps with things." in text

    def test_with_properties(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e3",
            name="cls",
            entity_type=EntityType.CLASS,
            file_path=Path("test.py"),
            start_line=1,
            end_line=10,
            signature="class cls:",
            properties={"decorator": "dataclass"},
        )
        text = create_entity_text(entity)
        assert "decorator" in text

    def test_name_only(self):
        from pysearch.core.types import CodeEntity, EntityType

        entity = CodeEntity(
            id="e4",
            name="x",
            entity_type=EntityType.VARIABLE,
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
        )
        text = create_entity_text(entity)
        assert text == "x"
