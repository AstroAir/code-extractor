"""Tests for pysearch.analysis.graphrag.core module."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysearch.analysis.graphrag.core import (
    EntityExtractor,
    RelationshipMapper,
)
from pysearch.core.types import CodeEntity, EntityType, Language, RelationType


# ---------------------------------------------------------------------------
# EntityExtractor
# ---------------------------------------------------------------------------
class TestEntityExtractor:
    """Tests for EntityExtractor class."""

    def test_init(self):
        extractor = EntityExtractor()
        assert isinstance(extractor.language_extractors, dict)
        assert Language.PYTHON in extractor.language_extractors
        assert Language.JAVASCRIPT in extractor.language_extractors
        assert Language.JAVA in extractor.language_extractors
        assert Language.CSHARP in extractor.language_extractors

    async def test_extract_from_python_file(self, tmp_path: Path):
        py_file = tmp_path / "test.py"
        py_file.write_text(
            "def hello():\n    pass\n\nclass World:\n    pass\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        assert isinstance(entities, list)
        names = [e.name for e in entities]
        assert "hello" in names
        assert "World" in names

    async def test_extract_empty_file(self, tmp_path: Path):
        py_file = tmp_path / "empty.py"
        py_file.write_text("", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        assert entities == []

    async def test_extract_functions(self, tmp_path: Path):
        py_file = tmp_path / "funcs.py"
        py_file.write_text(
            "def foo():\n    pass\n\ndef bar():\n    pass\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        funcs = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(funcs) == 2

    async def test_extract_classes_with_methods(self, tmp_path: Path):
        py_file = tmp_path / "cls.py"
        py_file.write_text(
            "class MyClass:\n    def method(self):\n        pass\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 1
        assert classes[0].name == "MyClass"

    async def test_extract_imports(self, tmp_path: Path):
        py_file = tmp_path / "imports.py"
        py_file.write_text(
            "import os\nfrom pathlib import Path\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        imports = [e for e in entities if e.entity_type == EntityType.IMPORT]
        assert len(imports) >= 2

    async def test_extract_variables(self, tmp_path: Path):
        py_file = tmp_path / "vars.py"
        py_file.write_text("x = 1\ny = 'hello'\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        variables = [e for e in entities if e.entity_type == EntityType.VARIABLE]
        assert len(variables) >= 2

    async def test_extract_async_function(self, tmp_path: Path):
        py_file = tmp_path / "async_func.py"
        py_file.write_text(
            "async def fetch():\n    return True\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        funcs = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(funcs) >= 1
        assert funcs[0].properties.get("is_async") is True

    async def test_extract_python_docstrings(self, tmp_path: Path):
        py_file = tmp_path / "doc.py"
        py_file.write_text(
            'def greet():\n    """Say hello."""\n    print("hi")\n', encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        funcs = [e for e in entities if e.entity_type == EntityType.FUNCTION]
        assert len(funcs) >= 1
        assert funcs[0].docstring is not None

    async def test_extract_class_inheritance(self, tmp_path: Path):
        py_file = tmp_path / "inherit.py"
        py_file.write_text(
            "class Base:\n    pass\n\nclass Child(Base):\n    pass\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        child = [e for e in entities if e.name == "Child"]
        assert len(child) == 1
        assert "Base" in child[0].properties.get("bases", [])

    async def test_extract_nonexistent_file(self, tmp_path: Path):
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(tmp_path / "nonexistent.py")
        assert entities == []

    async def test_extract_unsupported_language(self, tmp_path: Path):
        css_file = tmp_path / "style.css"
        css_file.write_text("body { color: red; }\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(css_file)
        assert entities == []

    async def test_extract_javascript_file(self, tmp_path: Path):
        js_file = tmp_path / "app.js"
        js_file.write_text(
            "function hello() {\n    console.log('hi');\n}\n"
            "class Widget {\n    constructor() {}\n}\n",
            encoding="utf-8",
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(js_file)
        names = [e.name for e in entities]
        assert "hello" in names
        assert "Widget" in names

    async def test_extract_java_file(self, tmp_path: Path):
        java_file = tmp_path / "App.java"
        java_file.write_text(
            "public class App {\n"
            "    public void run() {\n"
            "    }\n"
            "}\n",
            encoding="utf-8",
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(java_file)
        names = [e.name for e in entities]
        assert "App" in names

    async def test_extract_csharp_file(self, tmp_path: Path):
        cs_file = tmp_path / "Program.cs"
        cs_file.write_text(
            "public class Program {\n}\n", encoding="utf-8"
        )
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(cs_file)
        classes = [e for e in entities if e.entity_type == EntityType.CLASS]
        assert len(classes) >= 1

    async def test_entity_has_file_path(self, tmp_path: Path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("def f(): pass\n", encoding="utf-8")
        extractor = EntityExtractor()
        entities = await extractor.extract_from_file(py_file)
        assert len(entities) >= 1
        assert entities[0].file_path == py_file


# ---------------------------------------------------------------------------
# RelationshipMapper
# ---------------------------------------------------------------------------
class TestRelationshipMapper:
    """Tests for RelationshipMapper class."""

    def test_init(self):
        mapper = RelationshipMapper()
        assert mapper.dependency_analyzer is not None

    async def test_map_empty(self):
        mapper = RelationshipMapper()
        relationships = await mapper.map_relationships([], {})
        assert relationships == []

    async def test_map_inheritance(self):
        base = CodeEntity(
            id="class_Base_1",
            name="Base",
            entity_type=EntityType.CLASS,
            file_path=Path("a.py"),
            start_line=1,
            end_line=2,
            properties={"bases": []},
        )
        child = CodeEntity(
            id="class_Child_3",
            name="Child",
            entity_type=EntityType.CLASS,
            file_path=Path("a.py"),
            start_line=3,
            end_line=4,
            properties={"bases": ["Base"]},
        )
        mapper = RelationshipMapper()
        rels = await mapper.map_relationships([base, child], {})
        inherit_rels = [r for r in rels if r.relation_type == RelationType.INHERITS]
        assert len(inherit_rels) >= 1
        assert inherit_rels[0].source_entity_id == "class_Child_3"

    async def test_map_containment(self):
        cls = CodeEntity(
            id="class_Foo_1",
            name="Foo",
            entity_type=EntityType.CLASS,
            file_path=Path("a.py"),
            start_line=1,
            end_line=5,
            properties={},
        )
        method = CodeEntity(
            id="func_bar_2",
            name="bar",
            entity_type=EntityType.FUNCTION,
            file_path=Path("a.py"),
            start_line=2,
            end_line=4,
            properties={},
        )
        mapper = RelationshipMapper()
        rels = await mapper.map_relationships([cls, method], {})
        contain_rels = [r for r in rels if r.relation_type == RelationType.CONTAINS]
        assert len(contain_rels) >= 1

    async def test_map_call_relationships(self):
        caller = CodeEntity(
            id="func_main_1",
            name="main",
            entity_type=EntityType.FUNCTION,
            file_path=Path("a.py"),
            start_line=1,
            end_line=3,
            properties={},
        )
        callee = CodeEntity(
            id="func_helper_5",
            name="helper",
            entity_type=EntityType.FUNCTION,
            file_path=Path("a.py"),
            start_line=5,
            end_line=7,
            properties={},
        )
        file_contents = {
            Path("a.py"): "def main():\n    helper()\n    return\ndef helper():\n    pass\n"
        }
        mapper = RelationshipMapper()
        rels = await mapper.map_relationships([caller, callee], file_contents)
        call_rels = [r for r in rels if r.relation_type == RelationType.CALLS]
        assert len(call_rels) >= 1

    async def test_map_import_relationships(self):
        imp = CodeEntity(
            id="import_os_1",
            name="os",
            entity_type=EntityType.IMPORT,
            file_path=Path("a.py"),
            start_line=1,
            end_line=1,
            properties={},
        )
        func = CodeEntity(
            id="func_main_3",
            name="main",
            entity_type=EntityType.FUNCTION,
            file_path=Path("a.py"),
            start_line=3,
            end_line=5,
            properties={},
        )
        mapper = RelationshipMapper()
        rels = await mapper.map_relationships([imp, func], {})
        import_rels = [r for r in rels if r.relation_type == RelationType.IMPORTS]
        assert len(import_rels) >= 1
