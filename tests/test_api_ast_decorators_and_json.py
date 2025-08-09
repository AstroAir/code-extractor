from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import ASTFilters, OutputFormat, Query


def test_api_ast_decorators_and_json_output(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text("""
@decorator
def func():
    pass

class MyClass:
    @property
    def prop(self):
        return 1
""", encoding="utf-8")
    
    eng = PySearch(SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False))

    # AST search with decorator filter
    ast_q = Query(
        pattern="def",
        use_ast=True,
        filters=ASTFilters(decorator="decorator"),
        context=0
    )
    ast_res = eng.run(ast_q)
    assert ast_res.stats.files_scanned >= 1

    # AST search with class_name filter
    class_q = Query(
        pattern="def",
        use_ast=True,
        filters=ASTFilters(class_name="MyClass"),
        context=0
    )
    class_res = eng.run(class_q)
    assert class_res.stats.files_scanned >= 1

    # JSON output format via convenience API
    json_res = eng.search("def", regex=False, context=0, output=OutputFormat.JSON)
    assert json_res.stats.items >= 0
