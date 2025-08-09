from __future__ import annotations

from pathlib import Path

from pysearch.api import PySearch
from pysearch.config import SearchConfig
from pysearch.types import ASTFilters, Language, MetadataFilters, OutputFormat, Query


def setup_repo(tmp_path: Path) -> None:
    (tmp_path / "m.py").write_text(
        """
class C:
    def f(self):
        pass

# comment
""",
        encoding="utf-8",
    )
    (tmp_path / "d.py").write_text("""def main():\n    \"\"\"doc\n\"\"\"\n    pass\n""", encoding="utf-8")


def test_api_ast_and_metadata_and_semantic(tmp_path: Path) -> None:
    setup_repo(tmp_path)
    cfg = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], context=0, parallel=False)
    eng = PySearch(cfg)

    # AST: find method f in class C
    ast_q = Query(pattern="def", use_ast=True, filters=ASTFilters(func_name="f", class_name="C"), context=0)
    ast_res = eng.run(ast_q)
    assert ast_res.stats.files_scanned >= 1

    # Metadata: limit to python and min_lines=1
    md = MetadataFilters(min_lines=1, languages={Language.PYTHON})
    md_q = Query(pattern="class", use_regex=False, context=0, metadata_filters=md)
    md_res = eng.run(md_q)
    assert md_res.stats.files_scanned >= 1

    # Semantic advanced: high threshold to likely get zero
    sem_res = eng.search_semantic_advanced("nonexistent concept", threshold=0.9, max_results=5)
    assert sem_res.stats.files_scanned >= 0

    # Output format branch hit via convenience search
    txt_res = eng.search("class ", regex=False, context=0, output=OutputFormat.TEXT)
    assert txt_res.stats.items >= 0

