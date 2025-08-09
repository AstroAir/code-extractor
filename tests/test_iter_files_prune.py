from __future__ import annotations

from pathlib import Path

from pysearch.utils import iter_files


def make_tree(tmp: Path) -> None:
    # layout:
    # tmp/
    #   src/a.py
    #   src/b.txt
    #   .venv/lib/site.py
    #   .git/config
    #   pkg/__pycache__/c.pyc
    (tmp / "src").mkdir(parents=True, exist_ok=True)
    (tmp / "src" / "a.py").write_text("print('a')\n", encoding="utf-8")
    (tmp / "src" / "b.txt").write_text("not python\n", encoding="utf-8")

    (tmp / ".venv" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp / ".venv" / "lib" / "site.py").write_text("print('env')\n", encoding="utf-8")

    (tmp / ".git").mkdir(parents=True, exist_ok=True)
    (tmp / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    (tmp / "pkg" / "__pycache__").mkdir(parents=True, exist_ok=True)
    (tmp / "pkg" / "__pycache__" / "c.pyc").write_text("", encoding="utf-8")


def test_iter_files_prune_excluded_dirs(tmp_path: Path) -> None:
    make_tree(tmp_path)
    roots = [str(tmp_path)]
    include = ["**/*.py", "**/*.txt", "**/*"]
    exclude = ["**/.venv/**", "**/.git/**", "**/__pycache__/**"]
    # prune on
    files = list(
        iter_files(
            roots=roots,
            include=include,
            exclude=exclude,
            follow_symlinks=False,
            prune_excluded_dirs=True,
        )
    )
    rels = {str(Path(p).resolve().relative_to(tmp_path)) for p in files}
    # .venv/.git/__pycache__ should be pruned entirely
    assert "src/a.py" in rels
    assert "src/b.txt" in rels
    assert not any(p.startswith(".venv") for p in rels)
    assert not any(p.startswith(".git") for p in rels)
    assert not any("__pycache__" in p for p in rels)

    # prune off should still filter by exclude at file match time
    files2 = list(
        iter_files(
            roots=roots,
            include=include,
            exclude=exclude,
            follow_symlinks=False,
            prune_excluded_dirs=False,
        )
    )
    rels2 = {str(Path(p).resolve().relative_to(tmp_path)) for p in files2}
    # Even if walk enters excluded dirs, match_file will filter files out
    assert "src/a.py" in rels2
    assert "src/b.txt" in rels2
    assert not any(p.endswith("site.py") for p in rels2)
    assert not any(p.endswith("config") for p in rels2)
    assert not any(p.endswith(".pyc") for p in rels2)
