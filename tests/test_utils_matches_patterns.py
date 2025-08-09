from __future__ import annotations

from pathlib import Path

from pysearch.utils import matches_patterns


def test_matches_patterns_basic(tmp_path: Path) -> None:
    p = tmp_path / "src" / "a.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("print('x')\n", encoding="utf-8")

    assert matches_patterns(p, ["**/*.py"]) is True
    assert matches_patterns(p, ["**/*.txt"]) is False


def test_matches_patterns_exclude_dirs(tmp_path: Path) -> None:
    d = tmp_path / ".git" / "objects"
    d.mkdir(parents=True, exist_ok=True)
    f = d / "x"
    f.write_text("", encoding="utf-8")

    assert matches_patterns(f, ["**/.git/**"]) is True

