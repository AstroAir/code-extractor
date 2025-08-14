from __future__ import annotations

from pathlib import Path

from pysearch.analysis.language_detection import (
    detect_language,
    detect_language_by_content,
    detect_language_by_extension,
    detect_language_by_filename,
    detect_language_by_shebang,
    is_text_file,
)
from pysearch import Language


def test_language_detection_by_extension_and_filename(tmp_path: Path) -> None:
    assert detect_language_by_extension(Path("a.py")) == Language.PYTHON
    assert detect_language_by_filename(Path("Dockerfile")) == Language.DOCKERFILE


def test_language_detection_shebang_and_content() -> None:
    content = "#!/usr/bin/env python\nprint('x')\n"
    assert detect_language_by_shebang(content) == Language.PYTHON

    content_js = "function f() {}\nexport const x = 1;\n"
    lang = detect_language_by_content(content_js, {Language.JAVASCRIPT, Language.PYTHON})
    assert lang == Language.JAVASCRIPT


def test_language_detection_comprehensive(tmp_path: Path) -> None:
    p = tmp_path / "script.sh"
    p.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")
    lang = detect_language(p, p.read_text(encoding="utf-8"))
    assert lang == Language.SHELL

    assert is_text_file(Path("file.txt"))

