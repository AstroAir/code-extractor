from __future__ import annotations

from pathlib import Path
import time

from pysearch.config import SearchConfig
from pysearch.indexer import Indexer


def write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_indexer_lazy_vs_strict(tmp_path: Path) -> None:
    # Prepare small repo
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    write(f1, "print('a')\n")
    write(f2, "print('b')\n")

    # LAZY (strict_hash_check = False)
    cfg_lazy = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[], context=1)
    cfg_lazy.strict_hash_check = False
    idx_lazy = Indexer(cfg_lazy)

    changed1, removed1, total1 = idx_lazy.scan()
    idx_lazy.save()
    assert total1 == 2
    # first scan should mark both as changed (no prior index)
    assert set(changed1) == {f1, f2}
    assert removed1 == []

    # second scan without changes should mark none
    changed2, removed2, total2 = idx_lazy.scan()
    idx_lazy.save()
    assert total2 == 2
    assert changed2 == []
    assert removed2 == []

    # STRICT (strict_hash_check = True)
    cfg_strict = SearchConfig(paths=[str(tmp_path)], include=["**/*.py"], exclude=[], context=1)
    cfg_strict.strict_hash_check = True
    idx_strict = Indexer(cfg_strict)

    changed3, removed3, total3 = idx_strict.scan()
    idx_strict.save()
    assert total3 == 2
    assert set(changed3) == {f1, f2}
    assert removed3 == []

    # touch a.py with same content but update mtime (simulate editor touching file)
    old = f1.read_text(encoding="utf-8")
    time.sleep(0.01)  # ensure mtime progresses across platforms
    write(f1, old)  # same content

    # LAZY: mtime changed -> should be considered changed
    changed4, removed4, total4 = idx_lazy.scan()
    idx_lazy.save()
    assert f1 in changed4

    # STRICT: mtime changed but content same -> sha1 equal => not changed
    changed5, removed5, total5 = idx_strict.scan()
    idx_strict.save()
    assert f1 not in changed5

    # Modify content
    time.sleep(0.01)
    write(f2, "print('b2')\n")

    # STRICT: now should detect change for b.py
    changed6, removed6, total6 = idx_strict.scan()
    idx_strict.save()
    assert f2 in changed6