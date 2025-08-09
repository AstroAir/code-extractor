import json
import subprocess
import sys
from pathlib import Path


def run_cli(args, cwd: Path):
    exe = [sys.executable, "-m", "pysearch"] + args
    proc = subprocess.run(exe, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def test_cli_find_text(tmp_path: Path):
    pkg = tmp_path / "pkg"
    (pkg / "src").mkdir(parents=True, exist_ok=True)
    (pkg / "pyproject.toml").write_text(
        """
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "tmp-pkg"
version = "0.0.0"
        """,
        encoding="utf-8",
    )
    mod = pkg / "src" / "m.py"
    mod.write_text("def hello():\n    return 1\n", encoding="utf-8")

    # invoke CLI as module
    code, out, err = run_cli(
        [
            "find",
            "--path",
            str(pkg / "src"),
            "--include",
            "**/*.py",
            "--context",
            "0",
            "--format",
            "json",
            "hello",
        ],
        cwd=tmp_path,
    )

    assert code == 0
    # ensure json is parseable and contains at least one item
    data = json.loads(out)
    assert "items" in data and isinstance(data["items"], list)
    assert any("hello" in "".join(it["lines"]) for it in data["items"])
