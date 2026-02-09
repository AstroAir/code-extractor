#!/usr/bin/env python3
"""
Cross-platform task runner for PySearch project.

Usage (with uv - recommended):
    uv run scripts/tasks.py <command> [options]

Usage (without uv):
    python scripts/tasks.py <command> [options]

Commands:
    dev-install    Set up development environment
    lint           Run linting checks (ruff + black + mypy)
    format         Auto-format code (black + ruff --fix)
    test           Run tests with optional coverage and markers
    bench          Run performance benchmarks
    docs           Build, serve, check, or deploy documentation
    mcp            Run MCP server
    validate       Validate project structure and imports
    clean          Clean build/cache artifacts
    release        Build distribution packages
    help           Show this help message

Examples:
    uv run scripts/tasks.py dev-install
    uv run scripts/tasks.py lint
    uv run scripts/tasks.py test --coverage --markers unit
    uv run scripts/tasks.py docs serve
    uv run scripts/tasks.py clean
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"
DOCS_DIR = PROJECT_ROOT / "docs"
MCP_DIR = PROJECT_ROOT / "mcp"
BUILD_DIR = PROJECT_ROOT / "site"

IS_WINDOWS = platform.system() == "Windows"
IS_CI = os.environ.get("CI", "").lower() in ("true", "1", "yes")

# Directories / files to clean
CLEAN_DIRS = [
    "build",
    "dist",
    "sdist",
    "wheels",
    ".pytest_cache",
    ".mypy_cache",
    ".mypy_cache_temp",
    ".ruff_cache",
    ".coverage",
    "htmlcov",
    ".benchmarks",
    ".pysearch-cache",
    "site",
    ".mkdocs_cache",
    "docs_build",
    ".docstring_cache",
    ".api_docs_cache",
]
CLEAN_FILES = ["coverage.xml"]
CLEAN_PATTERNS_DIR = ["__pycache__"]
CLEAN_GLOB_DIR = ["src/*.egg-info"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Colors:
    """ANSI color helpers with automatic Windows compatibility."""

    _enabled: bool | None = None

    @classmethod
    def _is_enabled(cls) -> bool:
        if cls._enabled is not None:
            return cls._enabled
        # Respect NO_COLOR convention (https://no-color.org/)
        if os.environ.get("NO_COLOR"):
            cls._enabled = False
            return False
        # Force color in CI
        if os.environ.get("FORCE_COLOR"):
            cls._enabled = True
            return True
        # Check if stdout is a terminal
        cls._enabled = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        return cls._enabled

    @classmethod
    def _wrap(cls, code: str, text: str) -> str:
        if not cls._is_enabled():
            return text
        return f"\033[{code}m{text}\033[0m"

    @classmethod
    def red(cls, text: str) -> str:
        return cls._wrap("0;31", text)

    @classmethod
    def green(cls, text: str) -> str:
        return cls._wrap("0;32", text)

    @classmethod
    def yellow(cls, text: str) -> str:
        return cls._wrap("1;33", text)

    @classmethod
    def blue(cls, text: str) -> str:
        return cls._wrap("0;34", text)

    @classmethod
    def bold(cls, text: str) -> str:
        return cls._wrap("1", text)


def info(msg: str) -> None:
    print(f"{Colors.blue('INFO')}  {msg}")


def success(msg: str) -> None:
    print(f"{Colors.green('OK')}    {msg}")


def warning(msg: str) -> None:
    print(f"{Colors.yellow('WARN')}  {msg}", file=sys.stderr)


def error(msg: str) -> None:
    print(f"{Colors.red('ERR')}   {msg}", file=sys.stderr)


def run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with consistent settings."""
    merged_env = {**os.environ, **(env or {})}
    if cwd is None:
        cwd = PROJECT_ROOT
    info(f"Running: {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=check,
        capture_output=capture,
        text=True,
        env=merged_env,
    )


def python() -> str:
    """Return the current Python executable."""
    return sys.executable


def find_tool(name: str) -> str | None:
    """Find a tool in PATH or current venv."""
    return shutil.which(name)


def has_uv() -> bool:
    """Check if uv is available."""
    return find_tool("uv") is not None


def ensure_in_project_root() -> None:
    """Ensure we are working from the project root."""
    if not (PROJECT_ROOT / "pyproject.toml").exists():
        error("pyproject.toml not found. Are you in the project root?")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_dev_install(args: argparse.Namespace) -> None:
    """Set up the development environment."""
    ensure_in_project_root()

    info("Setting up PySearch development environment...")

    use_uv = has_uv() and not args.no_uv
    if use_uv:
        info("Using uv for faster installation")

    # Check/create virtual environment
    venv_dir = PROJECT_ROOT / ".venv"
    if not venv_dir.exists() and not args.no_venv:
        info("Creating virtual environment...")
        if use_uv:
            run([find_tool("uv"), "venv", str(venv_dir)])
        else:
            run([python(), "-m", "venv", str(venv_dir)])
        success("Virtual environment created at .venv/")
        warning("Activate it before continuing:")
        if IS_WINDOWS:
            warning("  .venv\\Scripts\\activate")
        else:
            warning("  source .venv/bin/activate")
        if not args.force:
            info("Re-run this command after activating, or use --force to continue anyway")
            return
    elif venv_dir.exists():
        info("Virtual environment already exists at .venv/")

    # Install package and dependencies
    if use_uv:
        run([find_tool("uv"), "pip", "install", "-U", "pip"])
        run([find_tool("uv"), "pip", "install", "-e", "."])
        run([find_tool("uv"), "pip", "install", "-e", ".[dev]"])
    else:
        run([python(), "-m", "pip", "install", "-U", "pip"])
        run([python(), "-m", "pip", "install", "-e", "."])
        run([python(), "-m", "pip", "install", "-e", ".[dev]"])

    # Install pre-commit hooks
    if find_tool("pre-commit"):
        info("Installing pre-commit hooks...")
        run([find_tool("pre-commit"), "install"])
    else:
        info("Installing pre-commit...")
        if use_uv:
            run([find_tool("uv"), "pip", "install", "pre-commit"])
        else:
            run([python(), "-m", "pip", "install", "pre-commit"])
        run([python(), "-m", "pre_commit", "install"])

    # Validate installation
    info("Validating installation...")
    result = run(
        [python(), "-c", "import pysearch; print(f'PySearch version: {pysearch.__version__}')"],
        check=False,
    )
    if result.returncode != 0:
        warning("PySearch import validation failed (may be expected on first install)")
    else:
        success("PySearch imports successfully")

    result = run(
        [
            python(),
            "-c",
            "from mcp.servers import pysearch_mcp_server; print('MCP server available')",
        ],
        check=False,
    )
    if result.returncode == 0:
        success("MCP server imports successfully")

    success("Development environment setup complete!")
    print()
    print("Next steps:")
    print("  python scripts/tasks.py test        # Run tests")
    print("  python scripts/tasks.py lint        # Run linting")
    print("  python scripts/tasks.py validate    # Validate project")
    print("  python scripts/tasks.py docs serve  # Serve docs locally")


def cmd_lint(args: argparse.Namespace) -> None:
    """Run linting checks."""
    ensure_in_project_root()
    info("Running linting checks...")
    failed = False

    # Ruff check
    info("Running ruff check...")
    result = run([python(), "-m", "ruff", "check", "."], check=False)
    if result.returncode != 0:
        error("Ruff check failed")
        failed = True
    else:
        success("Ruff check passed")

    # Black format check
    info("Running black format check...")
    result = run([python(), "-m", "black", "--check", "."], check=False)
    if result.returncode != 0:
        error("Black format check failed")
        failed = True
    else:
        success("Black format check passed")

    # Mypy type check (only on configured paths from pyproject.toml)
    if not args.skip_mypy:
        info("Running mypy type check...")
        result = run([python(), "-m", "mypy"], check=False)
        if result.returncode != 0:
            error("Mypy type check failed")
            failed = True
        else:
            success("Mypy type check passed")

    if failed:
        error("Linting checks failed")
        sys.exit(1)
    else:
        success("All linting checks passed")


def cmd_format(args: argparse.Namespace) -> None:
    """Auto-format code."""
    ensure_in_project_root()
    info("Formatting code...")

    if args.check:
        # Dry-run mode: just check, don't modify
        info("Running in check mode (no changes will be made)...")
        failed = False
        result = run([python(), "-m", "black", "--check", "--diff", "."], check=False)
        if result.returncode != 0:
            failed = True
        result = run([python(), "-m", "ruff", "check", "--diff", "."], check=False)
        if result.returncode != 0:
            failed = True
        if failed:
            error("Code is not properly formatted")
            sys.exit(1)
        else:
            success("Code is properly formatted")
        return

    # Run black
    run([python(), "-m", "black", "."])

    # Run ruff --fix
    run([python(), "-m", "ruff", "check", ".", "--fix"])

    # Run ruff format (for anything black missed)
    run([python(), "-m", "ruff", "format", "."], check=False)

    success("Code formatted successfully")


def cmd_test(args: argparse.Namespace) -> None:
    """Run tests."""
    ensure_in_project_root()
    info("Running tests...")

    cmd = [python(), "-m", "pytest"]

    # Coverage
    if args.coverage:
        cmd.extend(
            [
                "--cov=src/pysearch",
                "--cov-report=term-missing",
            ]
        )
        if args.coverage_xml:
            cmd.append("--cov-report=xml")
        if args.coverage_html:
            cmd.append("--cov-report=html")

    # Markers
    if args.markers:
        cmd.extend(["-m", args.markers])

    # Keyword filter
    if args.keyword:
        cmd.extend(["-k", args.keyword])

    # Verbosity
    if args.verbose:
        cmd.append("-v")

    # Fail fast
    if args.failfast:
        cmd.extend(["--maxfail=1", "-x"])

    # Specific test path
    if args.path:
        cmd.append(args.path)

    # Pass extra args
    if args.extra:
        cmd.extend(args.extra)

    result = run(cmd, check=False)
    if result.returncode != 0:
        error("Tests failed")
        sys.exit(result.returncode)
    success("Tests passed")


def cmd_bench(args: argparse.Namespace) -> None:
    """Run performance benchmarks."""
    ensure_in_project_root()
    info("Running performance benchmarks...")

    cmd = [python(), "-m", "pytest", "-q", "-k", "benchmark"]

    if args.compare:
        cmd.extend(["--benchmark-compare"])
    if args.save:
        cmd.extend(["--benchmark-save", args.save])

    if args.extra:
        cmd.extend(args.extra)

    result = run(cmd, check=False)
    if result.returncode != 0:
        error("Benchmarks failed")
        sys.exit(result.returncode)
    success("Benchmarks completed")


def cmd_docs(args: argparse.Namespace) -> None:
    """Build, serve, check, or deploy documentation."""
    ensure_in_project_root()

    sub = args.docs_command or "build"

    # Check dependencies
    if sub != "clean":
        _check_docs_deps()

    if sub == "build":
        _docs_validate_structure()
        _docs_clean()
        _docs_build()
        success("Documentation built successfully!")
        info(f"Built documentation is available in: {BUILD_DIR}/")

    elif sub == "serve":
        _docs_validate_structure()
        info("Starting documentation server...")
        info("Documentation will be available at http://localhost:8000")
        info("Press Ctrl+C to stop the server")

        cmd = [
            python(),
            "-m",
            "mkdocs",
            "serve",
            "--dev-addr=localhost:8000",
        ]
        if args.watch_src:
            cmd.extend(["--watch", "src/pysearch"])
        run(cmd, check=False)

    elif sub == "clean":
        _docs_clean()
        success("Documentation artifacts cleaned")

    elif sub == "check":
        _docs_validate_structure()
        _docs_clean()
        _docs_build(strict=True, verbose=True)
        _docs_check_links()
        success("Documentation validation completed!")

    elif sub == "deploy":
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            error("GITHUB_TOKEN environment variable required for deployment")
            sys.exit(1)
        _docs_validate_structure()
        _docs_clean()
        _docs_build(strict=True)
        _docs_check_links()
        run(
            [
                python(),
                "-m",
                "mkdocs",
                "gh-deploy",
                "--clean",
                "--message",
                "Deploy documentation [skip ci]",
            ]
        )
        success("Documentation deployed to GitHub Pages")
    else:
        error(f"Unknown docs command: {sub}")
        sys.exit(1)


def _check_docs_deps() -> None:
    """Check documentation dependencies are installed."""
    info("Checking documentation dependencies...")
    result = run(
        [python(), "-c", "import mkdocs; import mkdocstrings"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        error("mkdocs or mkdocstrings not found. Install with: pip install -e '.[dev]'")
        sys.exit(1)
    success("Documentation dependencies found")


def _docs_validate_structure() -> None:
    """Validate documentation structure against actual file layout."""
    info("Validating documentation structure...")

    required_files = [
        DOCS_DIR / "index.md",
        DOCS_DIR / "getting-started" / "installation.md",
        DOCS_DIR / "guide" / "usage.md",
        DOCS_DIR / "guide" / "configuration.md",
        PROJECT_ROOT / "mkdocs.yml",
    ]

    required_dirs = [
        DOCS_DIR / "api",
    ]

    all_ok = True
    for f in required_files:
        if not f.exists():
            error(f"Required file missing: {f.relative_to(PROJECT_ROOT)}")
            all_ok = False

    for d in required_dirs:
        if not d.exists():
            error(f"Required directory missing: {d.relative_to(PROJECT_ROOT)}")
            all_ok = False

    if not all_ok:
        sys.exit(1)
    success("Documentation structure validated")


def _docs_clean() -> None:
    """Clean documentation build artifacts."""
    for d in ["site", ".mkdocs_cache", "docs_build", ".docstring_cache", ".api_docs_cache"]:
        p = PROJECT_ROOT / d
        if p.exists():
            shutil.rmtree(p)


def _docs_build(*, strict: bool = True, verbose: bool = False) -> None:
    """Build documentation."""
    info("Building documentation...")
    cmd = [python(), "-m", "mkdocs", "build", "--clean"]
    if strict:
        cmd.append("--strict")
    if verbose:
        cmd.append("--verbose")
    run(cmd)


def _docs_check_links() -> None:
    """Check for broken links in built docs."""
    if find_tool("linkchecker"):
        info("Checking for broken links...")
        index = BUILD_DIR / "index.html"
        if index.exists():
            result = run(
                [find_tool("linkchecker"), str(index), "--check-extern"],
                check=False,
            )
            if result.returncode != 0:
                warning("Some links may be broken (check output above)")
            else:
                success("All links are valid")
        else:
            warning("Built site index.html not found, skipping link check")
    else:
        warning("linkchecker not installed, skipping link check")
        info("Install with: pip install linkchecker")


def cmd_mcp(args: argparse.Namespace) -> None:
    """Run MCP server."""
    ensure_in_project_root()

    server_path = MCP_DIR / "servers" / "pysearch_mcp_server.py"

    if not server_path.exists():
        error(f"MCP server not found: {server_path.relative_to(PROJECT_ROOT)}")
        sys.exit(1)

    transport = args.transport or "stdio"

    info(f"Starting PySearch MCP Server (transport: {transport})")
    info(f"Server path: {server_path.relative_to(PROJECT_ROOT)}")

    cmd = [python(), str(server_path)]
    if transport != "stdio":
        cmd.extend(["--transport", transport])
        if args.host:
            cmd.extend(["--host", args.host])
        if args.port:
            cmd.extend(["--port", str(args.port)])

    run(cmd, check=False)


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate project structure, imports, and optionally run full checks."""
    ensure_in_project_root()
    info("Validating PySearch project...")
    failed = False

    # 1. Check project structure
    info("Checking project structure...")
    required_dirs = [
        SRC_DIR / "pysearch",
        MCP_DIR / "servers",
        MCP_DIR / "shared",
        TESTS_DIR,
        DOCS_DIR,
        SCRIPT_DIR,
    ]
    required_files = [
        PROJECT_ROOT / "pyproject.toml",
        PROJECT_ROOT / "README.md",
        MCP_DIR / "README.md",
    ]
    for d in required_dirs:
        if not d.exists():
            error(f"Required directory missing: {d.relative_to(PROJECT_ROOT)}")
            failed = True
    for f in required_files:
        if not f.exists():
            error(f"Required file missing: {f.relative_to(PROJECT_ROOT)}")
            failed = True
    if not failed:
        success("Project structure validation passed")

    # 2. Check Python syntax
    info("Checking Python syntax...")
    py_files = list((SRC_DIR / "pysearch").glob("*.py"))
    syntax_ok = True
    for pf in py_files:
        result = run(
            [python(), "-m", "py_compile", str(pf)],
            check=False,
            capture=True,
        )
        if result.returncode != 0:
            error(f"Syntax error in {pf.relative_to(PROJECT_ROOT)}")
            syntax_ok = False
            failed = True
    if syntax_ok:
        success("Python syntax check passed")

    # 3. Check imports
    info("Checking package imports...")
    result = run(
        [python(), "-c", "import pysearch; print(f'Core: {pysearch.__version__}')"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        error("Core package import failed")
        failed = True
    else:
        success(f"Core package import passed ({result.stdout.strip()})")

    result = run(
        [python(), "-c", "from mcp.servers import pysearch_mcp_server; print('MCP OK')"],
        check=False,
        capture=True,
    )
    if result.returncode != 0:
        error("MCP server import failed")
        failed = True
    else:
        success("MCP server import passed")

    # 4. Optionally run full checks
    if args.full:
        info("Running full validation (lint + type + test)...")

        info("Running linting...")
        r = run([python(), "-m", "ruff", "check", "."], check=False)
        if r.returncode != 0:
            failed = True

        r = run([python(), "-m", "black", "--check", "."], check=False)
        if r.returncode != 0:
            failed = True

        info("Running type checking...")
        r = run([python(), "-m", "mypy"], check=False)
        if r.returncode != 0:
            failed = True

        info("Running tests...")
        r = run([python(), "-m", "pytest", "-q"], check=False)
        if r.returncode != 0:
            failed = True

    # 5. Check documentation
    if not args.skip_docs:
        info("Checking documentation...")
        mkdocs_yml = PROJECT_ROOT / "mkdocs.yml"
        if mkdocs_yml.exists():
            result = run(
                [python(), "-m", "mkdocs", "build", "--quiet"],
                check=False,
                capture=True,
            )
            if result.returncode != 0:
                warning("Documentation build failed (non-critical)")
            else:
                success("Documentation build passed")
        else:
            warning("mkdocs.yml not found, skipping documentation check")

    if failed:
        print()
        error("Validation failed! See errors above.")
        sys.exit(1)
    else:
        print()
        success("All validation checks passed!")
        print("Project is ready for development and deployment.")


def cmd_clean(args: argparse.Namespace) -> None:
    """Clean build, cache, and temporary artifacts."""
    ensure_in_project_root()
    info("Cleaning build and cache artifacts...")
    removed_count = 0

    # Remove known directories
    for dirname in CLEAN_DIRS:
        p = PROJECT_ROOT / dirname
        if p.exists():
            shutil.rmtree(p)
            info(f"  Removed {dirname}/")
            removed_count += 1

    # Remove known files
    for filename in CLEAN_FILES:
        p = PROJECT_ROOT / filename
        if p.exists():
            p.unlink()
            info(f"  Removed {filename}")
            removed_count += 1

    # Remove __pycache__ directories recursively
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        if pycache.is_dir():
            shutil.rmtree(pycache)
            removed_count += 1
    info("  Removed __pycache__ directories")

    # Remove egg-info directories
    for pattern in CLEAN_GLOB_DIR:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_dir():
                shutil.rmtree(p)
                info(f"  Removed {p.relative_to(PROJECT_ROOT)}/")
                removed_count += 1

    # Remove .pyc files
    if args.deep:
        pyc_count = 0
        for pyc in PROJECT_ROOT.rglob("*.pyc"):
            pyc.unlink()
            pyc_count += 1
        if pyc_count:
            info(f"  Removed {pyc_count} .pyc files")
            removed_count += pyc_count

    success(f"Cleaned {removed_count} items")


def cmd_release(args: argparse.Namespace) -> None:
    """Build distribution packages and optionally check them."""
    ensure_in_project_root()
    info("Building release artifacts...")

    # Install build tooling
    run([python(), "-m", "pip", "install", "-U", "build", "twine"])

    # Clean previous builds
    for d in ["dist", "build"]:
        p = PROJECT_ROOT / d
        if p.exists():
            shutil.rmtree(p)

    # Build
    run([python(), "-m", "build"])

    # Check
    run([python(), "-m", "twine", "check", "dist/*"])

    success("Release artifacts built and validated!")
    print()
    info("To upload to PyPI:")
    print("  TWINE_USERNAME=__token__ TWINE_PASSWORD=<token> twine upload dist/*")

    if args.publish:
        warning("Publishing to PyPI...")
        run([python(), "-m", "twine", "upload", "dist/*"])


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tasks",
        description="Cross-platform task runner for PySearch project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # dev-install
    p = subparsers.add_parser("dev-install", help="Set up development environment")
    p.add_argument("--no-uv", action="store_true", help="Don't use uv even if available")
    p.add_argument("--no-venv", action="store_true", help="Skip virtual environment creation")
    p.add_argument("--force", action="store_true", help="Continue even if venv was just created")

    # lint
    p = subparsers.add_parser("lint", help="Run linting checks")
    p.add_argument("--skip-mypy", action="store_true", help="Skip mypy type checking")

    # format
    p = subparsers.add_parser("format", help="Auto-format code")
    p.add_argument("--check", action="store_true", help="Check only, don't modify files (dry-run)")

    # test
    p = subparsers.add_parser("test", help="Run tests")
    p.add_argument("--coverage", action="store_true", help="Enable coverage reporting")
    p.add_argument("--coverage-xml", action="store_true", help="Generate XML coverage report")
    p.add_argument("--coverage-html", action="store_true", help="Generate HTML coverage report")
    p.add_argument("--markers", "-m", help="Run tests matching marker expression")
    p.add_argument("--keyword", "-k", help="Run tests matching keyword expression")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    p.add_argument("--failfast", "-x", action="store_true", help="Stop on first failure")
    p.add_argument("--path", "-p", help="Specific test file or directory")
    p.add_argument("extra", nargs="*", help="Extra arguments passed to pytest")

    # bench
    p = subparsers.add_parser("bench", help="Run performance benchmarks")
    p.add_argument("--compare", action="store_true", help="Compare with previous benchmark")
    p.add_argument("--save", help="Save benchmark results with this name")
    p.add_argument("extra", nargs="*", help="Extra arguments passed to pytest")

    # docs
    p = subparsers.add_parser("docs", help="Build, serve, check, or deploy documentation")
    p.add_argument(
        "docs_command",
        nargs="?",
        choices=["build", "serve", "clean", "check", "deploy"],
        default="build",
        help="Documentation sub-command (default: build)",
    )
    p.add_argument("--watch-src", action="store_true", help="Also watch src/ for changes (serve)")

    # mcp
    p = subparsers.add_parser("mcp", help="Run MCP server")
    p.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport protocol (default: stdio)",
    )
    p.add_argument("--host", default="127.0.0.1", help="Host for HTTP transport")
    p.add_argument("--port", type=int, default=9000, help="Port for HTTP transport")

    # validate
    p = subparsers.add_parser("validate", help="Validate project structure and imports")
    p.add_argument("--full", action="store_true", help="Run full validation (lint + type + test)")
    p.add_argument("--skip-docs", action="store_true", help="Skip documentation check")

    # clean
    p = subparsers.add_parser("clean", help="Clean build/cache artifacts")
    p.add_argument("--deep", action="store_true", help="Also remove all .pyc files")

    # release
    p = subparsers.add_parser("release", help="Build distribution packages")
    p.add_argument("--publish", action="store_true", help="Also publish to PyPI")

    # help (show detailed help)
    subparsers.add_parser("help", help="Show detailed help")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


COMMANDS = {
    "dev-install": cmd_dev_install,
    "lint": cmd_lint,
    "format": cmd_format,
    "test": cmd_test,
    "bench": cmd_bench,
    "docs": cmd_docs,
    "mcp": cmd_mcp,
    "validate": cmd_validate,
    "clean": cmd_clean,
    "release": cmd_release,
}


def main() -> None:
    # Enable ANSI on Windows
    if IS_WINDOWS:
        os.system("")  # Enables ANSI escape sequences on Windows 10+

    parser = build_parser()
    args = parser.parse_args()

    if not args.command or args.command == "help":
        parser.print_help()
        print()
        print("Available commands:")
        print(f"  {'dev-install':<15s}  Set up development environment (venv + deps + hooks)")
        print(f"  {'lint':<15s}  Run ruff + black + mypy checks")
        print(f"  {'format':<15s}  Auto-format code with black + ruff")
        print(f"  {'test':<15s}  Run tests with optional coverage/markers")
        print(f"  {'bench':<15s}  Run performance benchmarks")
        print(f"  {'docs':<15s}  Build/serve/check/deploy documentation")
        print(f"  {'mcp':<15s}  Run MCP server")
        print(f"  {'validate':<15s}  Validate project structure and imports")
        print(f"  {'clean':<15s}  Clean build/cache artifacts")
        print(f"  {'release':<15s}  Build distribution packages")
        print()
        if has_uv():
            print(Colors.green("uv detected") + " - recommended usage:")
            print("  uv run scripts/tasks.py <command> [options]")
        else:
            print("Tip: install uv for faster operations:")
            print("  curl -LsSf https://astral.sh/uv/install.sh | sh  # Unix")
            print('  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows')
        sys.exit(0)

    handler = COMMANDS.get(args.command)
    if handler is None:
        error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except subprocess.CalledProcessError as e:
        error(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print()
        warning("Interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
