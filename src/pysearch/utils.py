"""
Utility functions and helpers for pysearch.

This module provides essential utility functions used throughout the PySearch system,
including file operations, text processing, AST utilities, and path management.
The utilities are designed to be efficient, safe, and handle edge cases gracefully.

Key Functions:
    File Operations:
        - read_text_safely: Safe file reading with encoding detection and size limits
        - file_sha1: Efficient SHA1 hash computation for files
        - create_file_metadata: Comprehensive file metadata extraction

    Text Processing:
        - split_lines_keepends: Line splitting preserving line endings
        - extract_context: Context extraction around specific lines
        - highlight_spans: Text highlighting with custom markers

    AST Utilities:
        - iter_python_ast_nodes: Efficient AST node iteration
        - get_ast_node_info: Extract information from AST nodes

    Path Utilities:
        - iter_files_prune: Efficient file iteration with directory pruning
        - resolve_path_patterns: Pattern-based path resolution

Example:
    Basic file operations:
        >>> from pysearch.utils import read_text_safely, file_sha1
        >>> from pathlib import Path
        >>>
        >>> # Safe file reading
        >>> content = read_text_safely(Path("example.py"), max_bytes=1_000_000)
        >>> if content:
        ...     print(f"File has {len(content)} characters")
        >>>
        >>> # File hashing
        >>> hash_value = file_sha1(Path("example.py"))
        >>> print(f"SHA1: {hash_value}")

    Text processing:
        >>> from pysearch.utils import split_lines_keepends, highlight_spans
        >>>
        >>> # Line splitting
        >>> lines = split_lines_keepends("line1\\nline2\\nline3\\n")
        >>> print(lines)  # ['line1\\n', 'line2\\n', 'line3\\n']
        >>>
        >>> # Text highlighting
        >>> highlighted = highlight_spans("def main():", [(0, 3), (4, 8)])
        >>> print(highlighted)  # Highlighted "def" and "main"
"""

from __future__ import annotations

import ast
import hashlib
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

from .language_detection import detect_language, is_text_file
from .types import FileMetadata


@dataclass(slots=True)
class FileMeta:
    """
    Lightweight file metadata container.

    This class holds essential file metadata used for caching and change detection.
    It's optimized for performance with slots and minimal memory footprint.

    Attributes:
        path: Path to the file
        size: File size in bytes
        mtime: Last modification time as Unix timestamp
        sha1: Optional SHA1 hash of file content
    """

    path: Path
    size: int
    mtime: float
    sha1: str | None = None


def sha1_bytes(data: bytes) -> str:
    """
    Compute SHA1 hash of byte data.

    Args:
        data: Byte data to hash

    Returns:
        Hexadecimal SHA1 hash string

    Example:
        >>> sha1_bytes(b"hello world")
        '2aae6c35c94fcfb415dbe95f408b9ce91ee846ed'
    """
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    """
    Compute SHA1 hash of file content using chunked reading.

    This function reads files in chunks to avoid loading large files entirely
    into memory, making it suitable for processing large codebases efficiently.

    Args:
        path: Path to the file to hash
        chunk_size: Size of chunks to read at once (default: 1MB)

    Returns:
        Hexadecimal SHA1 hash string, or None if file cannot be read

    Example:
        >>> from pathlib import Path
        >>> hash_value = file_sha1(Path("large_file.py"))
        >>> if hash_value:
        ...     print(f"File hash: {hash_value}")

    Note:
        Returns None on any file access error (permission denied, file not found, etc.)
        to allow graceful handling in batch operations.
    """
    try:
        h = hashlib.sha1()
        with path.open("rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def read_text_safely(path: Path, max_bytes: int = 2_000_000) -> str | None:
    """
    Safely read text file with encoding detection and size limits.
    Enhanced to handle more file types and better encoding detection.
    """
    try:
        # Check if it's likely a text file first
        if not is_text_file(path):
            return None

        size = path.stat().st_size
        if size > max_bytes:
            return None

        # Read raw bytes
        with path.open("rb") as f:
            raw = f.read()

        # Try multiple encodings in order of preference
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for enc in encodings:
            try:
                content = raw.decode(enc)
                # Basic sanity check - ensure it's mostly printable text
                if _is_likely_text_content(content):
                    return content
            except UnicodeDecodeError:
                continue

        # Last resort - decode with errors ignored
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


def _is_likely_text_content(content: str) -> bool:
    """Check if content appears to be text (not binary)."""
    if not content:
        return True

    # Check for null bytes (common in binary files)
    if "\x00" in content:
        return False

    # Check ratio of printable characters
    printable_chars = sum(1 for c in content if c.isprintable() or c.isspace())
    ratio = printable_chars / len(content)

    return ratio > 0.7  # At least 70% printable characters


def build_pathspec(
    include: list[str], exclude: list[str]
) -> tuple[pathspec.PathSpec, pathspec.PathSpec]:
    inc = pathspec.PathSpec.from_lines("gitwildmatch", include or ["**/*"])
    exc = pathspec.PathSpec.from_lines("gitwildmatch", exclude or [])
    return inc, exc


def matches_patterns(path: Path, patterns: list[str] | tuple[str, ...]) -> bool:
    """Return True if the given path matches any of the gitwildmatch patterns.

    This helper mirrors the matching logic used elsewhere (absolute, normalized path).
    """
    if not patterns:
        return False
    spec = pathspec.PathSpec.from_lines("gitwildmatch", list(patterns))
    return spec.match_file(str(Path(path).resolve()))


def iter_files(
    roots: Iterable[str],
    include: list[str],
    exclude: list[str],
    follow_symlinks: bool = False,
    *,
    prune_excluded_dirs: bool = True,
    language_filter: set | None = None,
) -> Iterator[Path]:
    """
    遍历文件。支持通过 exclude 规则对目录进行剪枝以减少无谓遍历。
    注意：pathspec 的 gitwildmatch 针对目录匹配时通常以带斜杠的路径进行判断。
    """
    inc, exc = build_pathspec(include, exclude)
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # 预先计算 root 的绝对路径前缀长度，用于构造相对路径字符串以供 pathspec 判断
        root_abs = root_path.resolve()
        root_abs_str = str(root_abs)
        if not root_abs_str.endswith(os.sep):
            root_abs_str = root_abs_str + os.sep

        for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
            # 目录剪枝：原地修改 dirnames，阻止 os.walk 进入被排除的子树
            if prune_excluded_dirs and dirnames:
                pruned: list[str] = []
                for d in list(dirnames):
                    full = Path(dirpath) / d
                    rel_dir = str(full.resolve())
                    # 使用绝对路径字符串进行匹配，简化实现（与文件匹配一致）
                    if exc.match_file(rel_dir):
                        # 从 dirnames 移除该目录，达到剪枝目的
                        dirnames.remove(d)
                    else:
                        pruned.append(d)
                # pruned 列表仅用于可读性，无需返回

            for name in filenames:
                p = Path(dirpath) / name
                rel = str(p.resolve())
                if not inc.match_file(rel):
                    continue
                if exc.match_file(rel):
                    continue

                # Language filtering if specified
                if language_filter is not None:
                    detected_lang = detect_language(p)
                    if detected_lang not in language_filter:
                        continue

                yield p


def file_meta(path: Path) -> FileMeta | None:
    """仅返回最小 stat 信息（不读取全文、不计算 sha1）。"""
    try:
        st = path.stat()
        return FileMeta(path=path, size=st.st_size, mtime=st.st_mtime, sha1=None)
    except Exception:
        return None


def create_file_metadata(path: Path, content: str | None = None) -> FileMetadata | None:
    """Create enhanced file metadata with language detection."""
    try:
        st = path.stat()

        # Detect language
        language = detect_language(path, content)

        # Count lines if content is available
        line_count = None
        if content is not None:
            line_count = content.count("\n") + 1 if content else 0

        return FileMetadata(
            path=path,
            size=st.st_size,
            mtime=st.st_mtime,
            language=language,
            line_count=line_count,
            created_date=st.st_ctime,
            modified_date=st.st_mtime,
        )
    except Exception:
        return None


def extract_context(
    lines: list[str], start: int, end: int, window: int
) -> tuple[int, int, list[str]]:
    """
    lines: full file lines without trailing newlines normalization assumed
    start/end: 1-based line numbers of the primary match segment (inclusive)
    window: context lines to include before and after
    returns: (ctx_start, ctx_end, slice_lines)
    """
    n = len(lines)
    s = max(1, start - window)
    e = min(n, end + window)
    # convert to 0-based slice
    return s, e, lines[s - 1 : e]


def split_lines_keepends(text: str) -> list[str]:
    # 实际不保留换行符，名称容易误导；保留以兼容现有调用。
    # 未来可考虑更名为 split_lines，并保持行为不变。
    return text.splitlines()


def highlight_spans(
    line: str, spans: list[tuple[int, int]], marker_left: str = "[", marker_right: str = "]"
) -> str:
    """Lightweight span highlighting for plain text output when rich is unavailable in some contexts."""
    if not spans:
        return line
    # Ensure non-overlapping and sorted
    spans = sorted(spans, key=lambda x: x[0])
    out: list[str] = []
    last = 0
    for a, b in spans:
        a = max(0, min(len(line), a))
        b = max(0, min(len(line), b))
        if a < last:
            a = last
        if b < a:
            continue
        out.append(line[last:a])
        out.append(marker_left)
        out.append(line[a:b])
        out.append(marker_right)
        last = b
    out.append(line[last:])
    return "".join(out)


def iter_python_ast_nodes(src: str) -> ast.AST | None:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None
