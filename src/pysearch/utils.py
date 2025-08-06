from __future__ import annotations

import ast
import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import pathspec


@dataclass(slots=True)
class FileMeta:
    path: Path
    size: int
    mtime: float
    sha1: Optional[str] = None


def sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    """按块计算文件 sha1，避免一次性读取大文件。"""
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


def read_text_safely(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    try:
        size = path.stat().st_size
        if size > max_bytes:
            return None
        # Prefer UTF-8 with fallback
        with path.open("rb") as f:
            raw = f.read()
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return None


def build_pathspec(include: List[str], exclude: List[str]) -> tuple[pathspec.PathSpec, pathspec.PathSpec]:
    inc = pathspec.PathSpec.from_lines("gitwildmatch", include or ["**/*"])
    exc = pathspec.PathSpec.from_lines("gitwildmatch", exclude or [])
    return inc, exc


def iter_files(
    roots: Iterable[str],
    include: List[str],
    exclude: List[str],
    follow_symlinks: bool = False,
    *,
    prune_excluded_dirs: bool = True,
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
                pruned: List[str] = []
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
                yield p


def file_meta(path: Path) -> Optional[FileMeta]:
    """仅返回最小 stat 信息（不读取全文、不计算 sha1）。"""
    try:
        st = path.stat()
        return FileMeta(path=path, size=st.st_size, mtime=st.st_mtime, sha1=None)
    except Exception:
        return None


def extract_context(lines: List[str], start: int, end: int, window: int) -> Tuple[int, int, List[str]]:
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


def split_lines_keepends(text: str) -> List[str]:
    # 实际不保留换行符，名称容易误导；保留以兼容现有调用。
    # 未来可考虑更名为 split_lines，并保持行为不变。
    return text.splitlines()


def highlight_spans(line: str, spans: List[Tuple[int, int]], marker_left: str = "[", marker_right: str = "]") -> str:
    """Lightweight span highlighting for plain text output when rich is unavailable in some contexts."""
    if not spans:
        return line
    # Ensure non-overlapping and sorted
    spans = sorted(spans, key=lambda x: x[0])
    out: List[str] = []
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


def iter_python_ast_nodes(src: str) -> Optional[ast.AST]:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None