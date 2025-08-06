from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .config import SearchConfig
from .utils import FileMeta, file_meta, file_sha1, iter_files


CACHE_FILE = "index.json"


@dataclass(slots=True)
class IndexRecord:
    path: str
    size: int
    mtime: float
    sha1: str | None


class Indexer:
    """
    基于文件元数据(大小/mtime/sha1)的轻量索引器，支持增量更新。
    缓存结构: { "version": 1, "files": { rel_path: IndexRecord } }
    """

    def __init__(self, cfg: SearchConfig) -> None:
        self.cfg = cfg
        self.cache_dir = cfg.resolve_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / CACHE_FILE
        self._index: Dict[str, IndexRecord] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
                files = data.get("files", {})
                self._index = {
                    k: IndexRecord(**v) for k, v in files.items() if self._valid_record(v)
                }
            except Exception:
                self._index = {}
        self._loaded = True

    @staticmethod
    def _valid_record(v: dict) -> bool:
        return all(k in v for k in ("path", "size", "mtime", "sha1"))

    def save(self) -> None:
        out = {
            "version": 1,
            "generated_at": time.time(),
            "files": {k: asdict(r) for k, r in self._index.items()},
        }
        tmp = self.cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.cache_path)

    def _rel(self, p: Path) -> str:
        try:
            return str(p.resolve().relative_to(Path(self.cfg.paths[0]).resolve()))
        except Exception:
            return str(p)

    def scan(self) -> Tuple[List[Path], List[Path], int]:
        """
        返回 (changed_files, removed_files, total_seen)
        """
        self.load()
        seen: set[str] = set()
        changed: List[Path] = []
        total = 0
        for p in iter_files(
            roots=self.cfg.paths,
            include=self.cfg.include,
            exclude=self.cfg.exclude,
            follow_symlinks=self.cfg.follow_symlinks,
            prune_excluded_dirs=self.cfg.dir_prune_exclude,
        ):
            total += 1
            rel = self._rel(p)
            seen.add(rel)

            # 读取最小元数据（stat），必要时才计算 sha1
            meta = file_meta(p)
            if meta is None:
                continue

            rec = self._index.get(rel)
            # 先用 size/mtime 判断变化
            size_changed = (not rec) or (rec.size != meta.size)
            mtime_changed = (not rec) or (rec.mtime != meta.mtime)

            if not self.cfg.strict_hash_check:
                # 非严格模式：仅凭 size/mtime 判断；不计算 sha1
                if size_changed or mtime_changed:
                    self._index[rel] = IndexRecord(
                        path=rel,
                        size=meta.size,
                        mtime=meta.mtime,
                        sha1=rec.sha1 if rec else None,
                    )
                    changed.append(p)
            else:
                # 严格模式：当 size 或 mtime 变化时计算 sha1 并比较
                if size_changed or mtime_changed:
                    current_sha1 = file_sha1(p)
                    if (not rec) or (rec.sha1 != current_sha1):
                        self._index[rel] = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=current_sha1,
                        )
                        changed.append(p)
                    else:
                        # sha1 未变，保持记录但可刷新元数据
                        self._index[rel] = IndexRecord(
                            path=rel,
                            size=meta.size,
                            mtime=meta.mtime,
                            sha1=current_sha1,
                        )
                else:
                    # size/mtime 未变，不更新
                    pass

        # removed
        removed_rels = [rel for rel in self._index.keys() if rel not in seen]
        for rel in removed_rels:
            del self._index[rel]
        removed_paths = [Path(self.cfg.paths[0]) / rel for rel in removed_rels]

        return changed, removed_paths, total

    def iter_all_paths(self) -> Iterator[Path]:
        """
        遍历索引中已知的所有路径（存在性不保证）。
        """
        base = Path(self.cfg.paths[0])
        for rel in self._index.keys():
            yield base / rel

    def count_indexed(self) -> int:
        return len(self._index)