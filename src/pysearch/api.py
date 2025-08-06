from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

from .config import SearchConfig
from .indexer import Indexer
from .matchers import search_in_file
from .types import OutputFormat, Query, SearchItem, SearchResult, SearchStats
from .utils import read_text_safely


class PySearch:
    def __init__(self, config: Optional[SearchConfig] = None) -> None:
        self.cfg = config or SearchConfig()
        self.indexer = Indexer(self.cfg)

    def _search_file(self, path: Path, query: Query) -> List[SearchItem]:
        text = read_text_safely(path, max_bytes=self.cfg.max_file_bytes)
        if text is None:
            return []
        return search_in_file(path, text, query)

    def run(self, query: Query) -> SearchResult:
        t0 = time.perf_counter()
        changed, removed, total_seen = self.indexer.scan()
        self.indexer.save()

        paths = changed or list(self.indexer.iter_all_paths())

        items: List[SearchItem] = []

        if self.cfg.parallel:
            workers = self.cfg.workers or min(32, (os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(self._search_file, p, query): p for p in paths}
                for fut in as_completed(futs):
                    res = fut.result()
                    if res:
                        items.extend(res)
        else:
            for p in paths:
                res = self._search_file(p, query)
                if res:
                    items.extend(res)

        stats = SearchStats(
            files_scanned=total_seen,
            files_matched=len({it.file for it in items}),
            items=len(items),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            indexed_files=self.indexer.count_indexed(),
        )
        return SearchResult(items=items, stats=stats)

    # Convenience text/regex api
    def search(
        self,
        pattern: str,
        regex: bool = False,
        context: Optional[int] = None,
        output: OutputFormat = OutputFormat.TEXT,
        **kwargs,
    ) -> SearchResult:
        q = Query(
            pattern=pattern,
            use_regex=regex,
            use_ast=kwargs.get("use_ast", False),
            context=context if context is not None else self.cfg.context,
            output=output,
            filters=kwargs.get("filters"),
            search_docstrings=self.cfg.enable_docstrings,
            search_comments=self.cfg.enable_comments,
            search_strings=self.cfg.enable_strings,
        )
        return self.run(q)