from __future__ import annotations

"""
IDE 集成占位：为 IDE/编辑器提供简易接口，后续可扩展为 LSP/自定义协议。
"""

from dataclasses import asdict
from typing import Any

from ..core.api import PySearch
from .types import Query


def ide_query(engine: PySearch, query: Query) -> dict[str, Any]:
    """
    供 IDE 调用的查询接口，返回结构化 JSON（dict）。
    """
    res = engine.run(query)
    return {
        "items": [
            {
                "file": str(it.file),
                "start_line": it.start_line,
                "end_line": it.end_line,
                "lines": it.lines,
                "spans": [(li, (a, b)) for li, (a, b) in it.match_spans],
            }
            for it in res.items
        ],
        "stats": asdict(res.stats),
    }
