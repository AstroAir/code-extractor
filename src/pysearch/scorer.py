from __future__ import annotations

from typing import List

from .config import SearchConfig
from .types import SearchItem


def score_item(it: SearchItem, cfg: SearchConfig) -> float:
    """
    简单打分模型：
    - 文本命中跨度数量 * text_weight
    - 代码块行数越短得分略高（越聚焦），以对数抑制
    """
    text_hits = len(it.match_spans)
    span_score = text_hits * cfg.text_weight
    length = max(1, it.end_line - it.start_line + 1)
    focus_bonus = 1.0 / (1.0 + (length / 50.0))  # 50 行以上逐渐减弱
    return span_score + focus_bonus


def sort_items(items: List[SearchItem], cfg: SearchConfig) -> List[SearchItem]:
    return sorted(items, key=lambda it: (-score_item(it, cfg), it.file.as_posix(), it.start_line))