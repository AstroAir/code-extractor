#!/usr/bin/env python3
"""
Example 03: Fuzzy Search — 模糊搜索示例

Demonstrates:
- Single-algorithm fuzzy search / 单算法模糊搜索
- Multi-algorithm fuzzy search / 多算法组合模糊搜索
- Word-level fuzzy search / 词级模糊搜索
- Phonetic search / 语音搜索
- Spelling correction suggestions / 拼写纠正建议
"""

from __future__ import annotations

import sys
from pathlib import Path

from pysearch import PySearch, SearchConfig

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def print_items(items: list, max_items: int = 5) -> None:
    for item in items[:max_items]:
        print(f"  {item.file}:{item.start_line}-{item.end_line}")
        for line in item.lines[:3]:
            print(f"    {line.rstrip()}")
        print()


# ---------------------------------------------------------------------------
# 1. Single-algorithm fuzzy search / 单算法模糊搜索
# ---------------------------------------------------------------------------


def demo_single_fuzzy(engine: PySearch) -> None:
    section("1. Single-Algorithm Fuzzy Search / 单算法模糊搜索")

    # Search with Levenshtein distance — tolerates typos
    # 使用 Levenshtein 距离搜索 — 容忍拼写错误
    results = engine.fuzzy_search(
        pattern="serach",  # intentional typo for "search"
        max_distance=2,
        algorithm="levenshtein",
    )
    print("Fuzzy search for 'serach' (Levenshtein, dist=2):")
    print(f"  Found {len(results.items)} matches\n")
    print_items(results.items, max_items=3)

    # Jaro-Winkler algorithm — good for prefix similarity
    # Jaro-Winkler 算法 — 适合前缀相似
    results2 = engine.fuzzy_search(
        pattern="config",
        max_distance=2,
        algorithm="jaro_winkler",
    )
    print("Fuzzy search for 'config' (Jaro-Winkler):")
    print(f"  Found {len(results2.items)} matches\n")
    print_items(results2.items, max_items=3)


# ---------------------------------------------------------------------------
# 2. Multi-algorithm fuzzy search / 多算法组合模糊搜索
# ---------------------------------------------------------------------------


def demo_multi_algorithm(engine: PySearch) -> None:
    section("2. Multi-Algorithm Fuzzy Search / 多算法组合")

    # Combine multiple algorithms for better recall
    # 组合多种算法以提高召回率
    results = engine.multi_algorithm_fuzzy_search(
        pattern="indxer",  # intentional typo for "indexer"
        algorithms=["levenshtein", "damerau_levenshtein", "jaro_winkler"],
        max_distance=2,
        min_similarity=0.6,
    )
    print("Multi-algorithm fuzzy for 'indxer':")
    print(f"  Found {len(results.items)} matches\n")
    print_items(results.items, max_items=3)


# ---------------------------------------------------------------------------
# 3. Word-level fuzzy search / 词级模糊搜索
# ---------------------------------------------------------------------------


def demo_word_level_fuzzy(engine: PySearch) -> None:
    section("3. Word-Level Fuzzy Search / 词级模糊搜索")

    # Real similarity scoring on individual words
    # 对单个词进行真实相似度打分
    results = engine.word_level_fuzzy_search(
        pattern="macher",  # intentional typo for "matcher"
        max_distance=2,
        min_similarity=0.7,
        max_results=20,
    )
    print("Word-level fuzzy for 'macher' (min_similarity=0.7):")
    print(f"  Found {len(results.items)} matches in {results.stats.files_matched} files")
    print(f"  Elapsed: {results.stats.elapsed_ms:.1f}ms\n")
    print_items(results.items, max_items=3)


# ---------------------------------------------------------------------------
# 4. Phonetic search / 语音搜索
# ---------------------------------------------------------------------------


def demo_phonetic_search(engine: PySearch) -> None:
    section("4. Phonetic Search / 语音搜索")

    # Search for words that sound similar using Soundex
    # 使用 Soundex 搜索发音相似的词
    results = engine.phonetic_search(
        pattern="kache",  # sounds like "cache"
        algorithm="soundex",
    )
    print("Phonetic search for 'kache' (Soundex):")
    print(f"  Found {len(results.items)} matches\n")
    print_items(results.items, max_items=3)

    # Metaphone algorithm
    results2 = engine.phonetic_search(
        pattern="formater",  # sounds like "formatter"
        algorithm="metaphone",
    )
    print("Phonetic search for 'formater' (Metaphone):")
    print(f"  Found {len(results2.items)} matches\n")
    print_items(results2.items, max_items=3)


# ---------------------------------------------------------------------------
# 5. Spelling correction suggestions / 拼写纠正建议
# ---------------------------------------------------------------------------


def demo_suggest_corrections(engine: PySearch) -> None:
    section("5. Spelling Correction / 拼写纠正")

    # Get spelling suggestions based on codebase identifiers
    # 基于代码库标识符获取拼写建议
    test_words = ["conection", "configration", "serach"]

    for word in test_words:
        suggestions = engine.suggest_corrections(
            word=word,
            max_suggestions=5,
            algorithm="damerau_levenshtein",
        )
        print(f"  '{word}' → suggestions:")
        if suggestions:
            for suggestion, score in suggestions:
                print(f"    {suggestion} (similarity: {score:.3f})")
        else:
            print("    (no suggestions found)")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    search_path = sys.argv[1] if len(sys.argv) > 1 else "./src"

    if not Path(search_path).exists():
        print(f"Error: search path '{search_path}' does not exist.")
        sys.exit(1)

    config = SearchConfig(paths=[search_path], include=["**/*.py"], context=2)
    engine = PySearch(config)

    demo_single_fuzzy(engine)
    demo_multi_algorithm(engine)
    demo_word_level_fuzzy(engine)
    demo_phonetic_search(engine)
    demo_suggest_corrections(engine)

    print("\n✓ All fuzzy search examples completed.")


if __name__ == "__main__":
    main()
