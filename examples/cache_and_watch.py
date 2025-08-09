"""
Additional examples demonstrating CacheManager usage and FileWatcher basics.
Run: python examples/cache_and_watch.py
"""
from __future__ import annotations

from time import sleep

from pysearch.api import PySearch
from pysearch.cache_manager import CacheManager
from pysearch.config import SearchConfig


def cache_demo() -> None:
    cfg = SearchConfig(paths=["./src"], include=["**/*.py"], context=0)
    engine = PySearch(cfg)

    cache = CacheManager(backend="memory", default_ttl=5)

    key = "demo:def_search"
    res = engine.search("def", regex=False, context=0)
    cache.set(key, res)
    cached = cache.get(key)
    print("Cache hit?", bool(cached))
    sleep(6)
    expired = cache.get(key)
    print("Expired?", expired is None)


if __name__ == "__main__":
    cache_demo()

