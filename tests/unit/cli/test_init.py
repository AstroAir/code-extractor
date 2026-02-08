"""
Tests for pysearch.cli.__init__ module.

Verifies that all public symbols are correctly exported from the CLI package.
"""

from pysearch.cli import (
    bookmarks_cmd,
    cache_cmd,
    cli,
    config_cmd,
    deps_cmd,
    errors_cmd,
    find_cmd,
    history_cmd,
    index_cmd,
    main,
    repo_cmd,
    semantic_cmd,
    watch_cmd,
)


class TestExports:
    """Tests for public API exports from pysearch.cli."""

    def test_cli_exported(self):
        assert callable(cli)

    def test_main_exported(self):
        assert callable(main)

    def test_find_cmd_exported(self):
        assert callable(find_cmd)

    def test_history_cmd_exported(self):
        assert callable(history_cmd)

    def test_bookmarks_cmd_exported(self):
        assert callable(bookmarks_cmd)

    def test_semantic_cmd_exported(self):
        assert callable(semantic_cmd)

    def test_index_cmd_exported(self):
        assert callable(index_cmd)

    def test_deps_cmd_exported(self):
        assert callable(deps_cmd)

    def test_watch_cmd_exported(self):
        assert callable(watch_cmd)

    def test_cache_cmd_exported(self):
        assert callable(cache_cmd)

    def test_config_cmd_exported(self):
        assert callable(config_cmd)

    def test_repo_cmd_exported(self):
        assert callable(repo_cmd)

    def test_errors_cmd_exported(self):
        assert callable(errors_cmd)

    def test_all_list(self):
        import pysearch.cli as cli_module

        expected = {
            "main", "cli", "find_cmd", "history_cmd", "bookmarks_cmd",
            "semantic_cmd", "index_cmd", "deps_cmd", "watch_cmd",
            "cache_cmd", "config_cmd", "repo_cmd", "errors_cmd",
        }
        assert set(cli_module.__all__) == expected
