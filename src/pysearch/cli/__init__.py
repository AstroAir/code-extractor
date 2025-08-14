"""
Command-line interface implementation.

This module provides the command-line interface for pysearch:
- CLI command definitions and parsing
- Interactive command-line features
- Terminal output formatting
- Command-line argument validation

The CLI module makes pysearch accessible from the command line
with a rich set of options and user-friendly interface.
"""

from .main import main

__all__ = [
    "main",
]
