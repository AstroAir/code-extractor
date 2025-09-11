"""
Locking mechanisms for the enhanced indexing system.

This module provides file-based locking to prevent concurrent indexing
operations across multiple processes and avoid SQLite concurrent write errors.

Classes:
    IndexLock: File-based locking for indexing operations

Features:
    - Cross-process locking using file system
    - Stale lock detection and cleanup
    - Timeout support for lock acquisition
    - Automatic lock timestamp updates
    - Robust error handling
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, List

from ...utils.logging_config import get_logger

logger = get_logger()


class IndexLock:
    """
    Prevents concurrent indexing operations across multiple processes.

    Uses file-based locking to coordinate indexing operations and prevent
    SQLite concurrent write errors.
    """

    def __init__(self, cache_dir: Path):
        self.lock_file = cache_dir / "indexing.lock"
        self.cache_dir = cache_dir

    async def acquire(self, directories: List[str], timeout: float = 300.0) -> bool:
        """
        Acquire indexing lock.

        Args:
            directories: List of directories being indexed
            timeout: Maximum time to wait for lock

        Returns:
            True if lock acquired, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if not self.lock_file.exists():
                    # Create lock file
                    lock_data = {
                        "directories": directories,
                        "timestamp": time.time(),
                        "pid": os.getpid(),
                    }

                    # Atomic write
                    temp_file = self.lock_file.with_suffix(".tmp")
                    temp_file.write_text(str(lock_data))
                    temp_file.rename(self.lock_file)

                    return True
                else:
                    # Check if existing lock is stale
                    try:
                        existing_lock_data: dict[str, Any] = eval(
                            self.lock_file.read_text())
                        timestamp = float(existing_lock_data["timestamp"])
                        if time.time() - timestamp > 600:  # 10 minutes
                            logger.warning("Removing stale indexing lock")
                            self.lock_file.unlink()
                            continue
                    except Exception:
                        # Corrupted lock file, remove it
                        self.lock_file.unlink()
                        continue

                # Wait before retrying
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error acquiring index lock: {e}")
                await asyncio.sleep(1.0)

        return False

    async def release(self) -> None:
        """Release the indexing lock."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error(f"Error releasing index lock: {e}")

    async def update_timestamp(self) -> None:
        """Update lock timestamp to prevent stale lock detection."""
        try:
            if self.lock_file.exists():
                lock_data = eval(self.lock_file.read_text())
                lock_data["timestamp"] = time.time()
                self.lock_file.write_text(str(lock_data))
        except Exception as e:
            logger.error(f"Error updating lock timestamp: {e}")
