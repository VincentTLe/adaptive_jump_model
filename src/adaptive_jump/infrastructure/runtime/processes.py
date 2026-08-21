"""Cross-platform process-pool configuration."""

from __future__ import annotations

import multiprocessing
from multiprocessing.context import BaseContext


def process_pool_context() -> BaseContext:
    """Prefer forkserver where available and otherwise use spawn.

    ``forkserver`` keeps the Linux research runs isolated from the parent
    process, but it is not implemented on Windows. ``spawn`` has the same
    isolation property and is available on every supported Python platform.
    """
    methods = set(multiprocessing.get_all_start_methods())
    if "forkserver" in methods:
        return multiprocessing.get_context("forkserver")
    if "spawn" in methods:
        return multiprocessing.get_context("spawn")
    return multiprocessing.get_context()
