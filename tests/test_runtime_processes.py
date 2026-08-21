"""Cross-platform process-pool context checks."""

from __future__ import annotations

import multiprocessing

from adaptive_jump.infrastructure.runtime.processes import process_pool_context


def test_process_pool_context_uses_a_supported_start_method() -> None:
    context = process_pool_context()

    assert context.get_start_method() in multiprocessing.get_all_start_methods()
