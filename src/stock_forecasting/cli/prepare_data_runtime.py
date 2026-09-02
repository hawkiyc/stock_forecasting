"""Run Stage 1 preparation with scope-aware runtime resource detection."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager

from stock_forecasting.cli import prepare_data
from stock_forecasting.data import bar_store
from stock_forecasting.runtime_resources import (
    AvailableMemoryEstimate,
    detect_available_memory,
)


@contextmanager
def _runtime_memory_detection(estimate: AvailableMemoryEstimate) -> Iterator[None]:
    """Override the legacy probe only for this dedicated CLI process."""

    original = bar_store._detect_available_memory_bytes

    def available_memory_bytes() -> int:
        return estimate.available_bytes

    bar_store._detect_available_memory_bytes = available_memory_bytes
    try:
        yield
    finally:
        bar_store._detect_available_memory_bytes = original


def main() -> int:
    """Execute preparation with the shared cgroup-aware memory estimate."""

    estimate = detect_available_memory()
    print(
        json.dumps(
            {"bar_store_memory_detection": estimate.as_dict()},
            sort_keys=True,
        ),
        flush=True,
    )
    with _runtime_memory_detection(estimate):
        return prepare_data.main()


if __name__ == "__main__":
    raise SystemExit(main())
