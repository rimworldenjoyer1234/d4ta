"""Timing helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator


@dataclass
class TimerResult:
    """Simple timer result container."""

    seconds: float = 0.0


@contextmanager
def timer() -> Iterator[TimerResult]:
    """Context manager for elapsed time measurements."""
    start = time.perf_counter()
    result = TimerResult()
    try:
        yield result
    finally:
        result.seconds = time.perf_counter() - start
