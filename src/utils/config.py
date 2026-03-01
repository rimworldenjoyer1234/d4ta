"""Configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProfilingConfig:
    """Runtime options for profiling jobs."""

    chunk_size: int = 50_000
    top_k_categories: int = 20
    sample_values_per_col: int = 5
    out_dir: Path = Path("artifacts")
