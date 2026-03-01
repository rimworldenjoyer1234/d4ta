"""Micrograph sampling interfaces (Phase 2 stub)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class MicrographSamplingConfig:
    """Configuration for micrograph sampling."""

    n_min: int = 32
    n_max: int = 128
    stratify_label_col: Optional[str] = None
    min_positives: Optional[int] = None


class MicrographSampler:
    """Sampler interface for generating micrographs with N < 200."""

    def __init__(self, config: MicrographSamplingConfig):
        self.config = config

    def sample(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Sample one micrograph node set."""
        raise NotImplementedError("Phase 2 implementation pending")
