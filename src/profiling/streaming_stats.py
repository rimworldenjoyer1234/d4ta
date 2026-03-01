"""Streaming statistics for large tabular datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class NumericTracker:
    """Welford tracker for numeric statistics."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    def update(self, values: pd.Series) -> None:
        """Update tracker from a numeric series."""
        arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
        for x in arr:
            self.count += 1
            delta = x - self.mean
            self.mean += delta / self.count
            delta2 = x - self.mean
            self.m2 += delta * delta2
            self.min_value = x if self.min_value is None else min(self.min_value, x)
            self.max_value = x if self.max_value is None else max(self.max_value, x)

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Serialize stats."""
        variance = (self.m2 / (self.count - 1)) if self.count > 1 else 0.0
        return {
            "min": float(self.min_value) if self.min_value is not None else None,
            "max": float(self.max_value) if self.max_value is not None else None,
            "mean": float(self.mean) if self.count > 0 else None,
            "std": float(np.sqrt(variance)) if self.count > 0 else None,
            "count": int(self.count),
        }


@dataclass
class ColumnTracker:
    """Aggregate profile information for a column."""

    name: str
    seen: int = 0
    missing: int = 0
    numeric: NumericTracker = field(default_factory=NumericTracker)
    value_counts: Counter[str] = field(default_factory=Counter)
    unique_values: set[str] = field(default_factory=set)
    sample_values: List[str] = field(default_factory=list)

    def update(self, values: pd.Series, sample_limit: int) -> None:
        """Update with a chunk column."""
        self.seen += len(values)
        self.missing += int(values.isna().sum())
        non_null = values.dropna()

        self.numeric.update(non_null)

        as_str = non_null.astype(str)
        self.value_counts.update(as_str.tolist())
        self.unique_values.update(as_str.tolist())

        for value in as_str.tolist():
            if len(self.sample_values) >= sample_limit:
                break
            if value not in self.sample_values:
                self.sample_values.append(value)

    def missing_rate(self) -> float:
        """Return missing rate in [0, 1]."""
        return (self.missing / self.seen) if self.seen else 0.0

    def cardinality(self) -> int:
        """Return cardinality estimate (exact for current implementation)."""
        return len(self.unique_values)
