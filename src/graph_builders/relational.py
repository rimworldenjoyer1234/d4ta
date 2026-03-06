"""Relational graph builders with budget trimming."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.graph_builders.base import EdgeBudget, trim_to_budget


@dataclass(frozen=True)
class ShareEntityConfig:
    keys: Sequence[str]


class ShareEntityGraphBuilder:
    """Build directed edges among nodes sharing entity values."""

    def __init__(self, config: ShareEntityConfig):
        self.config = config

    def build(self, frame: pd.DataFrame, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        n = len(frame)
        m = budget.m_for_nodes(n)
        candidates: set[tuple[int, int]] = set()

        for key in self.config.keys:
            if key not in frame.columns:
                continue
            groups = frame.groupby(key).indices
            for _, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                for i, j in combinations(idxs, 2):
                    candidates.add((int(i), int(j)))
                    candidates.add((int(j), int(i)))

        if not candidates:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), {"padding_frac": 1.0}

        edges = np.array(list(candidates), dtype=np.int64)
        scores = np.ones(len(edges), dtype=np.float32)
        out_e, out_s = trim_to_budget(edges, scores, m)
        return out_e, out_s, {"padding_frac": float(max(0, m - len(edges)) / max(m, 1))}
