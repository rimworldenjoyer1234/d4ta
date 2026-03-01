"""Graph builder interfaces and edge-budget utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EdgeBudget:
    """Directed edge budget M = N * d_bar."""

    d_bar: int

    def m_for_nodes(self, n_nodes: int) -> int:
        """Compute directed edge budget."""
        return int(n_nodes * self.d_bar)


class GraphBuilder(Protocol):
    """Protocol for graph construction under a fixed budget."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        """Return directed edges as shape [E, 2] int array."""
        ...


def trim_to_budget(edges: np.ndarray, scores: np.ndarray, m: int) -> np.ndarray:
    """Trim or select edges to exact budget size using descending scores."""
    if len(edges) == 0 or m <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(scores)[::-1]
    return edges[order[:m]].astype(np.int64)
