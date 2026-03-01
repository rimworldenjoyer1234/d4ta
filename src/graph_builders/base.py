"""Graph builder interfaces and edge-budget utility."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class EdgeBudget:
    """Directed edge budget M = N * d_bar."""

    d_bar: int

    def m_for_nodes(self, n_nodes: int) -> int:
        return int(n_nodes * self.d_bar)


class GraphBuilder(Protocol):
    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        ...


def trim_to_budget(edges: np.ndarray, scores: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Trim/select edges to exact budget size by descending scores."""
    if m <= 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    if len(edges) == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    order = np.argsort(scores)[::-1]
    take = min(m, len(order))
    out_edges = edges[order[:take]].astype(np.int64)
    out_scores = scores[order[:take]].astype(np.float32)

    if take < m:
        n = int(edges[:, :2].max()) + 1 if edges.size else 1
        rng = np.random.default_rng(0)
        pad_src = rng.integers(0, n, size=(m - take, 1), endpoint=False)
        pad_dst = rng.integers(0, n, size=(m - take, 1), endpoint=False)
        pad = np.hstack([pad_src, pad_dst]).astype(np.int64)
        out_edges = np.vstack([out_edges, pad])
        out_scores = np.concatenate([out_scores, np.zeros(m - take, dtype=np.float32)])
    return out_edges, out_scores


def graph_regime_metrics(edge_index: np.ndarray, n_nodes: int) -> Dict[str, float]:
    """Compute reciprocity, isolates, weak components, giant component fraction."""
    if n_nodes == 0:
        return {"reciprocity": 0.0, "isolates": 0.0, "n_components": 0.0, "giant_comp_frac": 0.0}
    if edge_index.size == 0:
        return {"reciprocity": 0.0, "isolates": 1.0, "n_components": float(n_nodes), "giant_comp_frac": 1.0 / n_nodes}

    edges = {(int(u), int(v)) for u, v in edge_index.tolist()}
    reciprocal = sum((v, u) in edges for u, v in edges)
    reciprocity = reciprocal / max(len(edges), 1)

    deg = np.zeros(n_nodes, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    isolates = float((deg == 0).sum() / n_nodes)

    adj = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    seen = np.zeros(n_nodes, dtype=bool)
    comps = []
    for i in range(n_nodes):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        sz = 0
        while stack:
            cur = stack.pop()
            sz += 1
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        comps.append(sz)
    giant = max(comps) if comps else 0
    return {
        "reciprocity": float(reciprocity),
        "isolates": float(isolates),
        "n_components": float(len(comps)),
        "giant_comp_frac": float(giant / n_nodes),
    }
