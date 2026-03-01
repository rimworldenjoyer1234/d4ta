"""Similarity graph builders under fixed directed edge budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.graph_builders.base import EdgeBudget, trim_to_budget


def _pairwise_similarity(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    xn = x / norms
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def _all_directed_edges(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = sim.shape[0]
    src, dst = np.where(np.isfinite(sim))
    edges = np.vstack([src, dst]).T.astype(np.int64)
    scores = sim[src, dst].astype(np.float32)
    mask = src != dst
    return edges[mask], scores[mask]


@dataclass
class DirectedKNNBuilder:
    k: int | None = None

    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        n = x.shape[0]
        m = budget.m_for_nodes(n)
        sim = _pairwise_similarity(x)
        k = self.k if self.k is not None else max(1, int(np.ceil(m / max(n, 1))))
        parts = []
        scores = []
        for i in range(n):
            idx = np.argpartition(sim[i], -k)[-k:]
            parts.append(np.column_stack([np.full_like(idx, i), idx]))
            scores.append(sim[i, idx])
        edges = np.vstack(parts) if parts else np.zeros((0, 2), dtype=np.int64)
        vals = np.concatenate(scores) if scores else np.zeros((0,), dtype=np.float32)
        e, s = trim_to_budget(edges, vals, m)
        return e, s, {"padding_frac": float(max(0, m - len(edges)) / max(m, 1))}


class SymmetrizedKNNBuilder:
    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        e, _, _ = DirectedKNNBuilder().build(x, budget)
        rev = e[:, [1, 0]] if len(e) else e
        merged = np.vstack([e, rev]) if len(e) else e
        sim = _pairwise_similarity(x)
        scores = sim[merged[:, 0], merged[:, 1]] if len(merged) else np.zeros((0,), dtype=np.float32)
        out_e, out_s = trim_to_budget(merged, scores, budget.m_for_nodes(x.shape[0]))
        return out_e, out_s, {"padding_frac": 0.0}


class MutualKNNBuilder:
    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        n = x.shape[0]
        m = budget.m_for_nodes(n)
        sim = _pairwise_similarity(x)
        k = max(1, int(np.ceil(m / max(n, 1))))
        neigh = [set(np.argpartition(sim[i], -k)[-k:].tolist()) for i in range(n)]
        edges = []
        scores = []
        for i in range(n):
            for j in neigh[i]:
                if i in neigh[j] and i != j:
                    edges.append((i, j))
                    scores.append(sim[i, j])
        edges_arr = np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)
        scores_arr = np.array(scores, dtype=np.float32) if scores else np.zeros((0,), dtype=np.float32)
        if len(edges_arr) < m:
            all_e, all_s = _all_directed_edges(sim)
            missing = m - len(edges_arr)
            add_e, add_s = trim_to_budget(all_e, all_s, missing)
            edges_arr = np.vstack([edges_arr, add_e]) if len(edges_arr) else add_e
            scores_arr = np.concatenate([scores_arr, add_s]) if len(scores_arr) else add_s
            pad_frac = missing / max(m, 1)
        else:
            pad_frac = 0.0
        out_e, out_s = trim_to_budget(edges_arr, scores_arr, m)
        return out_e, out_s, {"padding_frac": float(pad_frac)}


@dataclass
class EpsilonRadiusBuilder:
    quantile: float = 0.95

    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        n = x.shape[0]
        m = budget.m_for_nodes(n)
        sim = _pairwise_similarity(x)
        valid = sim[np.isfinite(sim)]
        thr = float(np.quantile(valid, self.quantile)) if valid.size else 1.0
        src, dst = np.where(sim >= thr)
        edges = np.vstack([src, dst]).T.astype(np.int64) if src.size else np.zeros((0, 2), dtype=np.int64)
        scores = sim[src, dst].astype(np.float32) if src.size else np.zeros((0,), dtype=np.float32)
        out_e, out_s = trim_to_budget(edges, scores, m)
        return out_e, out_s, {"padding_frac": float(max(0, m - len(edges)) / max(m, 1))}


class TopMGlobalSimilarityBuilder:
    def build(self, x: np.ndarray, budget: EdgeBudget) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        sim = _pairwise_similarity(x)
        edges, scores = _all_directed_edges(sim)
        out_e, out_s = trim_to_budget(edges, scores, budget.m_for_nodes(x.shape[0]))
        return out_e, out_s, {"padding_frac": 0.0}
