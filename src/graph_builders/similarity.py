"""Similarity graph builders (Phase 2 stubs)."""

from __future__ import annotations

import numpy as np

from src.graph_builders.base import EdgeBudget


class DirectedKNNBuilder:
    """Directed kNN graph builder."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        raise NotImplementedError


class SymmetrizedKNNBuilder:
    """Symmetrized kNN union builder with trimming."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        raise NotImplementedError


class MutualKNNBuilder:
    """Mutual kNN intersection builder with fallback padding."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        raise NotImplementedError


class EpsilonRadiusBuilder:
    """Epsilon-radius/threshold builder with budget trimming."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        raise NotImplementedError


class TopMGlobalSimilarityBuilder:
    """Top-M global similarity edge selector."""

    def build(self, x: np.ndarray, budget: EdgeBudget) -> np.ndarray:
        raise NotImplementedError
