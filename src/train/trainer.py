"""Training loop API stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainerConfig:
    """Trainer config for micrograph batches."""

    epochs: int = 10
    lr: float = 1e-3


class MicrographTrainer:
    """Train/eval entrypoints for GNN models on micrograph batches."""

    def __init__(self, config: TrainerConfig):
        self.config = config

    def train_epoch(self, model: Any, batch_iter: Any) -> dict:
        raise NotImplementedError

    def evaluate(self, model: Any, batch_iter: Any) -> dict:
        raise NotImplementedError
