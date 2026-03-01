"""Cost measurement API."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostRecord:
    """Runtime and memory cost record."""

    edge_build_seconds: float
    graph_memory_bytes: int
    train_epoch_seconds: float
