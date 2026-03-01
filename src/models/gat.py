"""GAT model API (Phase 2 stub)."""

from __future__ import annotations

import torch
from torch import nn


class GATClassifier(nn.Module):
    """GAT classifier placeholder using PyTorch Geometric in Phase 2."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.heads = heads

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
