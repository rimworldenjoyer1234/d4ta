"""GCN model API (Phase 2 stub)."""

from __future__ import annotations

import torch
from torch import nn


class GCNClassifier(nn.Module):
    """GCN classifier placeholder using PyTorch Geometric in Phase 2."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError
