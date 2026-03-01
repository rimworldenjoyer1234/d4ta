"""GAT model for node classification."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATConv


class GATClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index)
