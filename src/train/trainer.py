"""Training and evaluation loops for micrograph batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from src.train.metrics import compute_metrics
from src.utils.timing import timer


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16


class MicrographTrainer:
    """Train/eval entrypoints for GNN models on micrograph batches."""

    def __init__(self, config: TrainerConfig, device: str = "cpu"):
        self.config = config
        self.device = torch.device(device)

    def train(self, model: nn.Module, train_graphs: List, val_graphs: List, multiclass: bool = False) -> Dict[str, float]:
        model = model.to(self.device)
        loader = DataLoader([g.data for g in train_graphs], batch_size=self.config.batch_size, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        criterion = nn.CrossEntropyLoss()

        epoch_times: List[float] = []
        for _ in range(self.config.epochs):
            model.train()
            with timer() as t:
                for batch in loader:
                    batch = batch.to(self.device)
                    opt.zero_grad()
                    out = model(batch.x, batch.edge_index)
                    loss = criterion(out, batch.y)
                    loss.backward()
                    opt.step()
            epoch_times.append(t.seconds)

        metrics = self.evaluate(model, val_graphs, multiclass=multiclass)
        metrics["epoch_time_s"] = float(np.mean(epoch_times)) if epoch_times else 0.0
        return metrics

    def evaluate(self, model: nn.Module, graphs: Iterable, multiclass: bool = False) -> Dict[str, float]:
        model.eval()
        ys: List[np.ndarray] = []
        logits: List[np.ndarray] = []
        loader = DataLoader([g.data for g in graphs], batch_size=self.config.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = model(batch.x, batch.edge_index)
                ys.append(batch.y.detach().cpu().numpy())
                logits.append(out.detach().cpu().numpy())

        y_true = np.concatenate(ys) if ys else np.array([])
        y_logit = np.concatenate(logits) if logits else np.zeros((0, 2), dtype=np.float32)
        if y_true.size == 0:
            return {"MCC": float("nan"), "AUCPR": float("nan"), "F1macro": float("nan")}
        return compute_metrics(y_true, y_logit, multiclass=multiclass)
