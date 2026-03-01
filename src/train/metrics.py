"""Evaluation metrics API."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray, multiclass: bool) -> Dict[str, float]:
    """Compute MCC, AUC-PR, and optional macro-F1."""
    out = {"mcc": float(matthews_corrcoef(y_true, y_pred))}
    if multiclass:
        out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    else:
        out["auc_pr"] = float(average_precision_score(y_true, y_score))
    return out
