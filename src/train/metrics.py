"""Evaluation metrics for RQ1."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, matthews_corrcoef


def compute_metrics(y_true: np.ndarray, logits: np.ndarray, multiclass: bool) -> Dict[str, float]:
    """Compute MCC, AUC-PR, and macro-F1."""
    if multiclass:
        y_pred = logits.argmax(axis=1)
        probs = logits
        out = {
            "MCC": float(matthews_corrcoef(y_true, y_pred)),
            "F1macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        try:
            out["AUCPR"] = float(average_precision_score(np.eye(probs.shape[1])[y_true], probs, average="macro"))
        except Exception:
            out["AUCPR"] = float("nan")
        return out

    scores = 1.0 / (1.0 + np.exp(-logits[:, 1])) if logits.shape[1] > 1 else 1.0 / (1.0 + np.exp(-logits[:, 0]))
    y_pred = (scores >= 0.5).astype(int)
    return {
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "AUCPR": float(average_precision_score(y_true, scores)),
        "F1macro": float(f1_score(y_true, y_pred, average="macro")),
    }
