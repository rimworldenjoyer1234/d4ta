"""Helpers to convert NumPy/Pandas scalars to plain Python types for JSON/YAML."""

from __future__ import annotations

from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    """Recursively convert values to YAML/JSON-safe builtin Python types."""
    if isinstance(value, dict):
        return {str(to_builtin(k)): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
