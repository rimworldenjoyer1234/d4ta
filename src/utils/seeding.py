"""Deterministic seeding utilities."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set deterministic seed for Python and NumPy.

    Torch seeding is intentionally omitted here to keep profiling independent from torch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
