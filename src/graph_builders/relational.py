"""Relational graph builder stub."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class ShareEntityConfig:
    """Relational builder configuration."""

    keys: Sequence[str]


class ShareEntityGraphBuilder:
    """Connect rows sharing one or more entity keys, then trim to budget."""

    def __init__(self, config: ShareEntityConfig):
        self.config = config

    def build(self, frame: pd.DataFrame, m: int):
        raise NotImplementedError
