"""Micrograph specification generation and persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MicrographSpec:
    """A reproducible micrograph sampling spec."""

    graph_id: str
    split: str
    N: int
    node_row_ids: List[int]
    seed: int


@dataclass(frozen=True)
class MicrographSpecConfig:
    n_graphs: int = 200
    n_min: int = 64
    n_max: int = 180
    min_pos: int = 2
    min_neg: int = 2


def _sample_row_ids(
    frame: pd.DataFrame,
    n: int,
    rng: np.random.Generator,
    enforce_balance: bool,
    min_pos: int,
    min_neg: int,
) -> List[int]:
    if not enforce_balance:
        return frame.sample(n=n, random_state=int(rng.integers(1, 2**31 - 1)))["row_id"].astype(int).tolist()

    pos = frame[frame["binary_label"] == 1]
    neg = frame[frame["binary_label"] == 0]
    p_take = min(min_pos, len(pos), n)
    n_take = min(min_neg, len(neg), max(0, n - p_take))

    picked = []
    if p_take > 0:
        picked.append(pos.sample(n=p_take, random_state=int(rng.integers(1, 2**31 - 1))))
    if n_take > 0:
        picked.append(neg.sample(n=n_take, random_state=int(rng.integers(1, 2**31 - 1))))
    picked_df = pd.concat(picked, ignore_index=True) if picked else frame.iloc[:0]

    remain = n - len(picked_df)
    if remain > 0:
        rest = frame[~frame["row_id"].isin(picked_df["row_id"])]
        if len(rest) >= remain:
            picked_df = pd.concat(
                [picked_df, rest.sample(n=remain, random_state=int(rng.integers(1, 2**31 - 1)))],
                ignore_index=True,
            )
    return picked_df["row_id"].astype(int).tolist()


def generate_specs(
    pool_df: pd.DataFrame,
    split: str,
    seed: int,
    config: MicrographSpecConfig,
    enforce_balance: bool,
) -> List[MicrographSpec]:
    """Generate deterministic micrograph specs for a split."""
    assert config.n_max <= 200, "N_max must be <= 200"
    rng = np.random.default_rng(seed)
    specs: List[MicrographSpec] = []
    for i in range(config.n_graphs):
        n = int(rng.integers(config.n_min, config.n_max + 1))
        row_ids = _sample_row_ids(pool_df, n, rng, enforce_balance, config.min_pos, config.min_neg)
        specs.append(MicrographSpec(graph_id=f"{split}_{seed}_{i:05d}", split=split, N=n, node_row_ids=row_ids, seed=seed))
    return specs


def write_specs_jsonl(path: Path, specs: Iterable[MicrographSpec]) -> None:
    """Persist specs as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for spec in specs:
            f.write(json.dumps(asdict(spec)) + "\n")


def read_specs_jsonl(path: Path) -> List[MicrographSpec]:
    """Load specs from JSONL."""
    out: List[MicrographSpec] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out.append(MicrographSpec(**row))
    return out
