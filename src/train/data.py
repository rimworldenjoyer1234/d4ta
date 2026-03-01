"""PyG data construction for micrograph specs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.graph_builders.base import EdgeBudget, graph_regime_metrics
from src.preprocess.feature_builder import transform_features
from src.sampling.micrograph_specs import MicrographSpec
from src.utils.timing import timer


@dataclass
class BuiltGraph:
    data: Data
    cost: Dict[str, float]


def build_graph_from_spec(
    pool_df: pd.DataFrame,
    spec: MicrographSpec,
    preprocessor_path: Path,
    builder: Any,
    dbar: int,
) -> BuiltGraph:
    """Build one PyG graph from a micrograph spec and graph builder."""
    node_df = pool_df[pool_df["row_id"].isin(spec.node_row_ids)].copy().reset_index(drop=True)
    feature_cols = [c for c in node_df.columns if c not in {"binary_label", "multiclass_label", "row_id", "Label", "Attack", "label", "difficulty"}]
    x_np = transform_features(node_df[feature_cols], preprocessor_path)

    with timer() as t:
        if builder.__class__.__name__ == "ShareEntityGraphBuilder":
            edge_index_np, _, extra = builder.build(node_df, EdgeBudget(dbar=dbar))
        else:
            edge_index_np, _, extra = builder.build(x_np, EdgeBudget(dbar=dbar))

    y_np = node_df["binary_label"].astype(int).to_numpy()
    edge_index = torch.tensor(edge_index_np.T, dtype=torch.long) if edge_index_np.size else torch.empty((2, 0), dtype=torch.long)
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    graph_bytes = float(x.element_size() * x.nelement() + edge_index.element_size() * edge_index.nelement() + y.element_size() * y.nelement())
    regime = graph_regime_metrics(edge_index_np, n_nodes=len(node_df))
    cost = {"build_time_s": t.seconds, "graph_bytes": graph_bytes, **regime, **extra}
    return BuiltGraph(data=data, cost=cost)


def build_graph_batch(
    pool_df: pd.DataFrame,
    specs: Iterable[MicrographSpec],
    preprocessor_path: Path,
    builder: Any,
    dbar: int,
) -> List[BuiltGraph]:
    return [build_graph_from_spec(pool_df, s, preprocessor_path, builder, dbar) for s in specs]
