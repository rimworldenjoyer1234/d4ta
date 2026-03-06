#!/usr/bin/env python3
"""Run RQ1 edge-budget experiments on reusable micrographs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph_builders.relational import ShareEntityConfig, ShareEntityGraphBuilder
from src.graph_builders.similarity import (
    DirectedKNNBuilder,
    EpsilonRadiusBuilder,
    MutualKNNBuilder,
    SymmetrizedKNNBuilder,
    TopMGlobalSimilarityBuilder,
)
from src.models.gat import GATClassifier
from src.models.gcn import GCNClassifier
from src.preprocess.feature_builder import fit_kdd_preprocessor, fit_netflow_preprocessor
from src.sampling.micrograph_specs import MicrographSpecConfig, generate_specs, read_specs_jsonl, write_specs_jsonl
from src.train.data import build_graph_batch
from src.train.trainer import MicrographTrainer, TrainerConfig
from src.utils.logging_utils import setup_logging
from src.utils.seeding import set_global_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    p.add_argument("--datasets", nargs="+", default=["unsw_nb15_netflow", "ton_iot_netflow", "nsl_kdd"])
    p.add_argument("--dbars", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--builders", nargs="+", default=["directed_knn", "sym_knn", "mutual_knn", "epsilon_radius", "topM_global", "share_entity"])
    p.add_argument("--models", nargs="+", default=["gcn", "gat"])
    p.add_argument("--seeds", nargs="+", type=int, default=[123])
    p.add_argument("--n_graphs", type=int, default=100)
    p.add_argument("--n_min", type=int, default=64)
    p.add_argument("--n_max", type=int, default=180)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def _load_pool(base: Path, dataset: str, split: str) -> pd.DataFrame:
    return pd.read_parquet(base / "pools" / dataset / f"{split}.parquet")


def _builder(name: str, dataset: str):
    if name == "directed_knn":
        return DirectedKNNBuilder()
    if name == "sym_knn":
        return SymmetrizedKNNBuilder()
    if name == "mutual_knn":
        return MutualKNNBuilder()
    if name == "epsilon_radius":
        return EpsilonRadiusBuilder()
    if name == "topM_global":
        return TopMGlobalSimilarityBuilder()
    if name == "share_entity":
        keys = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"] if dataset != "nsl_kdd" else ["service", "protocol_type", "flag"]
        return ShareEntityGraphBuilder(ShareEntityConfig(keys=keys))
    raise ValueError(name)


def _model(name: str, in_dim: int, out_dim: int):
    if name == "gcn":
        return GCNClassifier(in_dim=in_dim, hidden_dim=64, out_dim=out_dim)
    if name == "gat":
        return GATClassifier(in_dim=in_dim, hidden_dim=32, out_dim=out_dim)
    raise ValueError(name)


def _fit_preprocessor(out_dir: Path, dataset: str, train_df: pd.DataFrame) -> Path:
    prep_dir = out_dir / "preprocess" / dataset
    if dataset == "nsl_kdd":
        cat = ["protocol_type", "service", "flag"]
        exclude = {"binary_label", "multiclass_label", "row_id", *cat}
        num = [c for c in train_df.columns if c not in exclude]
        artifact = fit_kdd_preprocessor(train_df, categorical_cols=cat, numeric_cols=num, out_dir=prep_dir)
    else:
        exclude = {"binary_label", "multiclass_label", "row_id", "Label", "Attack", "IPV4_SRC_ADDR", "IPV4_DST_ADDR"}
        feats = [c for c in train_df.columns if c not in exclude]
        log_cols = [c for c in feats if "BYTES" in c or "PKTS" in c]
        artifact = fit_netflow_preprocessor(train_df, feature_cols=feats, log1p_cols=log_cols, out_dir=prep_dir)
    return artifact.model_path


def main() -> None:
    args = parse_args()
    setup_logging()

    rows = []
    out_results = args.out_dir / "results"
    out_results.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        train_pool = _load_pool(args.out_dir, dataset, "train")
        val_pool = _load_pool(args.out_dir, dataset, "val")
        test_pool = _load_pool(args.out_dir, dataset, "test")

        preprocessor_path = _fit_preprocessor(args.out_dir, dataset, train_pool)

        for seed in args.seeds:
            set_global_seed(seed)
            spec_cfg = MicrographSpecConfig(n_graphs=args.n_graphs, n_min=args.n_min, n_max=args.n_max)
            specs_dir = args.out_dir / "micrograph_specs" / dataset
            specs_dir.mkdir(parents=True, exist_ok=True)
            split_specs = {}
            for split, pool, balanced in (("train", train_pool, True), ("val", val_pool, False), ("test", test_pool, False)):
                p = specs_dir / f"{split}_seed{seed}.jsonl"
                if p.exists():
                    specs = read_specs_jsonl(p)
                else:
                    specs = generate_specs(pool, split=split, seed=seed, config=spec_cfg, enforce_balance=balanced)
                    write_specs_jsonl(p, specs)
                split_specs[split] = specs

            for bname in args.builders:
                builder = _builder(bname, dataset)
                for dbar in args.dbars:
                    train_graphs = build_graph_batch(train_pool, split_specs["train"], preprocessor_path, builder, dbar)
                    val_graphs = build_graph_batch(val_pool, split_specs["val"], preprocessor_path, builder, dbar)
                    test_graphs = build_graph_batch(test_pool, split_specs["test"], preprocessor_path, builder, dbar)

                    avg_build = float(np.mean([g.cost["build_time_s"] for g in test_graphs]))
                    avg_bytes = float(np.mean([g.cost["graph_bytes"] for g in test_graphs]))
                    avg_regime = {
                        k: float(np.mean([g.cost.get(k, 0.0) for g in test_graphs]))
                        for k in ["reciprocity", "isolates", "n_components", "giant_comp_frac", "padding_frac"]
                    }

                    for mname in args.models:
                        model = _model(mname, in_dim=train_graphs[0].data.x.shape[1], out_dim=2)
                        trainer = MicrographTrainer(TrainerConfig(epochs=args.epochs), device=args.device)
                        metrics = trainer.train(model, train_graphs=train_graphs, val_graphs=val_graphs, multiclass=False)
                        test_metrics = trainer.evaluate(model, test_graphs, multiclass=False)

                        for gspec in split_specs["test"]:
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "split": "test",
                                    "seed": seed,
                                    "graph_id": gspec.graph_id,
                                    "N": gspec.N,
                                    "dbar": dbar,
                                    "builder": bname,
                                    "model": mname,
                                    "MCC": test_metrics["MCC"],
                                    "AUCPR": test_metrics["AUCPR"],
                                    "F1macro": test_metrics["F1macro"],
                                    "build_time_s": avg_build,
                                    "graph_bytes": avg_bytes,
                                    "epoch_time_s": metrics["epoch_time_s"],
                                    **avg_regime,
                                }
                            )

    out = pd.DataFrame(rows)
    out_path = out_results / "rq1_results.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
