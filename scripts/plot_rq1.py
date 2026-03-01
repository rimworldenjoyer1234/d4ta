#!/usr/bin/env python3
"""Generate RQ1 plots/tables from result CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("artifacts/results/rq1_results.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("artifacts/plots"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    df["total_cost"] = df["build_time_s"] + df["epoch_time_s"]

    # Fig1 MCC vs dbar
    for dataset, sdf in df.groupby("dataset"):
        plt.figure(figsize=(7, 4))
        agg = sdf.groupby(["builder", "dbar"])["MCC"].agg(["mean", "std"]).reset_index()
        for b, bdf in agg.groupby("builder"):
            plt.errorbar(bdf["dbar"], bdf["mean"], yerr=bdf["std"].fillna(0), marker="o", label=b)
        plt.title(f"MCC vs dbar | {dataset}")
        plt.xlabel("dbar")
        plt.ylabel("MCC")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(args.out_dir / f"fig1_mcc_vs_dbar_{dataset}.png", dpi=160)
        plt.close()

    # Fig2 cost vs dbar
    for dataset, sdf in df.groupby("dataset"):
        plt.figure(figsize=(7, 4))
        agg = sdf.groupby(["builder", "dbar"])["total_cost"].mean().reset_index()
        for b, bdf in agg.groupby("builder"):
            plt.plot(bdf["dbar"], bdf["total_cost"], marker="o", label=b)
        plt.title(f"Cost vs dbar | {dataset}")
        plt.xlabel("dbar")
        plt.ylabel("build_time + epoch_time")
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(args.out_dir / f"fig2_cost_vs_dbar_{dataset}.png", dpi=160)
        plt.close()

    # Fig3 pareto
    for dataset, sdf in df.groupby("dataset"):
        agg = sdf.groupby(["builder", "dbar"], as_index=False).agg(MCC=("MCC", "mean"), total_cost=("total_cost", "mean"))
        plt.figure(figsize=(6, 4))
        plt.scatter(agg["total_cost"], agg["MCC"]) 
        for _, r in agg.iterrows():
            plt.annotate(f"{r['builder']}@{int(r['dbar'])}", (r["total_cost"], r["MCC"]), fontsize=6)
        plt.xlabel("total_cost")
        plt.ylabel("MCC")
        plt.title(f"Pareto (MCC vs cost) | {dataset}")
        plt.tight_layout()
        plt.savefig(args.out_dir / f"fig3_pareto_{dataset}.png", dpi=160)
        plt.close()

    # Table 1 top-1/top-2
    table_rows = []
    for (dataset, dbar), g in df.groupby(["dataset", "dbar"]):
        rank = g.groupby("builder", as_index=False).agg(MCC=("MCC", "mean"), total_cost=("total_cost", "mean"))
        rank = rank.sort_values(["MCC", "total_cost"], ascending=[False, True]).head(2)
        for i, (_, r) in enumerate(rank.iterrows(), start=1):
            table_rows.append({"dataset": dataset, "dbar": dbar, "rank": i, "builder": r["builder"], "MCC": r["MCC"], "total_cost": r["total_cost"]})
    pd.DataFrame(table_rows).to_csv(args.out_dir / "table1_top_builders.csv", index=False)
    print(f"Saved plots/table to {args.out_dir}")


if __name__ == "__main__":
    main()
