"""Dataset profiling document builder."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable

from src.profiling.schema import infer_column_type, TypeThresholds
from src.profiling.streaming_stats import ColumnTracker


def _top_categories(tracker: ColumnTracker, k: int) -> Dict[str, int]:
    top = dict(tracker.value_counts.most_common(k))
    other = sum(tracker.value_counts.values()) - sum(top.values())
    if other > 0:
        top["__other__"] = other
    return top


def _recommended_plan(columns: Iterable[ColumnTracker], thresholds: TypeThresholds) -> Dict[str, Any]:
    out: Dict[str, list[str]] = defaultdict(list)
    for col in columns:
        ctype = infer_column_type(col.name, col, thresholds)
        if ctype in {"int", "float", "bool"}:
            out["numeric"].append(col.name)
        elif ctype == "categorical":
            if col.cardinality() <= thresholds.categorical_small_max:
                out["categorical_small"].append(col.name)
            else:
                out["categorical_high"].append(col.name)
        elif ctype == "ip":
            out["ip_columns"].append(col.name)
        else:
            out["categorical_high"].append(col.name)

        if col.cardinality() <= 1:
            out["drop_constant"].append(col.name)
    return dict(out)


def build_profile(
    dataset_name: str,
    row_count: int,
    columns: Iterable[ColumnTracker],
    memory_bytes: int,
    label_distributions: Dict[str, Dict[str, int]],
    sanity_checks: Dict[str, Any],
    top_k_categories: int,
) -> Dict[str, Any]:
    """Build profile document."""
    thresholds = TypeThresholds()
    cols = list(columns)
    column_profiles: Dict[str, Any] = {}
    for col in cols:
        observed = max(col.seen - col.missing, 1)
        column_profiles[col.name] = {
            "missing_rate": col.missing_rate(),
            "cardinality": col.cardinality(),
            "percent_unique": col.cardinality() / observed,
            "top_categories": _top_categories(col, top_k_categories),
            "numeric_stats": col.numeric.to_dict(),
        }

    return {
        "dataset": dataset_name,
        "rows": row_count,
        "columns": len(cols),
        "estimated_memory_bytes": memory_bytes,
        "label_distributions": label_distributions,
        "sanity_checks": sanity_checks,
        "column_profiles": column_profiles,
        "recommended_preprocessing_plan": _recommended_plan(cols, thresholds),
    }


def write_profile_json(path: str, profile: Dict[str, Any]) -> None:
    """Write profile JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)
