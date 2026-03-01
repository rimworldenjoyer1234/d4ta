"""Schema inference and serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import yaml

from src.profiling.streaming_stats import ColumnTracker


@dataclass(frozen=True)
class TypeThresholds:
    """Heuristics for type inference."""

    bool_cardinality: int = 2
    categorical_small_max: int = 50
    categorical_high_max_ratio: float = 0.2


def infer_column_type(name: str, tracker: ColumnTracker, thresholds: TypeThresholds) -> str:
    """Infer semantic column type from aggregate stats."""
    lname = name.lower()
    if "ip" in lname and "addr" in lname:
        return "ip"

    card = tracker.cardinality()
    numeric_count = tracker.numeric.count
    observed = tracker.seen - tracker.missing
    numeric_ratio = (numeric_count / observed) if observed > 0 else 0.0

    if numeric_ratio > 0.98:
        if card <= thresholds.bool_cardinality:
            return "bool"
        return "int" if all(v.lstrip("-").isdigit() for v in tracker.unique_values if v) else "float"

    if card <= thresholds.categorical_small_max:
        return "categorical"

    unique_ratio = (card / observed) if observed > 0 else 0.0
    if unique_ratio > thresholds.categorical_high_max_ratio:
        return "string"
    return "categorical"


def build_schema(dataset_name: str, columns: Iterable[ColumnTracker]) -> Dict[str, Any]:
    """Build machine-readable schema document."""
    thresholds = TypeThresholds()
    schema_columns: List[Dict[str, Any]] = []
    for col in columns:
        inferred = infer_column_type(col.name, col, thresholds)
        col_info: Dict[str, Any] = {
            "name": col.name,
            "inferred_type": inferred,
            "missing_rate": col.missing_rate(),
            "cardinality": col.cardinality(),
            "example_values": col.sample_values,
        }
        if inferred in {"int", "float", "bool"}:
            col_info["numeric_stats"] = col.numeric.to_dict()
        schema_columns.append(col_info)

    return {"dataset": dataset_name, "columns": schema_columns}


def write_schema_yaml(path: str, schema: Dict[str, Any]) -> None:
    """Write schema YAML file."""
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(schema, f, sort_keys=False, allow_unicode=True)
