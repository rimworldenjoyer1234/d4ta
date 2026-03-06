"""Streaming dataset profiler orchestrators."""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import pandas as pd

from src.preprocess.ip_utils import ipv4_to_int
from src.profiling.profile import build_profile
from src.profiling.schema import build_schema
from src.profiling.streaming_stats import ColumnTracker

LOGGER = logging.getLogger(__name__)


def _is_benign(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"benign", "0", "normal", "normal."}


def _kdd_binary(value: object) -> int:
    text = str(value).strip().lower().rstrip(".")
    return 0 if text == "normal" else 1


class StreamingProfiler:
    """Generic streaming profiler for chunked DataFrames."""

    def __init__(self, dataset_name: str, top_k_categories: int, sample_values_per_col: int):
        self.dataset_name = dataset_name
        self.top_k_categories = top_k_categories
        self.sample_values_per_col = sample_values_per_col
        self.trackers: Dict[str, ColumnTracker] = {}
        self.row_count = 0
        self.memory_bytes = 0
        self.label_distributions: Dict[str, Counter[str]] = {}
        self.ip_parse_attempts = Counter[str]()

    def _tracker(self, col: str) -> ColumnTracker:
        if col not in self.trackers:
            self.trackers[col] = ColumnTracker(name=col)
        return self.trackers[col]

    def consume_chunk(self, chunk: pd.DataFrame) -> None:
        """Consume one DataFrame chunk."""
        self.row_count += len(chunk)
        self.memory_bytes += int(chunk.memory_usage(deep=True).sum())

        for col in chunk.columns:
            self._tracker(col).update(chunk[col], sample_limit=self.sample_values_per_col)

    def consume_csv_labels(self, chunk: pd.DataFrame) -> None:
        """Track CSV labels and binary mapping."""
        if "Attack" in chunk.columns:
            self.label_distributions.setdefault("Attack", Counter()).update(chunk["Attack"].astype(str).tolist())

        label_col = "Label" if "Label" in chunk.columns else None
        if label_col is not None:
            raw_vals = chunk[label_col].astype(str).tolist()
            self.label_distributions.setdefault("Label", Counter()).update(raw_vals)
            binary = [str(0 if _is_benign(v) else 1) for v in raw_vals]
            self.label_distributions.setdefault("binary_label", Counter()).update(binary)

    def consume_kdd_labels(self, chunk: pd.DataFrame) -> None:
        """Track KDD labels and binary mapping."""
        raw = chunk["label"].astype(str).str.strip().str.rstrip(".")
        self.label_distributions.setdefault("label", Counter()).update(raw.tolist())
        binary = [str(_kdd_binary(v)) for v in raw.tolist()]
        self.label_distributions.setdefault("binary_label", Counter()).update(binary)

    def consume_ip_sanity(self, chunk: pd.DataFrame, ip_cols: Iterable[str]) -> None:
        """Track IP parsing success."""
        for col in ip_cols:
            if col not in chunk.columns:
                continue
            for value in chunk[col].tolist():
                self.ip_parse_attempts["total"] += 1
                if ipv4_to_int(value) is not None:
                    self.ip_parse_attempts["ok"] += 1

    def finalize(self) -> Dict[str, object]:
        """Finalize profile and schema dictionaries."""
        columns = list(self.trackers.values())
        schema = build_schema(self.dataset_name, columns)
        success_rate = (
            self.ip_parse_attempts["ok"] / self.ip_parse_attempts["total"]
            if self.ip_parse_attempts["total"]
            else 0.0
        )
        profile = build_profile(
            dataset_name=self.dataset_name,
            row_count=self.row_count,
            columns=columns,
            memory_bytes=self.memory_bytes,
            label_distributions={k: dict(v) for k, v in self.label_distributions.items()},
            sanity_checks={"ip_parsing_success_rate": success_rate},
            top_k_categories=self.top_k_categories,
        )

        LOGGER.info(
            "Profiled %s: rows=%d, columns=%d, ip_success=%.4f",
            self.dataset_name,
            self.row_count,
            len(columns),
            success_rate,
        )
        return {"schema": schema, "profile": profile}


def profile_from_chunks(
    dataset_name: str,
    chunks: Iterator[pd.DataFrame],
    top_k_categories: int,
    sample_values_per_col: int,
    label_mode: str,
    ip_columns: Optional[list[str]] = None,
) -> Dict[str, object]:
    """Profile a dataset from chunk iterator."""
    profiler = StreamingProfiler(dataset_name, top_k_categories, sample_values_per_col)
    for chunk in chunks:
        profiler.consume_chunk(chunk)
        if label_mode == "csv":
            profiler.consume_csv_labels(chunk)
        elif label_mode == "kdd":
            profiler.consume_kdd_labels(chunk)
        if ip_columns:
            profiler.consume_ip_sanity(chunk, ip_columns)

    return profiler.finalize()
