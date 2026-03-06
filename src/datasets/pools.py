"""Pool preparation utilities for RQ1 experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from src.datasets.loaders import KDD_FEATURE_COLUMNS, iter_csv_chunks, iter_kdd_chunks


@dataclass(frozen=True)
class PoolSizeConfig:
    """Cap sizes for train/val/test pools."""

    train: int = 400_000
    val: int = 100_000
    test: int = 100_000


@dataclass(frozen=True)
class PoolBuildConfig:
    """Config for deterministic pool building."""

    chunk_size: int = 50_000
    seed: int = 123
    sizes: PoolSizeConfig = PoolSizeConfig()


def _hash_to_split(global_idx: int, seed: int) -> str:
    payload = f"{seed}:{global_idx}".encode("utf-8")
    h = hashlib.md5(payload).hexdigest()
    v = int(h[:8], 16) / 0xFFFFFFFF
    if v < 0.7:
        return "train"
    if v < 0.85:
        return "val"
    return "test"


def _binary_from_csv_label(series: pd.Series) -> pd.Series:
    def _map(v: object) -> int:
        txt = str(v).strip().lower()
        return 0 if txt in {"benign", "0", "normal", "normal."} else 1

    return series.map(_map).astype(int)


def _binary_from_kdd_label(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.rstrip(".").str.lower().map(lambda x: 0 if x == "normal" else 1).astype(int)


def _cap_stratified(df: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(df) <= cap:
        return df
    cls0 = df[df["binary_label"] == 0]
    cls1 = df[df["binary_label"] == 1]
    min_each = min(len(cls0), len(cls1), cap // 2)
    take0 = min_each
    take1 = min_each
    remain = cap - (take0 + take1)
    extras = pd.concat([cls0.iloc[take0:], cls1.iloc[take1:]], ignore_index=False)
    pick_extra = extras.sample(n=remain, random_state=int(rng.integers(1, 2**31 - 1))) if remain > 0 and len(extras) > 0 else extras.iloc[:0]
    out = pd.concat(
        [
            cls0.sample(n=take0, random_state=int(rng.integers(1, 2**31 - 1))) if take0 > 0 else cls0.iloc[:0],
            cls1.sample(n=take1, random_state=int(rng.integers(1, 2**31 - 1))) if take1 > 0 else cls1.iloc[:0],
            pick_extra,
        ],
        ignore_index=True,
    )
    return out.sample(frac=1.0, random_state=int(rng.integers(1, 2**31 - 1))).reset_index(drop=True)


def _append_with_cap(store: pd.DataFrame, incoming: pd.DataFrame, cap: int, rng: np.random.Generator) -> pd.DataFrame:
    merged = pd.concat([store, incoming], ignore_index=True)
    return _cap_stratified(merged, cap=cap, rng=rng)


def build_netflow_pools(
    csv_path: Path,
    dataset_name: str,
    out_dir: Path,
    config: PoolBuildConfig,
    feature_cols: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    """Build deterministic split pools for UNSW/ToN-IoT netflow CSV."""
    if feature_cols is None:
        feature_cols = [
            "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
            "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
            "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
            "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
            "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
            "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
            "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
            "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE",
        ]

    entity_cols = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL"]
    keep = list(dict.fromkeys(list(feature_cols) + entity_cols + ["Label", "Attack"]))

    split_data = {"train": pd.DataFrame(), "val": pd.DataFrame(), "test": pd.DataFrame()}
    caps = {"train": config.sizes.train, "val": config.sizes.val, "test": config.sizes.test}
    rng = np.random.default_rng(config.seed)

    global_idx = 0
    for chunk in iter_csv_chunks(csv_path, chunk_size=config.chunk_size):
        chunk = chunk[keep].copy()
        chunk["binary_label"] = _binary_from_csv_label(chunk["Label"])
        chunk["multiclass_label"] = chunk["Attack"].astype(str)
        chunk["row_id"] = np.arange(global_idx, global_idx + len(chunk), dtype=np.int64)
        split = chunk["row_id"].map(lambda i: _hash_to_split(int(i), config.seed))
        for s in ("train", "val", "test"):
            part = chunk[split == s]
            if not part.empty:
                split_data[s] = _append_with_cap(split_data[s], part, caps[s], rng)
        global_idx += len(chunk)

    base = out_dir / "pools" / dataset_name
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for s, df in split_data.items():
        p = base / f"{s}.parquet"
        df.to_parquet(p, index=False)
        out[s] = p
    return out


def build_kdd_pools(train_path: Path, test_path: Path, out_dir: Path, config: PoolBuildConfig) -> Dict[str, Path]:
    """Build NSL-KDD pools."""
    frames = list(iter_kdd_chunks(train_path, test_path, chunk_size=config.chunk_size))
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    df["binary_label"] = _binary_from_kdd_label(df["label"])
    df["multiclass_label"] = df["label"]
    df["row_id"] = np.arange(len(df), dtype=np.int64)

    entity_cols = ["service", "protocol_type", "flag"]
    keep = KDD_FEATURE_COLUMNS + entity_cols + ["binary_label", "multiclass_label", "row_id"]
    keep = list(dict.fromkeys(keep))
    df = df[keep]

    split = df["row_id"].map(lambda i: _hash_to_split(int(i), config.seed))
    rng = np.random.default_rng(config.seed)
    split_data = {
        "train": _cap_stratified(df[split == "train"], config.sizes.train, rng),
        "val": _cap_stratified(df[split == "val"], config.sizes.val, rng),
        "test": _cap_stratified(df[split == "test"], config.sizes.test, rng),
    }

    base = out_dir / "pools" / "nsl_kdd"
    base.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    for s, sdf in split_data.items():
        p = base / f"{s}.parquet"
        sdf.to_parquet(p, index=False)
        out[s] = p
    return out
