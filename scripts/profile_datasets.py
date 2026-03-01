#!/usr/bin/env python3
"""Profile UNSW-NB15 NetFlow, ToN-IoT NetFlow, and NSL-KDD datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.datasets.loaders import iter_csv_chunks, iter_kdd_chunks
from src.datasets.profiler import profile_from_chunks
from src.profiling.profile import write_profile_json
from src.profiling.schema import write_schema_yaml
from src.utils.config import ProfilingConfig
from src.utils.logging_utils import setup_logging
from src.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


def _write_outputs(out_dir: Path, dataset_name: str, payload: dict) -> None:
    profiles_dir = out_dir / "profiles"
    schemas_dir = out_dir / "schemas"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    schemas_dir.mkdir(parents=True, exist_ok=True)

    write_profile_json(str(profiles_dir / f"{dataset_name}.json"), payload["profile"])
    write_schema_yaml(str(schemas_dir / f"{dataset_name}.yaml"), payload["schema"])


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsw", type=Path, required=True, help="Path to NF-UNSW-NB15-v2.csv")
    parser.add_argument("--ton", type=Path, required=True, help="Path to NF-ToN-IoT-v2.csv")
    parser.add_argument("--kdd_train", type=Path, required=True, help="Path to KDDTrain+.txt")
    parser.add_argument("--kdd_test", type=Path, required=True, help="Path to KDDTest+.txt")
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"), help="Output artifacts directory")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk_size", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    """Run end-to-end dataset profiling."""
    args = parse_args()
    setup_logging()
    set_global_seed(args.seed)

    cfg = ProfilingConfig(chunk_size=args.chunk_size, out_dir=args.out_dir)
    LOGGER.info("Starting profiling with chunk_size=%d", cfg.chunk_size)

    unsw_payload = profile_from_chunks(
        dataset_name="unsw_nb15_netflow",
        chunks=iter_csv_chunks(args.unsw, cfg.chunk_size),
        top_k_categories=cfg.top_k_categories,
        sample_values_per_col=cfg.sample_values_per_col,
        label_mode="csv",
        ip_columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    )
    _write_outputs(cfg.out_dir, "unsw_nb15_netflow", unsw_payload)

    ton_payload = profile_from_chunks(
        dataset_name="ton_iot_netflow",
        chunks=iter_csv_chunks(args.ton, cfg.chunk_size),
        top_k_categories=cfg.top_k_categories,
        sample_values_per_col=cfg.sample_values_per_col,
        label_mode="csv",
        ip_columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR"],
    )
    _write_outputs(cfg.out_dir, "ton_iot_netflow", ton_payload)

    kdd_payload = profile_from_chunks(
        dataset_name="nsl_kdd",
        chunks=iter_kdd_chunks(args.kdd_train, args.kdd_test, cfg.chunk_size),
        top_k_categories=cfg.top_k_categories,
        sample_values_per_col=cfg.sample_values_per_col,
        label_mode="kdd",
    )
    _write_outputs(cfg.out_dir, "nsl_kdd", kdd_payload)

    for name, payload in [
        ("unsw_nb15_netflow", unsw_payload),
        ("ton_iot_netflow", ton_payload),
        ("nsl_kdd", kdd_payload),
    ]:
        prof = payload["profile"]
        LOGGER.info(
            "Summary | %s | rows=%s cols=%s mem_est=%sMB",
            name,
            prof["rows"],
            prof["columns"],
            round(prof["estimated_memory_bytes"] / (1024 * 1024), 2),
        )


if __name__ == "__main__":
    main()
