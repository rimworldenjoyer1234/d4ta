#!/usr/bin/env python3
"""Prepare deterministic split pools for RQ1 micrograph experiments."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.datasets.pools import PoolBuildConfig, PoolSizeConfig, build_kdd_pools, build_netflow_pools
from src.utils.logging_utils import setup_logging
from src.utils.seeding import set_global_seed

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unsw", type=Path, required=True)
    parser.add_argument("--ton", type=Path, required=True)
    parser.add_argument("--kdd_train", type=Path, required=True)
    parser.add_argument("--kdd_test", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--chunk_size", type=int, default=50_000)
    parser.add_argument("--train_cap", type=int, default=400_000)
    parser.add_argument("--val_cap", type=int, default=100_000)
    parser.add_argument("--test_cap", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_global_seed(args.seed)

    cfg = PoolBuildConfig(
        chunk_size=args.chunk_size,
        seed=args.seed,
        sizes=PoolSizeConfig(train=args.train_cap, val=args.val_cap, test=args.test_cap),
    )

    unsw = build_netflow_pools(args.unsw, "unsw_nb15_netflow", args.out_dir, cfg)
    ton = build_netflow_pools(args.ton, "ton_iot_netflow", args.out_dir, cfg)
    kdd = build_kdd_pools(args.kdd_train, args.kdd_test, args.out_dir, cfg)

    for name, out in (("unsw", unsw), ("ton", ton), ("kdd", kdd)):
        for split, path in out.items():
            LOGGER.info("pool=%s split=%s path=%s", name, split, path)


if __name__ == "__main__":
    main()
