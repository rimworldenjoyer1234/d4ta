"""Dataset loading and chunk iterators."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

KDD_FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]


def iter_csv_chunks(path: Path, chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield chunks for CSV datasets with header."""
    yield from pd.read_csv(path, chunksize=chunk_size, low_memory=False)


def iter_kdd_chunks(train_path: Path, test_path: Optional[Path], chunk_size: int) -> Iterator[pd.DataFrame]:
    """Yield chunks for NSL-KDD train+test files."""

    def _reader(path: Path) -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(path, header=None, chunksize=chunk_size, low_memory=False):
            ncols = chunk.shape[1]
            if ncols == 42:
                cols = KDD_FEATURE_COLUMNS + ["label"]
            elif ncols >= 43:
                cols = KDD_FEATURE_COLUMNS + ["label", "difficulty"] + [f"extra_{i}" for i in range(ncols - 43)]
            else:
                raise ValueError(f"Unexpected NSL-KDD column count {ncols} in {path}")
            chunk.columns = cols
            yield chunk

    yield from _reader(train_path)
    if test_path is not None:
        yield from _reader(test_path)
