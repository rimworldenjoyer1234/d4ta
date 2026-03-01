"""Feature preprocessing fit/transform helpers for RQ1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessArtifacts:
    """Persisted preprocessing outputs."""

    model_path: Path
    feature_names_path: Path


def _log1p_clip(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x.astype(float), a_min=0, a_max=None))


def _safe_feature_names(preprocessor: ColumnTransformer, fallback_cols: Sequence[str]) -> list[str]:
    """Get feature names from sklearn preprocessor with backward-compatible fallback.

    Older sklearn versions may fail when a pipeline step (e.g., FunctionTransformer)
    does not expose ``get_feature_names_out``. In that case we return deterministic
    fallback names based on the transformed width.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return [f"f_{i:04d}_{col}" for i, col in enumerate(fallback_cols)]


def fit_netflow_preprocessor(
    train_df: pd.DataFrame,
    feature_cols: Sequence[str],
    log1p_cols: Iterable[str],
    out_dir: Path,
) -> PreprocessArtifacts:
    """Fit netflow numeric-only preprocessor on train pool."""
    out_dir.mkdir(parents=True, exist_ok=True)
    log1p_cols = [c for c in log1p_cols if c in feature_cols]
    passthrough_cols = [c for c in feature_cols if c not in log1p_cols and c not in {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}]
    log_cols = [c for c in log1p_cols if c not in {"IPV4_SRC_ADDR", "IPV4_DST_ADDR"}]

    ct = ColumnTransformer(
        transformers=[
            ("logscale", Pipeline([("log1p", FunctionTransformer(_log1p_clip, validate=False)), ("std", StandardScaler())]), log_cols),
            ("std", Pipeline([("std", StandardScaler())]), passthrough_cols),
        ],
        remainder="drop",
    )
    ct.fit(train_df)
    model_path = out_dir / "preprocessor.joblib"
    names_path = out_dir / "feature_names.joblib"
    joblib.dump(ct, model_path)
    fallback_cols = list(log_cols) + list(passthrough_cols)
    joblib.dump(_safe_feature_names(ct, fallback_cols), names_path)
    return PreprocessArtifacts(model_path, names_path)


def fit_kdd_preprocessor(
    train_df: pd.DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
    out_dir: Path,
) -> PreprocessArtifacts:
    """Fit KDD preprocessor (one-hot + standardized numeric)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ct = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), list(categorical_cols)),
            ("num", StandardScaler(), list(numeric_cols)),
        ],
        remainder="drop",
    )
    ct.fit(train_df)
    model_path = out_dir / "preprocessor.joblib"
    names_path = out_dir / "feature_names.joblib"
    joblib.dump(ct, model_path)
    fallback_cols = list(categorical_cols) + list(numeric_cols)
    joblib.dump(_safe_feature_names(ct, fallback_cols), names_path)
    return PreprocessArtifacts(model_path, names_path)


def transform_features(df: pd.DataFrame, preprocessor_path: Path) -> np.ndarray:
    """Transform dataframe into model features."""
    pre = joblib.load(preprocessor_path)
    out = pre.transform(df)
    return np.asarray(out, dtype=np.float32)
