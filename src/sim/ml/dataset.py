"""
Dataset builder for telemetry anomaly detection.

Responsibilities:
- Load telemetry parquet
- Scale features
- Window time-series into fixed-length samples
"""

from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch


FEATURE_COLS = ["w0", "w1", "w2"]


def _resolve_feature_cols(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> List[str]:
    if feature_cols:
        return list(feature_cols)
    if all(col in df.columns for col in FEATURE_COLS):
        return FEATURE_COLS
    w_cols = [c for c in df.columns if c.startswith("w")]
    if w_cols:
        return sorted(w_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c != "fault"]


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def build_windows(df: pd.DataFrame):
    return df


def load_data(
    path: str = "telemetry.parquet",
    feature_cols: Optional[List[str]] = None,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, int, Dict[str, object]]:
    df = load_parquet(path)

    cols = _resolve_feature_cols(df, feature_cols)
    df = df.dropna(subset=cols).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - val_ratio))
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_x = train_df[cols].to_numpy(dtype=np.float32)
    val_x = val_df[cols].to_numpy(dtype=np.float32)

    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std == 0.0] = 1.0

    train_x = (train_x - mean) / std
    val_x = (val_x - mean) / std

    train_tensor = torch.from_numpy(train_x)
    val_tensor = torch.from_numpy(val_x)

    meta = {
        "feature_cols": cols,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "window_length": None,
        "val_fault": val_df["fault"].to_numpy().tolist() if "fault" in val_df.columns else None,
        "data_path": path,
    }

    return train_tensor, val_tensor, train_tensor.shape[1], meta
