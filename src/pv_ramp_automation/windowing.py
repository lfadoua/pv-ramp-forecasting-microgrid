from __future__ import annotations

import numpy as np
import pandas as pd


def build_direct_supervised_windows(df: pd.DataFrame, feature_columns: list[str], target_column: str, lookback: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Build direct windows where y is the single target value at t + horizon."""
    if lookback < 1 or horizon < 1:
        raise ValueError("lookback and horizon must be positive.")
    values = df[feature_columns].to_numpy(dtype=float)
    target = df[target_column].to_numpy(dtype=float)
    max_start = len(df) - lookback - horizon + 1
    if max_start <= 0:
        return np.empty((0, lookback, len(feature_columns))), np.empty((0,))
    x = np.stack([values[i : i + lookback] for i in range(max_start)])
    y = np.asarray([target[i + lookback + horizon - 1] for i in range(max_start)], dtype=float)
    return x, y
