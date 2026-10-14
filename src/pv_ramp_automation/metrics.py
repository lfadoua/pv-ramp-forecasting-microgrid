from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_pred), dtype=float)
    return float(math.sqrt(np.mean((yt - yp) ** 2)))


def official_nrmse(rmse_value: float, denominator: float = 2.0) -> float:
    return float(rmse_value / denominator)


def skill_rmse(model_rmse: float, persistence_rmse: float) -> float | None:
    if persistence_rmse == 0:
        return None
    return float(1.0 - (model_rmse / persistence_rmse))
