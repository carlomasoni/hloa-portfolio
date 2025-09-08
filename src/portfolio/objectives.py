"""
objectives.py

Objective functions for optimization:

- Negative Sharpe ratio (for maximizers → minimizer form)
- Mean–variance trade-off with penalty parameter
- Penalty wrappers for turnover, leverage, or constraint violations
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0
) -> float:
    w = np.asarray(weights, dtype=float)
    ex = float(np.dot(w, (mu - rf)))
    vol = float(np.sqrt(np.dot(w, cov.values @ w)))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol


def mean_variance(
    weights: np.ndarray,
    mu: pd.Series,
    cov: pd.DataFrame,
    risk_aversion: float = 1.0,
    rf: float = 0.0,
) -> float:
    w = np.asarray(weights, dtype=float)
    ret = float(np.dot(w, (mu - rf)))
    var = float(np.dot(w, cov.values @ w))
    # maximise: return - 0.5 * lambda * variance
    return ret - 0.5 * risk_aversion * var
