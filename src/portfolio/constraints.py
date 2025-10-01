from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd

def project_capped_simplex(w, cap=0.05, total=1.0, tol=1e-12, max_iter=100):

    w = np.asarray(w, float)
    N = w.size
    if N * cap + 1e-15 < total:
        raise ValueError(f"Infeasible cap: need cap >= {1.0/N:.6f}")
    lo = np.min(w) - cap
    hi = np.max(w)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x = np.minimum(cap, np.maximum(0.0, w - mid))
        s = x.sum()
        if abs(s - total) <= tol:
            break
        if s > total:
            lo = mid
        else:
            hi = mid
    x_sum = x.sum()
    if x_sum > 0:
        x *= total / x_sum
    np.clip(x, 0.0, cap, out=x)
    x *= total / max(x.sum(), 1e-16)
    return x
def sharpe_ratio(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> float:
    w = np.asarray(weights, dtype=float)
    ex = float(w @ (mu - rf))
    vol = float(np.sqrt(w @ cov.values @ w))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol


def apply_bounds(
    X_next: np.ndarray, bounds: Union[Tuple[np.ndarray, np.ndarray], str, None] = None) -> np.ndarray:
    if bounds is None:
        return X_next

    if isinstance(bounds, tuple) and len(bounds) == 2:
        lb, ub = bounds
        if lb.ndim == 1:
            lb = lb[None, :]
        if ub.ndim == 1:
            ub = ub[None, :]
        X_next = np.clip(X_next, lb, ub)

    elif bounds == "simplex":
        X_next = np.maximum(X_next, 0.0)
        row_sums = X_next.sum(axis=1, keepdims=True)
        X_next = np.where(row_sums > 0, X_next / row_sums, 1.0 / X_next.shape[1])

    elif isinstance(bounds, str) and bounds.startswith("simplex_"):
        if bounds == "simplex_long_only":
            X_next = np.maximum(X_next, 0.0)
            row_sums = X_next.sum(axis=1, keepdims=True)
            X_next = np.where(row_sums > 0, X_next / row_sums, 1.0 / X_next.shape[1])

        elif bounds == "simplex_long_short":
            row_sums = X_next.sum(axis=1, keepdims=True)
            X_next = np.where(row_sums != 0, X_next / row_sums, 1.0 / X_next.shape[1])

    return X_next





