from __future__ import annotations

from typing import Tuple, Union

import numpy as np

def project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / w.size

def clip_box(w: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(w, dtype=float), low, high)

def turnover(prev: np.ndarray, new: np.ndarray) -> float:
    prev = np.asarray(prev, dtype=float)
    new = np.asarray(new, dtype=float)
    return float(np.abs(prev - new).sum() / 2.0)

def apply_bounds(
    X_next: np.ndarray, bounds: Union[Tuple[np.ndarray, np.ndarray], str, None] = None
) -> np.ndarray:
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

def project_capped_simplex(
    w: np.ndarray, total: float = 1.0, cap: float = 0.05
) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, cap)
    n = w.size
    if np.isclose(w.sum(), total):
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = -1
    theta = 0.0
    for j in range(n):
        theta = (cssv[j] - total) / (j + 1)
        if u[j] - theta > 0:
            rho = j
    theta = (cssv[rho] - total) / (rho + 1) if rho >= 0 else 0.0
    x = np.clip(w - theta, 0.0, cap)
    s = x.sum()
    if not np.isclose(s, total):
        if s > 0:
            x *= total / s
            x = np.clip(x, 0.0, cap)
    return x