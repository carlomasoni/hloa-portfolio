from __future__ import annotations

from typing import Tuple, Union

import numpy as np


def project_capped_simplex(
    w: np.ndarray, total: float = 1.0, cap: float = 0.05
) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, cap)
    
    if np.isclose(w.sum(), total):
        return w
    
    current_sum = w.sum()
    
    if current_sum > 0:
        scale_factor = total / current_sum
        x = w * scale_factor
        
        if x.max() > cap:
            x = np.clip(x, 0.0, cap)
            new_sum = x.sum()
            if new_sum > 0:
                x *= total / new_sum
                x = np.clip(x, 0.0, cap)
        return x
    else:
        n = w.size
        equal_weight = min(total / n, cap)
        x = np.full(n, equal_weight)
        remaining = total - x.sum()
        if remaining > 0:
            x[0] += remaining
            x = np.clip(x, 0.0, cap)
        return x



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


