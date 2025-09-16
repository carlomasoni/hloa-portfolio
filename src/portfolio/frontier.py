"""
Efficient frontier construction and wrappers:

- Build portfolios across lambda/target-return grids
- Compute risk, return, Sharpe for each solution
- Identify and return max-Sharpe portfolio

-> read up on algos and correct current work cus not fully sure if this is right... 
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .objectives import sharpe_ratio
from .constraints import project_capped_simplex
def max_sharpe_capped(mu: pd.Series, cov: pd.DataFrame, cap: float = 0.05) -> np.ndarray:
    n = mu.size
    # Start from equal weights under cap
    w = np.full(n, min(1.0 / n, cap), dtype=float)
    s = w.sum()
    if not np.isclose(s, 1.0):
        w = project_capped_simplex(w, total=1.0, cap=cap)
    # Simple projected ascent
    step = 0.05
    for _ in range(500):
        grad = mu.values / (np.sqrt(w @ cov.values @ w) + 1e-12) - (
            (w @ mu.values) * (cov.values @ w)
        ) / (np.power(w @ cov.values @ w, 1.5) + 1e-12)
        w = w + step * grad
        w = project_capped_simplex(w, total=1.0, cap=cap)
        step *= 0.995
    return w


def _project_simplex(w: np.ndarray) -> np.ndarray:
    w = np.maximum(w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / w.size


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, n_trials: int = 2000, seed: int | None = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = mu.size
    best_w = np.ones(n) / n
    best_s = -np.inf
    for _ in range(n_trials):
        w = _project_simplex(rng.random(n))
        s = sharpe_ratio(w, mu, cov, rf=0.0)
        if s > best_s:
            best_s = s
            best_w = w
    return best_w


def sample_frontier(mu: pd.Series, cov: pd.DataFrame, n: int = 25, seed: int | None = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_assets = mu.size
    W = np.zeros((n, n_assets))
    R = np.zeros(n)
    V = np.zeros(n)
    for i in range(n):
        w = _project_simplex(rng.random(n_assets))
        W[i] = w
        R[i] = float(w @ mu.values)
        V[i] = float(np.sqrt(w @ cov.values @ w))
    return W, R, V
