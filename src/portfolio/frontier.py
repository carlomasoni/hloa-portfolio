"""
frontier.py

Efficient frontier construction and wrappers:

- Build portfolios across lambda/target-return grids
- Compute risk, return, Sharpe for each solution
- Identify and return max-Sharpe portfolio
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def max_sharpe(
    mu: pd.Series, cov: pd.DataFrame, n_trials: int = 20000, seed: int | None = None
) -> np.ndarray:
    """Brute-force max-Sharpe by random sampling on the simplex."""
    g = np.random.default_rng(seed)
    n = len(mu)
    best_w = np.zeros(n)
    best_sr = float("-inf")
    for _ in range(n_trials):
        w = g.random(n)
        w /= w.sum()
        ret = float(np.dot(w, mu))
        vol = float(np.sqrt(np.dot(w, cov.values @ w)))
        if vol > 0:
            sr = ret / vol
            if sr > best_sr:
                best_sr, best_w = sr, w
    return best_w


def sample_frontier(
    mu: pd.Series, cov: pd.DataFrame, n: int = 100, seed: int | None = None
):
    """Return (weights, returns, vols) sampled on the simplex."""
    g = np.random.default_rng(seed)
    W, R, V = [], [], []
    for _ in range(n):
        w = g.random(len(mu))
        w /= w.sum()
        r = float(np.dot(w, mu))
        v = float(np.sqrt(np.dot(w, cov.values @ w)))
        W.append(w)
        R.append(r)
        V.append(v)
    return np.array(W), np.array(R), np.array(V)
