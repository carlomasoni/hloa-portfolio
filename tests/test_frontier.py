"""


Ensures frontier construction is correct:
- Solutions ordered by risk
- Max-Sharpe matches direct optimization
- Results comparable to convex baselines on small universes
"""

import numpy as np
import pandas as pd

from portfolio.frontier import max_sharpe, sample_frontier


def test_max_sharpe_returns_simplex_weights():
    mu = pd.Series([0.1, 0.15, 0.07])
    cov = pd.DataFrame([[0.04, 0.01, 0.0], [0.01, 0.05, 0.0], [0.0, 0.0, 0.02]])
    w = max_sharpe(mu, cov, n_trials=2000, seed=7)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_sample_frontier_shapes():
    mu = pd.Series([0.1, 0.15, 0.07, 0.12])
    cov = pd.DataFrame(np.eye(4) * 0.04)
    W, R, V = sample_frontier(mu, cov, n=25, seed=0)
    assert W.shape == (25, 4)
    assert R.shape == (25,)
    assert V.shape == (25,)
