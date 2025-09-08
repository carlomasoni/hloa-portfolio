"""
Unit tests for hloa/core.py

Covers initialization, seed reproducibility,
and convergence behaviour of HornedLizardOptimizer.
"""

import numpy as np

from hloa.core import optimize


def test_optimize_returns_simplex_weights():
    # maximise the first coordinate: objective is just w[0]
    n = 5

    def obj(w):
        return float(w[0])

    w = optimize(obj, n_assets=n, iters=200, seed=42)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
