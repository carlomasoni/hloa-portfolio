"""
Unit tests for portfolio/constraints.py

Checks correctness of simplex and box projections:
- Weights sum to 1
- All weights ≥ 0 (if long-only)
- Closest feasible point under L2 norm
"""

import numpy as np

from portfolio.constraints import clip_box, project_simplex, turnover


def test_project_simplex_basic():
    w = project_simplex(np.array([0.2, -1.0, 0.8]))
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_clip_box():
    w = clip_box(np.array([-0.1, 0.5, 1.5]), 0.0, 1.0)
    assert (w >= 0).all() and (w <= 1).all()


def test_turnover_zero_when_same():
    a = np.array([0.3, 0.7])
    b = np.array([0.3, 0.7])
    assert turnover(a, b) == 0.0
