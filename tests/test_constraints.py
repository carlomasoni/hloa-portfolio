import numpy as np
from portfolio.constraints import project_simplex, clip_box, turnover

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
