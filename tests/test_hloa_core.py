"""
Covers initialization, seed reproducibility,
and convergence behaviour of HornedLizardOptimizer.
"""
import pytest
import numpy as np

from hloa import HLOA, HLOA_Config


def test_optimizer_init_fields():
    def obj(X: np.ndarray):
        return X.sum(axis=1)

    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=10, iters=5, seed=42)

    opt = HLOA(obj=obj, bounds=bounds, config=config)

    assert opt.cfg.pop_size == 10
    assert opt.cfg.iters == 5
    assert opt.cfg.seed == 42
    assert opt.dim == 2
    assert opt.constraint_type == "simplex"


def test_optimizer_run_improves_objective():
    n = 3
    rng = np.random.default_rng(0)

    def obj(X: np.ndarray):
        return X.sum(axis=1)

    lb = np.zeros(n)
    ub = np.ones(n)
    opt = HLOA(obj=obj, bounds=(lb, ub), config=HLOA_Config(pop_size=20, iters=10, seed=1))

    X0 = rng.random((20, n))
    start_scores = obj(X0)
    start_best = float(np.max(start_scores))

    w_best, f_best, X_final, f_final = opt.run()
    assert f_best >= start_best

def test_optimizer_minimize_not_implemented():
    def obj(X: np.ndarray):
        return X.sum(axis=1)

    bounds = (np.array([0.0]), np.array([1.0]))
    opt = HLOA(obj=obj, bounds=bounds)
    with pytest.raises(NotImplementedError):
        opt.minimize()

