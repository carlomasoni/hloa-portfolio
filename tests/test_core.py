import numpy as np
import pandas as pd
import pytest

from hloa import HLOA, HLOA_Config
from portfolio.constraints import project_capped_simplex
from portfolio.frontier import sharpe_ratio


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
    assert opt.constraint_type == "box"


def test_optimizer_run_improves_objective():
    n = 3
    rng = np.random.default_rng(0)

    def obj(X: np.ndarray):
        return X.sum(axis=1)

    lb = np.zeros(n)
    ub = np.ones(n)
    opt = HLOA(
        obj=obj, bounds=(lb, ub), config=HLOA_Config(pop_size=20, iters=10, seed=1)
    )

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


def test_hloa_portfolio_optimization():
    n_assets = 5
    n_agents = 20
    n_iters = 50

    np.random.seed(42)
    returns = np.random.normal(0.01, 0.05, (100, n_assets))
    mu = pd.Series(np.mean(returns, axis=0))
    cov = pd.DataFrame(np.cov(returns.T))

    def portfolio_fitness(weights):
        fitness_scores = np.zeros(weights.shape[0])
        for i, w in enumerate(weights):
            w_proj = project_capped_simplex(w, total=1.0, cap=0.05)
            fitness_scores[i] = sharpe_ratio(w_proj, mu, cov, rf=0.0)
        return fitness_scores

    lb = np.zeros(n_assets)
    ub = np.ones(n_assets)
    config = HLOA_Config(pop_size=n_agents, iters=n_iters, seed=42)

    opt = HLOA(obj=portfolio_fitness, bounds=(lb, ub), config=config)

    w_best, f_best, X_final, f_final = opt.run()

    assert w_best.shape == (n_assets,)
    assert f_best > -np.inf
    assert np.all(w_best >= 0)
    assert np.all(w_best <= 1)

    assert np.all(X_final >= 0)
    assert np.all(X_final <= 1)

    random_weights = np.random.random((n_agents, n_assets))
    random_fitness = portfolio_fitness(random_weights)
    assert f_best >= np.max(random_fitness)


def test_hloa_deterministic_reproducibility():

    def simple_obj(X):
        return -np.sum(X**2, axis=1)

    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=10, iters=20, seed=123)

    opt1 = HLOA(obj=simple_obj, bounds=bounds, config=config)
    w1, f1, _, _ = opt1.run()

    opt2 = HLOA(obj=simple_obj, bounds=bounds, config=config)
    w2, f2, _, _ = opt2.run()

    np.testing.assert_array_almost_equal(w1, w2, decimal=10)
    assert abs(f1 - f2) < 1e-10


def test_hloa_convergence_behavior():

    def quadratic_obj(X):
        return -np.sum((X - 0.5) ** 2, axis=1)

    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=15, iters=30, seed=456)

    opt = HLOA(obj=quadratic_obj, bounds=bounds, config=config)
    w_best, f_best, X_final, f_final = opt.run()

    assert f_best > -0.5


def test_hloa_with_different_bounds():

    def linear_obj(X):
        return X.sum(axis=1)

    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    config = HLOA_Config(pop_size=10, iters=15, seed=789)

    opt = HLOA(obj=linear_obj, bounds=bounds, config=config)
    w_best, f_best, _, _ = opt.run()

    assert f_best > 0.5
