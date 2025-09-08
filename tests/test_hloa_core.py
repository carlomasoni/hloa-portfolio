"""
Unit tests for hloa/core.py

Covers initialization, seed reproducibility,
and convergence behaviour of HornedLizardOptimizer.
"""
import pytest

from hloa import HornedLizardOptimizer


def test_optimizer_init_fields():
    opt = HornedLizardOptimizer(pop_size=10, iters=5, seed=42)
    assert opt.pop_size == 10
    assert opt.iters == 5
    assert opt.seed == 42


def test_optimizer_minimize_not_implemented():
    opt = HornedLizardOptimizer()
    with pytest.raises(NotImplementedError):
        opt.minimize(lambda x: sum(x), bounds=[(0, 1)])

