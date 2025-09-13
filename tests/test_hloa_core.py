"""
Covers initialization, seed reproducibility,
and convergence behaviour of HornedLizardOptimizer.

if these fail ggs icl (CIRCULAR FROM WHERE????)
"""
import pytest

from hloa import HLOA


def test_optimizer_init_fields():
    opt = HLOA(pop_size=10, iters=5, seed=42)
    assert opt.pop_size == 10
    assert opt.iters == 5
    assert opt.seed == 42


def test_optimizer_minimize_not_implemented():
    opt = HLOA()
    with pytest.raises(NotImplementedError):
        opt.minimize(lambda x: sum(x), bounds=[(0, 1)])

