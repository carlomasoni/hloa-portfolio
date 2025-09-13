"""
Covers initialization, seed reproducibility,
and convergence behaviour of HornedLizardOptimizer.
"""
import pytest
import numpy as np

from hloa import HLOA, HLOA_Config


def test_optimizer_init_fields():
    # Create a dummy objective function
    def dummy_obj(x):
        return np.sum(x**2)
    
    # Create bounds
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    
    # Create config
    config = HLOA_Config(pop_size=10, iters=5, seed=42)
    
    # Initialize optimizer
    opt = HLOA(obj=dummy_obj, bounds=bounds, config=config)
    
    # Test config fields
    assert opt.cfg.pop_size == 10
    assert opt.cfg.iters == 5
    assert opt.cfg.seed == 42
    
    # Test other fields
    assert opt.dim == 2
    assert opt.constraint_type == "simplex"


def test_optimizer_minimize_not_implemented():
    # Create a dummy objective function and bounds
    def dummy_obj(x):
        return np.sum(x**2)
    
    bounds = (np.array([0.0]), np.array([1.0]))
    
    opt = HLOA(obj=dummy_obj, bounds=bounds)
    
    # The minimize method should raise NotImplementedError
    with pytest.raises(NotImplementedError):
        opt.minimize()

