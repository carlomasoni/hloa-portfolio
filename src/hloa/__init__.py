"""
hloa package

Exposes the Horned Lizard Optimisation Algorithm (HLOA).
Import the optimizer class directly:

    from hloa import HornedLizardOptimizer
"""




from .core import optimize
from .utils import rng, random_simplex

__all__ = ["optimize", "rng", "random_simplex"]
