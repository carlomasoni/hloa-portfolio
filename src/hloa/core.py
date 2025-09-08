"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer`.
"""

from .core import HornedLizardOptimizer
__all__ = ["HornedLizardOptimizer"]


class HornedLizardOptimizer:
    def __init__(self, pop_size=50, iters=500, seed=None, **kwargs):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed

    def minimize(self, func, bounds, constraints=None, callback=None):
        raise NotImplementedError()








