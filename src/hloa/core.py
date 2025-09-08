"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer`.
"""


class HornedLizardOptimizer:
    def __init__(self, pop_size=50, iters=500, seed=None, **kwargs):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed

    def minimize(self, func, bounds, constraints=None, callback=None):
        """Placeholder until the algorithm is implemented."""
        raise NotImplementedError("HLOA minimize() not yet implemented.")





