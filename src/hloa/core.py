"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer`.
"""

from token import OP
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

from pandas._config import config
from pandas.core.common import consensus_name_attr

from portfolio.constraints import project_simplex, clip_box, turnover

@dataclass

class HLOA_Config:
    def __init__(self, pop_size=50, iters=500, seed=None, **kwargs):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed

class HLOA: 
    def __init__ (self,
                obj: Callable[[np.ndarray], float],
                bounds: Tuple[np.ndarray, np.ndarray],
                constraint_type: str = "simplex",  # "simplex", "box", or "none"
                config: HLOA_Config = HLOA_Config()):
        self.obj = obj
        self.lb, self.ub = bounds
        self.dim = self.lb.size
        self.constraint_type = constraint_type
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

    def _clip(self, X): 
        return np.clip(X, self.lb, self.ub)
    
    def _apply_constraints(self, X: np.ndarray) -> np.ndarray:
        if self.constraint_type == "simplex":
            return project_simplex(X)
        elif self.constraint_type == "box":
            return clip_box(X, self.lb, self.ub)
        else:  # "none"
            return self._clip(X)
    
    def optimize(self):
        pass 









