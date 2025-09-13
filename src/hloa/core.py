"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer`.
"""

# libs
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

# cleanup
#from portfolio.constraints import project_simplex, clip_box, turnover
from hloa.ops import crypsis, sigma, skin_darken, skin_lighten, blood_squirt, alpha_msh, move_to_escape


@dataclass
class HLOA_Config:
    def __init__(self, pop_size=50, iters=500, seed=None, penalty_coefficient=1.0):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed
        self.penalty_coefficient = penalty_coefficient

class HLOA: 
    def __init__ (self,
                obj: Callable[[np.ndarray], float],
                bounds: Tuple[np.ndarray, np.ndarray],
                constraint_type: str = "simplex",  
                config: HLOA_Config = HLOA_Config()):
        self.obj = obj
        self.lb, self.ub = bounds
        self.dim = self.lb.size
        self.constraint_type = constraint_type
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)



            
            















