"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer` with:

- Population initialization
- Leader/elite selection
- Position update rules from the paper
- Stopping criteria and history tracking
"""

from __future__ import annotations
from typing import Protocol, Callable
import numpy as np
from .utils import random_simplex, rng as make_rng

class Objective(Protocol):
    def __call__(self, w: np.ndarray) -> float: ...

def optimize(objective: Objective, n_assets: int, iters: int = 2000, seed: int | None = None) -> np.ndarray:
    """
    Very simple baseline optimiser: random search on the simplex maximizing the objective.
    Replace with HLOA proper later.
    """
    g = make_rng(seed)
    best_w = random_simplex(g, n_assets)
    best_val = float(objective(best_w))
    for _ in range(iters):
        w = random_simplex(g, n_assets)
        v = float(objective(w))
        if v > best_val:
            best_w, best_val = w, v
    return best_w
