"""
- RNG seeding and reproducibility
- Pop. initialisation helpers
- Projection shortcuts (simple clamps, random init)
- Numerical stability utilities

WIP -> add after operands are done ]

"""

from __future__ import annotations
import numpy as np
import random
from typing import Optional



def rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(seed)




def random_simplex(rng: np.random.Generator, n: int, size: int = 1):
    if size == 1:
        x = rng.random(n)
        return x / x.sum()
    X = rng.random((size, n))
    return X / X.sum(axis=1, keepdims=True)

def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


