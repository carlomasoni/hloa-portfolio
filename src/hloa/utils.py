"""
utils.py

Utility functions to support HLOA, including:

- RNG seeding and reproducibility
- Population initialisation helpers
- Projection shortcuts (simple clamps, random init)
- Numerical stability utilities
"""

from __future__ import annotations

import numpy as np


def rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy Generator with an optional seed."""
    return np.random.default_rng(seed)


def random_simplex(rng: np.random.Generator, n: int, size: int = 1):
    """Draw weights on the probability simplex."""
    if size == 1:
        x = rng.random(n)
        return x / x.sum()
    X = rng.random((size, n))
    return X / X.sum(axis=1, keepdims=True)
