from __future__ import annotations
import numpy as np

def project_simplex(w: np.ndarray) -> np.ndarray:
    """Project onto simplex: w >= 0, sum w = 1."""
    w = np.maximum(np.asarray(w, dtype=float), 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones_like(w) / w.size

def clip_box(w: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.clip(np.asarray(w, dtype=float), low, high)

def turnover(prev: np.ndarray, new: np.ndarray) -> float:
    prev = np.asarray(prev, dtype=float)
    new = np.asarray(new, dtype=float)
    return float(np.abs(prev - new).sum() / 2.0)
