from typing import Generator

import numpy as np

from portfolio.constraints import apply_bounds


def sigma(rng: np.random.Generator) -> int:
    return int(rng.random() > 0.5)


def crypsis(
    X: np.ndarray,
    X_best: np.ndarray,
    t: int,
    max_iter: int,
    bounds: tuple | None = None,
    rng: Generator | None = None,
    c1: float = 1.0,
    c2: float = 0.5,
    delta: float = 2.0,
    decay_eps: float = 0.1,
    sigma_func=sigma,
) -> np.ndarray:
    if rng is None:
        rng = np.random.Generator(np.random.PCG64())

    n, d = X.shape

    r_idx = np.empty((n, 4), dtype=int)
    all_idx = np.arange(n)
    for i in range(n):
        if n >= 4:
            r_idx[i] = rng.permutation(all_idx)[:4]
        else:
            r_idx[i] = rng.choice(all_idx, size=4, replace=True)

    r1, r2, r3, r4 = r_idx.T

    map1 = c1 * (np.sin(X[r1]) - np.cos(X[r2]))
    map2 = c2 * (np.cos(X[r3]) - np.sin(X[r4]))

    sigma_values = np.array([sigma_func(rng) for _ in range(n)])
    new_map2 = ((-1.0) ** sigma_values)[:, None] * map2

    amplitude_decay = delta * ((1 - (t + 1) / max_iter) + (decay_eps / max_iter))

    X_next = X_best[None, :] + amplitude_decay * (map1 + new_map2)

    if bounds is not None:
        X_next = apply_bounds(X_next, bounds)

    return X_next


def skin_lord(
    X: np.ndarray,
    X_best: np.ndarray,
    idx_worst: int,
    rng: Generator | None = None,
    sigma_func=sigma,
    bounds: tuple | None = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.Generator(np.random.PCG64())

    n, d = X.shape
    if n >= 4:
        r1, r2, r3, r4 = rng.choice(np.arange(n), size=4, replace=False)
    else:
        r1, r2, r3, r4 = rng.choice(np.arange(n), size=4, replace=True)

    light_values = [0.0, 0.404661]
    dark_values = [0.544510, 1.0]

    l1, l2 = rng.choice(light_values, size=2)
    d1, d2 = rng.choice(dark_values, size=2)

    s = sigma_func(rng)
    s12 = np.sin(X[r1] - X[r2])
    s34 = np.sin(X[r3] - X[r4])

    l_agent = (
        X_best
        + 0.5 * l1 * s12
        - ((-1.0) ** s) * (0.5 * l2 * s34)
    )
    d_agent = (
        X_best
        + 0.5 * d1 * s12
        - ((-1.0) ** s) * (0.5 * d2 * s34)
)

    if rng.random() < 0.5:
        new_agent = l_agent
    else:
        new_agent = d_agent

    X_new = X.copy()
    X_new[idx_worst] = new_agent

    if bounds is not None:
        X_new[idx_worst : idx_worst + 1] = apply_bounds(
            X_new[idx_worst : idx_worst + 1], bounds
        )

    return X_new


def blood_squirt(
    X: np.ndarray,
    X_best: np.ndarray,
    t: int,
    max_iter: int,
    v0: float = 1.0,
    alpha: float = np.pi / 2,
    g: float = 9.807e-3,
    epsilon: float = 1e-6,
    bounds: tuple | None = None,
) -> np.ndarray:
    a = v0 * np.cos(alpha * (t / max_iter)) + epsilon
    b = v0 * np.sin(alpha - alpha * (t / max_iter)) - g + epsilon

    X_next = a * X_best[None, :] + b * X

    if bounds is not None:
        X_next = apply_bounds(X_next, bounds)


    return X_next


def move_to_escape(
    X: np.ndarray,
    X_best: np.ndarray,
    rng: Generator,
    bounds: tuple | None = None,
    clip: float | None = 10.0,
) -> np.ndarray:
    n, d = X.shape

    walk = rng.uniform(-1.0, 1.0, size=(n, 1))

    epsilon = rng.standard_cauchy(size=(n, 1))
    if clip is not None:
        epsilon = np.clip(epsilon, -clip, clip)

    X_next = X_best[None, :] + walk * ((0.5 - epsilon) * X)

    if bounds is not None:
        X_next = apply_bounds(X_next, bounds)
    return X_next


def alpha_msh(
    X: np.ndarray,
    fitness: np.ndarray,
    rng: Generator,
    threshold: float = 0.3,
    strategy: str = "uniform",
    X_best: np.ndarray | None = None,
    bounds: tuple | None = None,
    mix_strength: float = 0.5,
    sigma_func=sigma,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = X.shape

    f_min = np.min(fitness)
    f_max = np.max(fitness)

    if f_max == f_min:
        msh = np.ones_like(fitness, dtype=float)
    else:
        msh = (f_max - fitness) / (f_max - f_min)

    reset_condition = msh < threshold
    if not np.any(reset_condition):
        return X, msh, reset_condition

    idx_best = int(np.argmax(fitness))
    X_best = X[idx_best]

    m = int(np.sum(reset_condition))
    r1 = rng.integers(0, n, size=m)
    r2 = rng.integers(0, n - 1, size=m)
    r2 = r2 + (r2 >= r1)

    s = np.fromiter((sigma_func(rng) for _ in range(m)), dtype=int)

    new = X_best[None, :] + 0.5 * (X[r1] - ((-1.0) ** s)[:, None] * X[r2])

    X_new = X.copy()
    X_new[reset_condition] = new

    if bounds is not None:
        X_new = apply_bounds(X_new, bounds)

    return X_new, msh, reset_condition
