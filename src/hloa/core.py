from typing import Callable, Tuple

import numpy as np
from numpy.random import PCG64, Generator

from hloa.ops import alpha_msh, blood_squirt, crypsis, move_to_escape, sigma, skin_lord
from portfolio.constraints import apply_bounds


class HLOA_Config:
    def __init__(
        self,
        pop_size: int = 300,
        iters: int = 2000,
        seed: int | None = 42,
        p_mimic: float = 0.6,
        p_flee: float = 0.2,
        alpha_msh_threshold: float = 0.3,
    ):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed
        self.p_mimic = p_mimic
        self.p_flee = p_flee
        self.alpha_msh_threshold = alpha_msh_threshold


class HLOA:
    def __init__(
        self,
        obj: Callable[[np.ndarray], np.ndarray],
        bounds: Tuple[np.ndarray, np.ndarray] | str | None,
        config: HLOA_Config | None = None,
    ):
        self.fitness = obj
        self.bounds = bounds
        self.cfg = config if config is not None else HLOA_Config()
        self.rng: Generator = Generator(PCG64(self.cfg.seed))
        if isinstance(bounds, tuple):
            self.lb, self.ub = (np.asarray(b, dtype=float) for b in bounds)
            self.N = self.lb.size
            self.dim = self.N
            self.constraint_type = "simplex"
        else:
            self.lb = None
            self.ub = None
            self.N = None
            self.dim = 0
            self.constraint_type = "none"

    def _init_population(self) -> np.ndarray:
        if isinstance(self.bounds, tuple):
            X = self.rng.uniform(self.lb, self.ub, size=(self.cfg.pop_size, self.N))
        else:
            X = self.rng.random((self.cfg.pop_size, self.dim or 1))
        if self.bounds is not None:
            X = apply_bounds(X, self.bounds)
        return X

    def run(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        X = self._init_population()
        f = self.fitness(X)
        best_idx = int(np.argmax(f))
        w_best = X[best_idx].copy()
        f_best = float(f[best_idx])

        for i in range(self.cfg.iters):
            if self.rng.random() < self.cfg.p_mimic:
                X = crypsis(
                    X,
                    w_best,
                    i,
                    self.cfg.iters,
                    bounds=(
                        self.bounds
                        if not isinstance(self.bounds, tuple)
                        else (self.lb, self.ub)
                    ),
                    rng=self.rng,
                )
            else:
                if self.rng.random() < self.cfg.p_flee:
                    X = move_to_escape(
                        X,
                        w_best,
                        rng=self.rng,
                        bounds=(
                            self.bounds
                            if not isinstance(self.bounds, tuple)
                            else (self.lb, self.ub)
                        ),
                    )
                else:
                    X = blood_squirt(
                        X,
                        w_best,
                        i,
                        self.cfg.iters,
                        bounds=(
                            self.bounds
                            if not isinstance(self.bounds, tuple)
                            else (self.lb, self.ub)
                        ),
                    )

            f = self.fitness(X)
            worst_idx = int(np.argmin(f))
            X = skin_lord(
                X,
                w_best,
                worst_idx,
                rng=self.rng,
                bounds=(
                    self.bounds
                    if not isinstance(self.bounds, tuple)
                    else (self.lb, self.ub)
                ),
                sigma_func=sigma,
            )

            f = self.fitness(X)
            X, _, reset_mask = alpha_msh(
                X,
                f,
                rng=self.rng,
                threshold=self.cfg.alpha_msh_threshold,
                bounds=(self.lb, self.ub),
                sigma_func=sigma,
            )

            a = int(np.argmax(f))
            if f[a] > f_best:
                f_best = float(f[a])
                w_best = X[a].copy()

        return w_best, f_best, X, f

    def minimize(self):
        raise NotImplementedError
