"""
core.py

Implements the Horned Lizard Optimisation Algorithm (HLOA).
Contains the main optimizer class `HornedLizardOptimizer`.
"""

# libs
import numpy as np
from numpy.random import Generator, PCG64
from typing import Tuple
from hloa.ops import crypsis, sigma, skin_lord, blood_squirt, alpha_msh, move_to_escape
from portfolio.frontier import max_sharpe



class HLOA_Config:
    def __init__(
        self,
        pop_size: int = 150,
        iters: int = 750,
        seed: int | None = 42,
        p_mimic: float = 0.6,
        p_flee: float = 0.2,   
    ):
        self.pop_size = pop_size
        self.iters = iters
        self.seed = seed
        self.p_mimic = p_mimic
        self.p_flee = p_flee



class HLOA: 
    def __init__ (
        self,
        fitness: max_sharpe,
        bounds: Tuple[np.ndarray, np.ndarray],    
        cfg: HLOAConfig = HLOAConfig(),
    ):
        self.fitness = fitness
        self.lb, self.ub = (np.asarray(b, dtype=float) for b in bounds)
        self.N = self.lb.size
        self.cfg = cfg
        self.rng: Generator = Generator(PCG64(cfg.seed))

        def run(self) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
            X = self._init_population()

            f = self.fitness(X)
            best_idx = int(np.argmin(f))
            w_best = X[best_idx].copy()
            f_best = float(f[best_idx])

            for i in range(self.cfg.iters):
                if self.rng.random() < self.cfg.p_mimic:
                    X = crypsis(
                    X, w_best, i, self.cfg.iters,
                    bounds=(self.lb, self.ub),
                    rng=self.rng,
                    )
                else:
                    if self.rng.random() < self.cfg.p_flee:
                        X = move_to_escape(X, w_best, rng=self.rng, bounds=(self.lb, self.ub))
                    else:
                        X = blood_squirt(X, w_best, t, self.cfg.iters, bounds=(self.lb, self.ub))

                f = self.fitness(X)
                worst_idx = int(np.argmax(f))
                X = skin_lord(X, w_best, worst_idx, rng=self.rng, bounds=(self.lb, self.ub), sigma_func=sigma)

                f = self.fitness(X)
                a = int(np.argmin(f))
                if f[a] < f_best:
                    f_best = float(f[a])
                    w_best = X[a].copy()


            return w_best, -f_best, X, f
        





            
            















