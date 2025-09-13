"""
Seperated individual operators for the main algorithm for simplicity
Implement each one from the paper
CRYPSIS
- frame colour coordinates into CIE space -> rect (a,b)and polar (c,h) (check the gaussian integral project from last year for ez conversion if its still there)
                                    c = sq. (a^2 + b^2), h = arctan(b/a) 
                                    [inv] a = c cos h, b = c sin h
- form colour variable from p,q,r,s 
colorVar = b_p - a_p +- (a_r = b_s ) 

- then map color operations onto search agents X_i (t + 1). from current X_best with random agents + sigma from above 

"""
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
    omega: float = 2.0,
    decay_eps: float = 0.1,
    sigma_func = sigma,
) -> np.ndarray:

    if rng is None:
        rng = np.random.Generator(np.random.PCG64())
    
    n, d = X.shape

    r_idx = np.empty((n, 4), dtype=int)
    all_idx = np.arange(n)
    for i in range(n):
        replace = n < 4
        r_idx[i] = rng.choice(all_idx, size=4, replace=replace)
    
    #make sure r1 doesnt equal r2 etc 

    r1, r2, r3, r4 = r_idx.T

    # COLOR_VAR from paper. 
    map1 = c1 * (np.sin(X[r1]) - np.cos(X[r2]))
    map2 = c2 * (np.cos(X[r3]) - np.sin(X[r4]))

    sigma_values = np.array([sigma_func(rng) for _ in range(n)])
    new_map2 = ((-1.0) ** sigma_values)[:, None] * map2

    amplitude_decay = omega * ((1 - (t + 1) / max_iter) + (decay_eps / max_iter))

    # CHOOSE THE BEST LIZARD

    X_next = X_best[None, :] + amplitude_decay * (map1 + new_map2)

    X_next = apply_bounds(X_next, bounds)

    return X_next



def skin_lighten():
    return

def skin_darken():
    return

def blood_squirt():
    return

def move_to_escape():
    return

def alpha_msh():
    return


