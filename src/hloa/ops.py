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

SKIN LIGHTENING/DARKENING ]
- pick 4 peers
- sample pallette values, update only worst agent with either eq9 or eq 10
"""
from typing import Generator
import numpy as np
from numpy.random import PCG64 
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
        if n >= 4:
            r_idx[i] = rng.permutation(all_idx)[:4]
        else:
            r_idx[i] = rng.choice(all_idx, size=4, replace=True)

    r1, r2, r3, r4 = r_idx.T


    map1 = c1 * (np.sin(X[r1]) - np.cos(X[r2]))
    map2 = c2 * (np.cos(X[r3]) - np.sin(X[r4]))

    sigma_values = np.array([sigma_func(rng) for _ in range(n)])
    new_map2 = ((-1.0) ** sigma_values)[:, None] * map2

    amplitude_decay = omega * ((1 - (t + 1) / max_iter) + (decay_eps / max_iter))

    X_next = X_best[None, :] + amplitude_decay * (map1 + new_map2)

    X_next = apply_bounds(X_next, bounds)

    return X_next


def skin_lighten(
    X: np.ndarray,
    X_best: np.ndarray,
    idx_worst: int,
    rng: Generator | None = None, 
    sigma_func = sigma,
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


    light_values = [0.0, 0.404661]
    dark_values  = [0.544510, 1.0]

    l1,l2 = rng.choice(light_values, size = 2)
    d1,d2 = rng.choice(dark_values, size = 2)

    s = sigma_func(rng)

    l_agent = (
        X_best + ( 0.5 * l1 *(np.sin(X[r1]) - X[r2]) ) - (  ((-1.0) ** s) * (0.5  * l2 * np.sin(X[r3]) - X[r4]))
    
    )
    
    d_agent = (
        X_best + ( 0.5 * d1 *(np.sin(X[r1]) - X[r2]) ) - (  ((-1.0) ** s) * (0.5  * d2 * np.sin(X[r3]) - X[r4]))
    )

    if rng.random() < 0.5:
        new_agent = l_agent
    else:
        new_agent = d_agent

    X_new  = X.copy()
    X_new[idx_worst] = new_agent
    return X_new



    


    


    






    
    return

def skin_darken():
    return

def blood_squirt():
    return

def move_to_escape():
    return

def alpha_msh():
    return


