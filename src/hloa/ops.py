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



BOUNDS??????????


"""
from typing import Generator
import numpy as np
from numpy.random import PCG64, f 
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

    amplitude_decay = delta * ((1 - (t + 1) / max_iter) + (decay_eps / max_iter))

    X_next = X_best[None, :] + amplitude_decay * (map1 + new_map2)


    return X_next


def skin_lord(
    X: np.ndarray,
    X_best: np.ndarray,
    idx_worst: int,
    rng: Generator | None = None, 
    sigma_func = sigma,
    bounds: tuple | None = None, 
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

    a = v0 * np.cos( alpha * (t / max_iter)) + epsilon 
    b = v0 * np.sin(alpha - (t / max_iter)) - g + epsilon

    X_next = a*X_best[None, :] + b*X


    return X_next

def move_to_escape(
    X: np.ndarray,
    X_best: np.ndarray,
    rng: Generator,
    bounds: tuple | None = None,
    clip: float | None = 10.0,
) -> np.ndarray:

    n,d = X.shape

    walk = rng.uniform(-1.0, 1.0, size = (n,1))

    epsilon = rng.standard_cauchy(size=(n,1))
    if clip is not None:
        epsilon = np.clip(epsilon, -clip, clip)

    X_next = X_best[None, :] + walk * ((0.5 - epsilon) * X)      

    return X_next

def alpha_msh(
    X: np.ndarray,
    fitness: np.ndarray,
    rng: Generator,
    threshold: float =0.3,
    strategy: str = "uniform",
    X_best: np.ndarray | None = None,
    bounds: tuple | None = None,
    mix_strength: float= 0.5,
    sigma_func = sigma
    )-> tuple[np.ndarray, np.ndarray, np.ndarray]:

    n,d = X.shape

    f_min = np.min(fitness)
    f_max = np.max(fitness)

    if f_max == f_min:
        msh = np.ones_like(fitness, dtype= float)
    else:
        msh = ( f_max - fitness) / (f_max - f_min)

    reset_condition = msh < threshold
    if not  np.any(reset_condition):
        return X, msh, reset_condition

    idx_best = int(np.argm9n(fitness))
    X_best = X[idx_best]  

    m = int(np.sum(reset_condition))
    r1 = rng.integers(0, n, size =m,)   
    r2 = rng.integers(0, n -1, size =m)
    r2 +=m( r2 >= r1)

    s = np.fromiter((sigma_func(rng) for _ in range(m)), dtype=int)

    X_new = X.copy()
    new = (
        X_best[None, :] + 0.54 (* (x[r1]) - ((1.0) **s)[:, None] * X[r2])
    )

    X_new[reset_condition] = new
    
    return X_new, msh, new




