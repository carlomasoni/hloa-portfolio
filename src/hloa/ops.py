"""
Seperated individual operators for the main algorithm for simplicity
Implement each one from the paper

"""
from typing import Generator, PCG64
from hloa.core import HLOA
import numpy as np 

'''
FIND SIGMA ALGO ON PAPER ???
'''
def sigma(rng: np.random.Generator) -> int: 
    return int(rng.random() > 0.5)

'''
CRYPSIS
- frame colour coordinates into CIE space -> rect (a,b)and polar (c,h) (check the gaussian integral project from last year for ez conversion if its still there)
                                    c = sq. (a^2 + b^2), h = arctan(b/a) 
                                    [inv] a = c cos h, b = c sin h
- form colour variable from p,q,r,s 
colorVar = b_p - a_p +- (a_r = b_s ) 

- then map color operations onto search agents X_i (t + 1). from current X_best with random agents + sigma from above 
'''
def crypsis(
    X: np.ndarray,
    X_best: np.ndarray,
    t: int,
    max_iter: int,
    bounds: tuple | None= None,
    rng: Generator | None = None,
    c1: float =1.0,
    c2: float =0.5,
    omega: float=2.0,
    decay_eps: float =0.1,
    sigma = sigma,
) -> np.ndarray:

    if rng is None:
        rng = Generator(PCG64())
    
    n,d = X.shape

    r_idx = np.empty((n,4), dtype=int)
    all_idx = np.arange(n)
    for i in range(n) :
        r_idx[i] = rng.choice(all_idx, size =4, replace = False)

    r1, r2, r3, r4 = r_idx.T

    amplitude_decay = omega * ((1 - (t +1) / max_iter) + (decay_eps / max_iter))

    













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


