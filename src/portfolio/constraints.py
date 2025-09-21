"""

- Simplex projection (long-only, sum of weights = 1)
- Box constraints (w_min ≤ w ≤ w_max) with renormalization
- Composable projectors for long-only, long-short, leverage caps





    Parameters:
    -----------
    X_next : np.ndarray
        The next generation of search agents (n_agents, n_dimensions)
    bounds : Union[Tuple[np.ndarray, np.ndarray], str, None]
        Bounds specification:
        - None: No bounds applied
        - "simplex": Simplex constraint (sum to 1, non-negative) - for portfolio weights
        - "simplex_long_only": Long-only portfolio (non-negative weights, sum to 1)
        - "simplex_long_short": Long-short portfolio (weights can be negative, sum to 1)
        - (lower_bounds, upper_bounds): Box constraints with arrays of same shape

    Returns:
    --------
    np.ndarray
        Constrained search agents
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np



