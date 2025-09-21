"""
Efficient frontier construction and wrappers:

- Build portfolios across lambda/target-return grids
- Compute risk, return, Sharpe for each solution
- Identify and return max-Sharpe portfolio

-> read up on algos and correct current work cus not fully sure if this is right...
"""

from __future__ import annotations

import numpy as np
import pandas as pd

