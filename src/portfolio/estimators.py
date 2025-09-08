"""
estimators.py

Estimators for expected returns (mu) and covariance (Sigma):

- Sample mean and covariance
- EWMA estimators
- Shrinkage methods (Ledoit–Wolf via scikit-learn)
"""



from __future__ import annotations
import pandas as pd

def mean_returns(returns: pd.DataFrame) -> pd.Series:
    return returns.mean()

def cov_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()
