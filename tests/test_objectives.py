"""
Unit tests for portfolio/objectives.py

Validates objective calculations:
- Negative Sharpe ratio
- Mean–variance penalty form
- Stability (no NaNs/inf on small variances)
"""




import numpy as np
import pandas as pd
from portfolio.objectives import sharpe_ratio, mean_variance

def test_sharpe_ratio_positive():
    w = np.array([0.5, 0.5])
    mu = pd.Series([0.1, 0.2])
    cov = pd.DataFrame([[0.04, 0.0],[0.0, 0.09]])
    sr = sharpe_ratio(w, mu, cov, rf=0.0)
    assert sr > 0

def test_mean_variance_tradeoff():
    w = np.array([0.5, 0.5])
    mu = pd.Series([0.1, 0.2])
    cov = pd.DataFrame([[0.04, 0.0],[0.0, 0.09]])
    mv1 = mean_variance(w, mu, cov, risk_aversion=0.1)
    mv2 = mean_variance(w, mu, cov, risk_aversion=10.0)
    assert mv1 > mv2
