"""
data.py

Functions for handling market data:

- Load price data from CSVs or DataFrames
- Convert to log or simple returns
- Frequency conversion (daily, weekly, monthly)
- Annualisation of mean and covariance
"""

from __future__ import annotations

import pandas as pd


def load_returns(path: str, freq: str = "D") -> pd.DataFrame:
    """
    Load a CSV with a DatetimeIndex in column 0 and asset columns after.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if freq:
        df = df.asfreq(freq)
    return df.dropna(how="any")
