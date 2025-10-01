import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from pypfopt import expected_returns, risk_models

from hloa.core import HLOA, HLOA_Config
from portfolio.constraints import project_capped_simplex, sharpe_ratio
from portfolio.sx50 import fetch_latest_sx5e_constituents


print("Imports successful")






LOOKBACK_DAYS = 2000         
INTERVAL = "1mo"            
CAP = 0.05                   
SEED = 42
TICKERS = fetch_latest_sx5e_constituents()

SUFFIX_MAP = {
    ".AS": ".nl",  
    ".DE": ".de",  
    ".PA": ".fr",  
    ".SW": ".ch",  
    ".MI": ".it",  
    ".MC": ".es",  
    ".L":  ".uk",  
}

def ensure_feasible_cap(N: int, cap: float) -> float:
    """Return a cap that satisfies N * cap >= 1 (tiny epsilon for safety)."""
    min_cap = 1.0 / N
    if N * cap < 1.0:
        print(f"[info] Requested cap {cap:.4%} infeasible with N={N}. "
              f"Relaxing to {min_cap:.4%}.")
        return min_cap + 1e-12
    return cap


def _stooq_candidates(ticker: str):
    yield ticker
    for ysuf, ssuf in SUFFIX_MAP.items():
        if ticker.endswith(ysuf):
            base = ticker[: -len(ysuf)]
            yield (base + ssuf).lower()
            break
    yield ticker.split(".")[0]
    yield ticker.split(".")[0].lower()


def fetch_one(ticker: str, start_date: str, end_date: str) -> pd.Series:
    for sym in _stooq_candidates(ticker):
        try:
            df = pdr.DataReader(sym, "stooq", start=start_date, end=end_date)
            if df is not None and not df.empty:
                s = df["Close"].copy(); s.name = ticker  
                return s.sort_index()
        except Exception:
            continue
    return pd.Series(dtype=float, name=ticker)

def download_prices(tickers, start_date, end_date) -> pd.DataFrame:
    out = []
    for t in tickers:
        s = fetch_one(t, start_date, end_date)
        if s.size > 0:
            out.append(s)
    if not out:
        raise ValueError("Stooq returned no data for all tickers (check symbols).")
    prices = pd.concat(out, axis=1).sort_index()
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if prices.shape[1] == 0:
        raise ValueError("All fetched series are empty after cleaning.")
    print(f"Fetched {len(out)} series. Columns kept: {list(prices.columns)}")
    return prices



    
def resample_prices(prices: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, int]:
    interval = interval.lower()
    if interval not in {"1d","1wk","1mo"}:
        raise ValueError('interval must be "1d", "1wk", or "1mo"')
    if interval == "1d":
        res = prices.asfreq("B").ffill(); ann = 252
    elif interval == "1wk":
        res = prices.resample("W-FRI").last().ffill(); ann = 52
    else: 
        res = prices.resample("ME").last().ffill(); ann = 12
    min_obs = max(12, int(0.8 * ann * 2))
    keep = res.columns[res.notna().sum() >= min_obs]
    res = res[keep]
    if res.shape[1] == 0:
        raise ValueError("No assets remain after resampling/cleaning.")
    print(f"Resampled to {interval}. {len(res)} columns kept.")
    return res, ann


def get_prices(tickers, lookback_days=LOOKBACK_DAYS, interval=INTERVAL):
    end = datetime.now(); start = end - timedelta(days=lookback_days)
    prices_daily = download_prices(tickers, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    print("Prices downloaded successfully - ready for HLOA")
    return resample_prices(prices_daily, interval=interval)




def optimize_portfolio_with_HLOA(
    tickers=TICKERS,
    lookback_days=LOOKBACK_DAYS,
    interval=INTERVAL,
    cap=CAP,
    seed=SEED,
):
    prices, ann = get_prices(tickers, lookback_days=lookback_days, interval=interval)

    N = prices.shape[1]
    cap = ensure_feasible_cap(N, cap)

    mu = expected_returns.mean_historical_return(prices, frequency=ann)
    Sigma = risk_models.CovarianceShrinkage(prices, frequency=ann).ledoit_wolf()

    def fitness(batch):
        scores = np.empty(batch.shape[0], dtype=float)
        for i, w in enumerate(batch):
            wp = project_capped_simplex(w, total=1.0, cap=cap)
            scores[i] = sharpe_ratio(wp, mu, Sigma, rf=0.0)
        return scores

    lb = np.zeros(N)
    ub = np.ones(N)
    cfg = HLOA_Config(pop_size=200, iters=1000, seed=seed)
    opt = HLOA(obj=fitness, bounds=(lb, ub), config=cfg)

    w_best, _, _, _ = opt.run()
    w_opt = project_capped_simplex(w_best, total=1.0, cap=cap)
    w_opt = np.maximum(w_opt, 0.0)
    w_opt = w_opt / w_opt.sum()

    return dict(sorted(zip(mu.index.tolist(), w_opt), key=lambda kv: kv[1], reverse=True))


def main():
    weights = optimize_portfolio_with_HLOA()
    for ticker, w in weights.items():
        print(f"{ticker},{w:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

