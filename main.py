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


LOOKBACK_DAYS = 2000         # roughly ~8 years if monthly
INTERVAL = "1mo"             # "1d" | "1wk" | "1mo"
CAP = 0.05                   # 5% per name
SEED = 42
TICKERS = [
    'ASML.AS','SAP.DE','SIE.DE','ALV.DE','BMW.DE','VOW3.DE','NESN.SW','ROG.SW',
    'OR.PA','SAN.PA','MC.PA','BNP.PA','ENEL.MI','ENI.MI','SAN.MC','BBVA.MC'
]



def fetch_one(ticker: str, start_date: str, end_date: str) -> pd.Series:
    try:
        df = pdr.DataReader(ticker, "stooq", start=start_date, end=end_date)
        if df is None or df.empty:
            return pd.Series(dtype=float, name=ticker)
        s = df["Close"].copy()
        s.name = ticker
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float, name=ticker)

def download_prices(tickers, start_date, end_date) -> pd.DataFrame:
    series = []
    for t in tickers:
        s = fetch_one(t, start_date, end_date)
        if s.size > 0:
            series.append(s)
    if not series:
        raise ValueError(f"NO DATA PULLED FROM STOOQ FOR ANY TICKER. -> CHECK [ fetch_one_stooq ] FUNCTION.")
    prices = pd.concat(series, axis=1).sort_index()
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")
    return prices



    
def resample_prices(prices: pd.DataFrame, interval: str) -> tuple[pd.DataFrame, int]:
    interval = interval.lower()
    if interval not in {"1d", "1wk", "1mo"}:
        raise ValueError('WRONG INTERVAL - MUST BE ONE OF {"1d","1wk","1mo"}')

    if interval == "1d":
        resampled = prices.asfreq("B").ffill()
        ann = 252
    elif interval == "1wk":
        resampled = prices.resample("W-FRI").last().ffill()
        ann = 52
    else:
        resampled = prices.resample("M").last().ffill()
        ann = 12

    min_obs = max(12, int(0.8 * ann * 2))
    prices = prices.dropna(axis=1, how="all").drop(
        columns=prices.columns[prices.count() < min_obs],
        inplace=True
    )

    return prices, ann


def get_prices_stooq(tickers, lookback_days=LOOKBACK_DAYS, interval=INTERVAL) -> tuple[pd.DataFrame, int]:
    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")

    daily = download_prices(tickers, start_date, end_date)
    return resample_prices(daily, interval=interval)




def optimize_portfolio_with_HLOA(
    tickers=TICKERS,
    lookback_days=LOOKBACK_DAYS,
    interval=INTERVAL,
    cap=CAP,
    seed=SEED,
):
    prices, ann = get_prices_stooq(tickers, lookback_days=lookback_days, interval=interval)

    N = prices.shape[1]
    if N * cap < 1.0:
        cap = 1.0 / N - 1e-12  

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

    return dict(sorted(zip(mu.index.tolist(), w_opt), key=lambda kv: kv[1], reverse=True))


def main():
    weights = optimize_portfolio_with_HLOA()
    for ticker, w in weights.items():
        print(f"{ticker},{w:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

    