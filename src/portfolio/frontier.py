import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt import expected_returns, risk_models


def sharpe_ratio(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> float:
    w = np.asarray(weights, dtype=float)
    ex = float(w @ (mu - rf))
    vol = float(np.sqrt(w @ cov.values @ w))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol


def _download_prices(tickers, start_date, end_date, interval="1mo"):
    if not tickers:
        raise ValueError("No tickers provided.")
    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )
    if df.empty:
        raise ValueError("Download returned empty frame.")

    if isinstance(df.columns, pd.MultiIndex):
        close = df.xs("Adj Close", axis=1, level=1)
        close = close.sort_index(axis=1)
    else:
        close = pd.DataFrame(df["Adj Close"])
        close.columns = [tickers[0]]

    close = close.dropna(how="all", axis=1).dropna(how="all", axis=0)
    return close


EUROSTOXX50 = [
    'ASML.AS','SAP.DE','SIE.DE','ALV.DE','BMW.DE','VOW3.DE','NESN.SW','ROG.SW',
    'OR.PA','SAN.PA','MC.PA','BNP.PA','ENEL.MI','ENI.MI','SAN.MC','BBVA.MC'
]

'''
CHANGE INTERVAL TO 1D, 1WK, 1MO WHATEVER YOU WANT. 
'''
def get_portfolio_data(
    lookback_days: int = 2000,
    interval: str = "1mo",           
    min_years: float = 2.0,         
    tickers: list[str] = None
):
    if tickers is None:
        tickers = EUROSTOXX50

    end_time = datetime.now()
    start_time = end_time - timedelta(days=lookback_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")

    prices = _download_prices(tickers, start_date, end_date, interval=interval)


    interval_to_ann = {"1d": 252, "1wk": 52, "1mo": 12}
    ann = interval_to_ann.get(interval, 12)


    min_obs = int(min_years * ann * 0.8)  
    keep = prices.columns[prices.notna().sum() >= max(min_obs, 12)]
    prices = prices[keep]

    if prices.shape[1] == 0:
        raise ValueError("NO DATA PRESENT - CHECK YAHOO FINANCE (ELSE USE FRED?) :(")

    return prices, ann

def optimize_portfolio_sharpe(
    lookback_days: int = 2000,
    interval: str = "1mo",
    ann_override: int | None = None,
    cap: float = 0.05
):
    from hloa.core import HLOA, HLOA_Config
    from portfolio.constraints import project_capped_simplex

    prices, ann = get_portfolio_data(lookback_days=lookback_days, interval=interval)
    if ann_override is not None:
        ann = ann_override  

    N = prices.shape[1]
    if N * cap < 1.0:
        cap = 1.0 / N - 1e-12

   
    mu = expected_returns.mean_historical_return(prices, frequency=ann)           
    Sigma = risk_models.CovarianceShrinkage(prices, frequency=ann).ledoit_wolf()  

    def portfolio_fitness(weights_batch):
        scores = np.empty(weights_batch.shape[0], dtype=float)
        for i, w in enumerate(weights_batch):
            w_proj = project_capped_simplex(w, total=1.0, cap=cap)
            scores[i] = sharpe_ratio(w_proj, mu, Sigma, rf=0.0)  

    lb = np.zeros(N)
    ub = np.ones(N)
    config = HLOA_Config(pop_size=200, iters=1000, seed=42)

    opt = HLOA(obj=portfolio_fitness, bounds=(lb, ub), config=config)
    w_best, f_best, _, _ = opt.run()

    w_opt = project_capped_simplex(w_best, total=1.0, cap=cap)
    port_ret = float(w_opt @ mu.values)
    port_vol = float(np.sqrt(w_opt @ Sigma.values @ w_opt))
    sr = sharpe_ratio(w_opt, mu, Sigma, rf=0.0)

    return {
        "interval": interval,
        "annualisation_used": ann,
        "cap_used": cap,
        "n_assets": N,
        "optimal_weights": dict(zip(mu.index.tolist(), w_opt)),
        "expected_return": port_ret,
        "volatility": port_vol,
        "sharpe_ratio": sr,
        "optimization_fitness": f_best,
    }









if __name__ == "__main__":
    res = optimize_portfolio_sharpe(
        lookback_days=2000,
        interval="1mo",  
        cap=0.05
    )
    print(res["sharpe_ratio"], res["cap_used"], res["interval"], res["annualisation_used"])


