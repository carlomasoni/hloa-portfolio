import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from pypfopt import expected_returns, risk_models

def sharpe_ratio(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> float:
    w = np.asarray(weights, dtype=float)
    ex = float(np.dot(w, (mu - rf)))
    vol = float(np.sqrt(np.dot(w, cov.values @ w)))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol

def get_risk_free_rate(currency='EUR', source='yfinance'):
    rf_default = 0.02
    if source != 'yfinance' or yf is None:
        return rf_default
    try:
        if currency.upper() == 'EUR':
            ticker = "DE10Y-DE"

        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        if data.empty:
            return rf_default
        last = float(data['Close'].iloc[-1])
        rf = last / 100.0
        if not np.isfinite(rf) or rf < -0.05 or rf > 0.15:
            return rf_default
        return rf
    except Exception:
        return rf_default

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
        close = df.xs('Adj Close', axis=1, level=1)
        close = close.sort_index(axis=1)
    else:
        close = pd.DataFrame(df['Adj Close'])
        close.columns = [tickers[0]]

    close = close.dropna(how='all', axis=1).dropna(how='all', axis=0)
    return close

def get_portfolio_data(time_period_days=30, include_eurostoxx=True):
    benchmarks = ['^FCHI', '^GDAXI', '^FTSE', 'FEZ']
    eurostoxx50_stocks = [
        'ASML.AS', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'VOW3.DE', 'NESN.SW', 'ROG.SW',
        'OR.PA', 'SAN.PA', 'MC.PA', 'BNP.PA', 'ENEL.MI', 'ENI.MI', 'SAN.MC', 'BBVA.MC'
    ]

    investable = eurostoxx50_stocks if include_eurostoxx else []

    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_period_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")

    prices_investable = _download_prices(investable, start_date, end_date, interval="1mo")
    min_obs = 24 
    sufficient = prices_investable.columns[prices_investable.notna().sum() >= min_obs]
    prices_investable = prices_investable[sufficient]

    prices_bench = pd.DataFrame()
    try:
        prices_bench = _download_prices(benchmarks, start_date, end_date, interval="1mo")
    except Exception:
        pass


    return prices_investable, prices_bench

def optimize_portfolio_sharpe(time_period_days=30, include_eurostoxx=True, risk_free_rate=None, currency='EUR'):
    from hloa.core import HLOA, HLOA_Config
    from portfolio.constraints import project_capped_simplex


    prices, _bench = get_portfolio_data(time_period_days, include_eurostoxx)

    if prices.shape[1] == 0:
        raise ValueError("ERROR - NO DATA PRESENT - CHECK YAHOO FINANCE (ELSE USE FRED?)")

    cap = 0.05
    N = prices.shape[1]
    if N * cap < 1.0:
        cap = 1.0 / N - 1e-12

    mu = expected_returns.mean_historical_return(prices, frequency=12)     
    Sigma = risk_models.CovarianceShrinkage(prices, frequency=12).ledoit_wolf()


    rf = risk_free_rate if risk_free_rate is not None else get_risk_free_rate(currency=currency, source='yfinance')


    def portfolio_fitness(weights_batch):
        scores = np.empty(weights_batch.shape[0], dtype=float)
        for i, w in enumerate(weights_batch):
            w_proj = project_capped_simplex(w, total=1.0, cap=cap)
            scores[i] = sharpe_ratio(w_proj, mu, Sigma, rf=rf)
        return scores


    lb = np.zeros(N)
    ub = np.ones(N)
    config = HLOA_Config(pop_size=200, iters=1000, seed=42, project_on_init=True) 

    opt = HLOA(obj=portfolio_fitness, bounds=(lb, ub), config=config)
    w_best, f_best, _, _ = opt.run()

    w_opt = project_capped_simplex(w_best, total=1.0, cap=cap)
    port_ret = float(np.dot(w_opt, mu.values))
    port_vol = float(np.sqrt(w_opt @ Sigma.values @ w_opt))
    sr = sharpe_ratio(w_opt, mu, Sigma, rf=rf)

    return {
        'optimal_weights': dict(zip(mu.index.tolist(), w_opt)),
        'sharpe_ratio': sr,
        'expected_return': port_ret,
        'volatility': port_vol,
        'risk_free_rate': rf,
        'n_assets': N,
        'cap_used': cap,
        'asset_names': mu.index.tolist(),
        'optimization_fitness': f_best,
    }
