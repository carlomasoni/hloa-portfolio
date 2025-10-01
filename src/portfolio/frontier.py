from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

import yfinance as yf
import pypfopt


def sharpe_ratio(
    weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0
) -> float:
    w = np.asarray(weights, dtype=float)
    ex = float(np.dot(w, (mu - rf)))
    vol = float(np.sqrt(np.dot(w, cov.values @ w)))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol

assets = [
    '^FCHI', '^GDAXI', '^FTSE',              
    'FEZ']  
eurostoxx50_stocks_full = [
    'ASML.AS', 'UNA.AS', 'AD.AS', 'KPN.AS', 'INGA.AS', 'PHIA.AS', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAYN.DE', 'BMW.DE', 'VOW3.DE','NOVN.SW', 'ROG.SW', 'NESN.SW', 'UHR.SW', 'CSGN.SW', 'GIVN.SW','OR.PA', 'SAN.PA', 'MC.PA', 'AI.PA', 'GLE.PA', 'BNP.PA', 'TTE.PA', 'EL.PA', 'VIE.PA','ABI.BR', 'SOLB.BR','ENEL.MI', 'ISP.MI', 'ENI.MI', 'UCG.MI', 'G.MI','SAN.MC', 'BBVA.MC', 'ITX.MC', 'IBE.MC', 'REP.MC','NOKIA.HE', 'SAMPO.HE','NOVO-B.CO', 'DSV.CO','CRH.L', 'GLEN.L'
]
eurostoxx50_stocks = [
    'ASML.AS', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'BMW.DE', 'VOW3.DE', 'NESN.SW', 'ROG.SW', 
    'OR.PA', 'SAN.PA', 'MC.PA', 'BNP.PA', 'ENEL.MI', 'ENI.MI', 'SAN.MC', 'BBVA.MC'
]

def get_portfolio_data(time_period_days=30, include_eurostoxx=True):
    if yf is None or pypfopt is None:
        raise ImportError("yfinance and pypfopt are required for portfolio data. Install with: pip install yfinance pypfopt")
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_period_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")
    all_tickers = assets.copy()
    if include_eurostoxx:
        all_tickers.extend(eurostoxx50_stocks)
    batch_size = 10
    all_data = {}

    print(f"Downloading data for {len(all_tickers)} tickers in batches of {batch_size}...")

    for i in range(0, len(all_tickers), batch_size):
        batch_tickers = all_tickers[i:i+batch_size]
        print(f"Downloading batch {i//batch_size + 1}/{(len(all_tickers) + batch_size - 1)//batch_size}: {batch_tickers}")

        try:
            batch_data = yf.download(
                batch_tickers, 
                start=start_date, 
                end=end_date, 
                interval="1mo", 
                auto_adjust=False, 
                progress=False,
                group_by='ticker'
            )
            
            if not batch_data.empty:
                all_data.update({ticker: batch_data for ticker in batch_tickers})

            import time
            time.sleep(1)

        except Exception as e:
            print(f"Warning: Failed to download batch {batch_tickers}: {e}")
            continue

    if not all_data:
        raise ValueError("No data could be downloaded. Please check your internet connection and try again.")
    stock_data = pd.concat(all_data.values(), axis=1, keys=all_data.keys())

    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = ['_'.join(col).strip() for col in stock_data.columns.values]
        adj_close_cols = [col for col in stock_data.columns if 'Adj Close' in col]
        stock_df = stock_data[adj_close_cols].copy()
        stock_df.columns = [col.replace('_Adj Close', '') for col in stock_df.columns]
    else:
        stock_df = pd.DataFrame(stock_data['Adj Close']).rename(columns={'Adj Close': all_tickers[0]})

    returns_and_vols = {}

    valid_tickers = []
    for ticker in stock_df.columns:
        prices = stock_df[[ticker]].dropna()
        if len(prices) > 10:
            try:
                returns = pypfopt.expected_returns.returns_from_prices(prices)
                exp_return = pypfopt.expected_returns.mean_historical_return(
                    returns, returns_data=True, compounding=True, frequency=12
                )
                vol = returns.std() * np.sqrt(12)
                if (not np.isnan(exp_return.iloc[0]) and 
                    not np.isnan(vol.iloc[0]) and 
                    not np.isinf(exp_return.iloc[0]) and 
                    not np.isinf(vol.iloc[0])):

                    returns_and_vols[ticker] = {
                        "ret": exp_return.iloc[0], 
                        "vol": vol.iloc[0]
                    }
                    valid_tickers.append(ticker)
            except:
                continue
    stock_df = stock_df[valid_tickers]
    stock_df = stock_df.reindex(sorted(stock_df.columns), axis=1)
    corr_matrix_df = stock_df.corr(method="pearson")

    benchmark_tickers = ['^GSPC', '^IXIC', '^DJI', '^FCHI', '^GDAXI', '^FTSE', 'FEZ']
    benchmark_df = stock_df[[ticker for ticker in benchmark_tickers if ticker in stock_df.columns]].copy()

    ret_and_vol = pd.DataFrame(returns_and_vols).transpose()
    return stock_df, benchmark_df, ret_and_vol, corr_matrix_df

def get_risk_free_rate(currency='EUR', source='yfinance'):
    if source == 'yfinance':
        if yf is None:
            raise ImportError("yfinance is required for risk-free rate data. Install with: pip install yfinance")
        try:
            if currency == 'EUR':
                bund_ticker = "^TNX"
                risk_free_data = yf.download(bund_ticker, period="1mo", interval="1d", progress=False)
                if not risk_free_data.empty:
                    latest_yield = risk_free_data['Close'].iloc[-1]
                    risk_free_rate = float(latest_yield) / 100
                    print(f"Using risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
                else:
                    risk_free_rate = 0.02
                    print("Using default risk-free rate: 2.00%")
            else:
                print(f"Currency {currency} not supported, using default 2%")
                risk_free_rate = 0.02
        except Exception as e:
            print(f"Warning: Could not fetch risk-free rate ({e}), using default 2%")
            risk_free_rate = 0.02
    else:
        risk_free_rate = 0.025
    return risk_free_rate

def optimize_portfolio_sharpe(time_period_days=30, include_eurostoxx=True, risk_free_rate=None, currency='EUR'):
    from hloa.core import HLOA, HLOA_Config
    from portfolio.constraints import project_capped_simplex
    
    if risk_free_rate is None:
        risk_free_rate = get_risk_free_rate(currency=currency, source='yfinance')

    stock_df, _, ret_and_vol, _ = get_portfolio_data(
        time_period_days, include_eurostoxx
    )
    
    mu = ret_and_vol['ret']
    cov_matrix = stock_df.pct_change().dropna().cov() * 12

    n_assets = len(mu)

    def portfolio_fitness(weights):
        fitness_scores = np.zeros(weights.shape[0])
        for i, w in enumerate(weights):
            w_proj = project_capped_simplex(w, total=1.0, cap=0.05)
            fitness_scores[i] = sharpe_ratio(w_proj, mu, cov_matrix, rf=risk_free_rate)
        return fitness_scores

    lb = np.zeros(n_assets)
    ub = np.ones(n_assets)
    config = HLOA_Config(pop_size=200, iters=1000, seed=42)

    opt = HLOA(obj=portfolio_fitness, bounds=(lb, ub), config=config)

    w_best, f_best, _, _ = opt.run()

    w_optimal = project_capped_simplex(w_best, total=1.0, cap=0.05)

    portfolio_return = float(np.dot(w_optimal, mu))
    portfolio_vol = float(np.sqrt(np.dot(w_optimal, cov_matrix.values @ w_optimal)))
    sharpe_ratio_final = sharpe_ratio(w_optimal, mu, cov_matrix, rf=risk_free_rate)

    results = {
        'optimal_weights': dict(zip(mu.index, w_optimal)),
        'sharpe_ratio': sharpe_ratio_final,
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'risk_free_rate': risk_free_rate,
        'n_assets': n_assets,
        'asset_names': list(mu.index),
        'optimization_fitness': f_best
    }

    return results

if __name__ == "__main__":
    results = optimize_portfolio_sharpe(
        time_period_days=2000,
        include_eurostoxx=True,
        risk_free_rate=None,
        currency='EUR'
    )