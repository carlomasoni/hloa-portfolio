from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pypfopt

assets = [
    '^FCHI', '^GDAXI', '^FTSE',              
    'FEZ'  
]

eurostoxx50_stocks = [
    'ASML.AS', 'UNA.AS', 'AD.AS', 'KPN.AS', 'INGA.AS', 'PHIA.AS', 'SAP.DE', 'SIE.DE', 'ALV.DE', 'DTE.DE', 'BAYN.DE', 'BMW.DE', 'VOW3.DE','NOVN.SW', 'ROG.SW', 'NESN.SW', 'UHR.SW', 'CSGN.SW', 'GIVN.SW','OR.PA', 'SAN.PA', 'MC.PA', 'AI.PA', 'GLE.PA', 'BNP.PA', 'TTE.PA', 'EL.PA', 'VIE.PA','ABI.BR', 'SOLB.BR','ENEL.MI', 'ISP.MI', 'ENI.MI', 'UCG.MI', 'G.MI','SAN.MC', 'BBVA.MC', 'ITX.MC', 'IBE.MC', 'REP.MC','NOKIA.HE', 'SAMPO.HE','NOVO-B.CO', 'DSV.CO','CRH.L', 'GLEN.L'
]

def get_portfolio_data(time_period_days=30, include_eurostoxx=True):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_period_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")
    all_tickers = assets.copy()
    if include_eurostoxx:
        all_tickers.extend(eurostoxx50_stocks)
    stock_data = yf.download(
        all_tickers, 
        start=start_date, 
        end=end_date, 
        interval="1mo", 
        auto_adjust=False, 
        progress=True,
        group_by='ticker'
    )
    
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
        try:
            if currency == 'EUR':
                bund_ticker = "^TNX"
                risk_free_data = yf.download(bund_ticker, period="1mo", interval="1d")
                if not risk_free_data.empty:
                    latest_yield = risk_free_data['Close'].iloc[-1]
                    risk_free_rate = float(latest_yield) / 100
                else:
                    risk_free_rate = 0.02
            else:
                print(f"Currency {currency} not supported, using default 2%")
                risk_free_rate = 0.02
        except:
            risk_free_rate = 0.02
    else:
        risk_free_rate = 0.025
    return risk_free_rate

def optimize_portfolio_sharpe(time_period_days=30, include_eurostoxx=True, risk_free_rate=None, currency='EUR'):
    from hloa.core import HLOA, HLOA_Config
    from portfolio.objectives import sharpe_ratio
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
    sharpe_ratio_final = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
    
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