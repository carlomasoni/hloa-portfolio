from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta
import pypfopt

assets = [
    '^FCHI', '^GDAXI', '^FTSE',              
    'FEZ'  
]

eurostoxx50_stocks = [
    'ASML.AS', 'SAP.DE', 'NOVN.SW', 'ASML.AS', 'SAP.DE',
    'NOVN.SW', 'ROG.SW', 'NESN.SW', 'OR.PA', 'SAN.PA',
    'MC.PA', 'AI.PA', 'PHIA.AS', 'INGA.AS', 'UNA.AS',
    'AD.AS', 'ABI.BR', 'KPN.AS', 'GLE.PA', 'ENEL.MI'
]

def get_portfolio_data(time_period_days=2000, include_eurostoxx=True):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_period_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")
    
    print(f"Loading data from {start_date} to {end_date}")
    
    all_tickers = assets.copy()
    if include_eurostoxx:
        all_tickers.extend(eurostoxx50_stocks)
    
    print("Downloading all asset data...")
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
    
    stock_df = stock_df.reindex(sorted(stock_df.columns), axis=1)
    corr_matrix_df = stock_df.corr(method="pearson")
    
    print("Calculating expected returns and volatilities...")
    returns_and_vols = {}
    
    for ticker in stock_df.columns:
        if ticker in stock_df.columns:
            prices = stock_df[[ticker]].dropna()
            if len(prices) > 1:
                returns = pypfopt.expected_returns.returns_from_prices(prices)
                exp_return = pypfopt.expected_returns.mean_historical_return(
                    returns, returns_data=True, compounding=True, frequency=12
                )
                vol = returns.std() * np.sqrt(12)
                returns_and_vols[ticker] = {
                    "ret": exp_return.iloc[0], 
                    "vol": vol.iloc[0]
                }
    
    benchmark_tickers = ['^GSPC', '^IXIC', '^DJI', '^FCHI', '^GDAXI', '^FTSE', 'FEZ']
    benchmark_df = stock_df[[ticker for ticker in benchmark_tickers if ticker in stock_df.columns]].copy()
    
    ret_and_vol = pd.DataFrame(returns_and_vols).transpose()
    
    print(f"Successfully loaded data for {len(stock_df.columns)} assets")
    print(f"Data shape: {stock_df.shape}")
    
    return stock_df, benchmark_df, ret_and_vol, corr_matrix_df

def get_risk_free_rate(currency='EUR', source='yfinance'):
    if source == 'yfinance':
        try:
            if currency == 'EUR':
                bund_ticker = "^TNX"
                risk_free_data = yf.download(bund_ticker, period="1mo", interval="1d")
                if not risk_free_data.empty:
                    latest_yield = risk_free_data['Close'].iloc[-1]
                    risk_free_rate = float(latest_yield) / 100.0
                    print(f"Risk-free rate from {bund_ticker}: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
                else:
                    print("Could not fetch risk-free rate data, using default 2%")
                    risk_free_rate = 0.02
                    
            elif currency == 'USD':
                treasury_ticker = "^TNX"
                risk_free_data = yf.download(treasury_ticker, period="1mo", interval="1d")
                if not risk_free_data.empty:
                    latest_yield = risk_free_data['Close'].iloc[-1]
                    risk_free_rate = float(latest_yield) / 100.0
                    print(f"Risk-free rate from {treasury_ticker}: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
                else:
                    print("Could not fetch risk-free rate data, using default 2%")
                    risk_free_rate = 0.02
                    
            else:
                print(f"Currency {currency} not supported, using default 2%")
                risk_free_rate = 0.02
                
        except Exception as e:
            print(f"Error fetching risk-free rate: {e}")
            print("Using default risk-free rate of 2%")
            risk_free_rate = 0.02
            
    elif source == 'manual':
        risk_free_rates = {
            'EUR': 0.025,
            'USD': 0.045,
            'GBP': 0.050,
        }
        risk_free_rate = risk_free_rates.get(currency, 0.02)
        print(f"Manual risk-free rate for {currency}: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
    
    return risk_free_rate

def optimize_portfolio_sharpe(time_period_days=2000, include_eurostoxx=True, risk_free_rate=None, currency='EUR'):
    from hloa.core import HLOA, HLOA_Config
    from portfolio.objectives import sharpe_ratio
    from portfolio.constraints import project_capped_simplex
    
    if risk_free_rate is None:
        print("Calculating risk-free rate...")
        risk_free_rate = get_risk_free_rate(currency=currency, source='yfinance')
    else:
        print(f"Using provided risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
    
    stock_df, benchmark_df, ret_and_vol, corr_matrix = get_portfolio_data(
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
    config = HLOA_Config(pop_size=100, iters=500, seed=42)
    
    opt = HLOA(obj=portfolio_fitness, bounds=(lb, ub), config=config)
    
    print("Running HLOA optimization...")
    w_best, f_best, X_final, f_final = opt.run()
    
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
    
    print(f"Optimization complete!")
    print(f"Optimal Sharpe Ratio: {sharpe_ratio_final:.4f}")
    print(f"Expected Return: {portfolio_return:.4f}")
    print(f"Volatility: {portfolio_vol:.4f}")
    
    return results

if __name__ == "__main__":
    print("="*50)
    print("RISK-FREE RATE CALCULATION")
    print("="*50)
    
    eur_rf = get_risk_free_rate(currency='EUR', source='yfinance')
    print(f"EUR Risk-free rate: {eur_rf:.4f} ({eur_rf*100:.2f}%)")
    
    usd_rf = get_risk_free_rate(currency='USD', source='yfinance')
    print(f"USD Risk-free rate: {usd_rf:.4f} ({usd_rf*100:.2f}%)")
    
    manual_rf = get_risk_free_rate(currency='EUR', source='manual')
    print(f"Manual EUR rate: {manual_rf:.4f} ({manual_rf*100:.2f}%)")
    
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    results = optimize_portfolio_sharpe(
        time_period_days=2000,
        include_eurostoxx=True,
        risk_free_rate=None,
        currency='EUR'
    )
    
    print("\n" + "="*50)
    print("OPTIMIZATION WITH MANUAL RISK-FREE RATE")
    print("="*50)
    
    results_manual = optimize_portfolio_sharpe(
        time_period_days=2000,
        include_eurostoxx=True,
        risk_free_rate=0.025,
        currency='EUR'
    )
    
    print("\n" + "="*50)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*50)
    print(f"Number of assets: {results['n_assets']}")
    print(f"Optimal Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Expected Annual Return: {results['expected_return']:.4f}")
    print(f"Expected Annual Volatility: {results['volatility']:.4f}")
    
    print("\nOptimal Portfolio Weights:")
    print("-" * 30)
    for asset, weight in results['optimal_weights'].items():
        if weight > 0.001:
            print(f"{asset:15s}: {weight:.3f} ({weight*100:.1f}%)")