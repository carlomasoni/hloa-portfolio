from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pypfopt
from typing import Dict, List, Tuple
import itertools

def get_portfolio_data_qubo(time_period_days=2000, include_eurostoxx=True):
    end_time = datetime.now()
    start_time = end_time - timedelta(days=time_period_days)
    end_date = end_time.strftime("%Y-%m-%d")
    start_date = start_time.strftime("%Y-%m-%d")
    
    print(f"Loading data from {start_date} to {end_date}")
    
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
    
    ret_and_vol = pd.DataFrame(returns_and_vols).transpose()
    
    print(f"Successfully loaded data for {len(stock_df.columns)} assets")
    print(f"Data shape: {stock_df.shape}")
    
    return stock_df, ret_and_vol

def get_risk_free_rate_qubo(currency='EUR', source='yfinance'):
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

def construct_qubo_matrix(mu: pd.Series, cov_matrix: pd.DataFrame, risk_free_rate: float, 
                         max_assets: int = 10, penalty_weight: float = 1.0) -> np.ndarray:
    n_assets = len(mu)
    
    Q = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i == j:
                Q[i, j] = -(mu.iloc[i] - risk_free_rate) + penalty_weight * (1 - 2 * max_assets)
            else:
                Q[i, j] = penalty_weight * (1 + cov_matrix.iloc[i, j])
    
    return Q

def solve_qubo_brute_force(Q: np.ndarray, max_assets: int = 10) -> Tuple[np.ndarray, float]:
    n_assets = Q.shape[0]
    best_solution = np.zeros(n_assets)
    best_value = float('inf')
    
    print(f"Solving QUBO with brute force for {n_assets} assets...")
    
    for num_selected in range(1, min(max_assets + 1, n_assets + 1)):
        for combination in itertools.combinations(range(n_assets), num_selected):
            x = np.zeros(n_assets)
            x[list(combination)] = 1
            
            objective = x.T @ Q @ x
            
            if objective < best_value:
                best_value = objective
                best_solution = x.copy()
    
    return best_solution, best_value

def solve_qubo_simulated_annealing(Q: np.ndarray, max_assets: int = 10, 
                                  n_iterations: int = 10000, 
                                  initial_temp: float = 100.0) -> Tuple[np.ndarray, float]:
    n_assets = Q.shape[0]
    
    x = np.zeros(n_assets)
    num_selected = np.random.randint(1, min(max_assets + 1, n_assets + 1))
    selected_indices = np.random.choice(n_assets, size=num_selected, replace=False)
    x[selected_indices] = 1
    
    current_obj = x.T @ Q @ x
    best_solution = x.copy()
    best_obj = current_obj
    
    temperature = initial_temp
    
    print(f"Solving QUBO with simulated annealing for {n_assets} assets...")
    
    for iteration in range(n_iterations):
        x_new = x.copy()
        
        if np.random.random() < 0.5:
            if np.sum(x) < max_assets and np.sum(x) < n_assets:
                zero_indices = np.where(x == 0)[0]
                if len(zero_indices) > 0:
                    idx = np.random.choice(zero_indices)
                    x_new[idx] = 1
        else:
            one_indices = np.where(x == 1)[0]
            if len(one_indices) > 0:
                idx = np.random.choice(one_indices)
                x_new[idx] = 0
        
        new_obj = x_new.T @ Q @ x_new
        
        if new_obj < current_obj or np.random.random() < np.exp(-(new_obj - current_obj) / temperature):
            x = x_new
            current_obj = new_obj
            
            if current_obj < best_obj:
                best_solution = x.copy()
                best_obj = current_obj
        
        temperature *= 0.995
    
    return best_solution, best_obj

def calculate_portfolio_metrics(weights: np.ndarray, mu: pd.Series, 
                              cov_matrix: pd.DataFrame, risk_free_rate: float) -> Dict:
    selected_assets = weights.astype(bool)
    
    if np.sum(selected_assets) == 0:
        return {
            'sharpe_ratio': 0.0,
            'expected_return': 0.0,
            'volatility': 0.0,
            'num_assets': 0
        }
    
    selected_mu = mu[selected_assets]
    selected_cov = cov_matrix.loc[selected_assets, selected_assets]
    
    equal_weights = np.ones(len(selected_mu)) / len(selected_mu)
    
    portfolio_return = float(np.dot(equal_weights, selected_mu))
    portfolio_vol = float(np.sqrt(np.dot(equal_weights, selected_cov.values @ equal_weights)))
    
    if portfolio_vol > 0:
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    else:
        sharpe_ratio = 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'num_assets': int(np.sum(selected_assets)),
        'selected_assets': mu.index[selected_assets].tolist()
    }

def optimize_portfolio_qubo(time_period_days=2000, include_eurostoxx=True, 
                           risk_free_rate=None, currency='EUR', 
                           max_assets=10, method='simulated_annealing',
                           penalty_weight=1.0) -> Dict:
    if risk_free_rate is None:
        print("Calculating risk-free rate...")
        risk_free_rate = get_risk_free_rate_qubo(currency=currency, source='yfinance')
    else:
        print(f"Using provided risk-free rate: {risk_free_rate:.4f} ({risk_free_rate*100:.2f}%)")
    
    stock_df, ret_and_vol = get_portfolio_data_qubo(time_period_days, include_eurostoxx)
    
    mu = ret_and_vol['ret']
    cov_matrix = stock_df.pct_change().dropna().cov() * 12
    
    print(f"Constructing QUBO matrix for {len(mu)} assets...")
    Q = construct_qubo_matrix(mu, cov_matrix, risk_free_rate, max_assets, penalty_weight)
    
    if method == 'brute_force':
        if len(mu) > 15:
            print("Warning: Brute force is slow for large problems. Consider using simulated annealing.")
        solution, objective = solve_qubo_brute_force(Q, max_assets)
    elif method == 'simulated_annealing':
        solution, objective = solve_qubo_simulated_annealing(Q, max_assets)
    else:
        raise ValueError("Method must be 'brute_force' or 'simulated_annealing'")
    
    metrics = calculate_portfolio_metrics(solution, mu, cov_matrix, risk_free_rate)
    
    results = {
        'binary_weights': solution,
        'qubo_objective': objective,
        'sharpe_ratio': metrics['sharpe_ratio'],
        'expected_return': metrics['expected_return'],
        'volatility': metrics['volatility'],
        'num_assets': metrics['num_assets'],
        'selected_assets': metrics['selected_assets'],
        'risk_free_rate': risk_free_rate,
        'method': method,
        'max_assets': max_assets
    }
    
    print(f"QUBO optimization complete!")
    print(f"Method: {method}")
    print(f"Selected {metrics['num_assets']} assets")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Expected Return: {metrics['expected_return']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    
    return results

if __name__ == "__main__":
    print("="*60)
    print("QUBO PORTFOLIO OPTIMIZATION")
    print("="*60)
    
    print("\n1. Simulated Annealing Method:")
    print("-" * 40)
    results_sa = optimize_portfolio_qubo(
        time_period_days=2000,
        include_eurostoxx=True,
        risk_free_rate=None,
        currency='EUR',
        max_assets=8,
        method='simulated_annealing',
        penalty_weight=1.0
    )
    
    print(f"\nSelected Assets: {results_sa['selected_assets']}")
    
    print("\n2. Brute Force Method (small problem):")
    print("-" * 40)
    results_bf = optimize_portfolio_qubo(
        time_period_days=2000,
        include_eurostoxx=False,
        risk_free_rate=0.025,
        currency='EUR',
        max_assets=5,
        method='brute_force',
        penalty_weight=1.0
    )
    
    print(f"\nSelected Assets: {results_bf['selected_assets']}")
    
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Simulated Annealing - Sharpe: {results_sa['sharpe_ratio']:.4f}, Assets: {results_sa['num_assets']}")
    print(f"Brute Force - Sharpe: {results_bf['sharpe_ratio']:.4f}, Assets: {results_bf['num_assets']}")