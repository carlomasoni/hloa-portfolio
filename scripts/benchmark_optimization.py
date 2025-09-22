import sys
import os
import time
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def create_mock_data():
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=60, freq='ME')
    n_assets = 15
    
    mock_prices = pd.DataFrame(
        np.random.randn(60, n_assets) * 0.02 + 1.0,
        index=dates,
        columns=[f'ASSET_{i:02d}' for i in range(n_assets)]
    )
    
    mock_prices = mock_prices.cumprod() * 100
    
    return mock_prices

def benchmark_frontier_optimization():
    print("Benchmarking Frontier Optimization (HLOA)...")
    
    try:
        from portfolio.frontier import optimize_portfolio_sharpe
        
        with patch('portfolio.frontier.yf.download') as mock_download, \
             patch('portfolio.frontier.get_risk_free_rate') as mock_rf:
            
            mock_data = create_mock_data()
            mock_download.return_value = mock_data
            mock_rf.return_value = 0.025
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean, \
                 patch('portfolio.frontier.HLOA') as mock_hloa:
                
                mock_returns.return_value = mock_data.pct_change().dropna()
                mock_mean.return_value = pd.Series(np.random.randn(15) * 0.1 + 0.08)
                
                mock_optimizer = patch('portfolio.frontier.HLOA').start()
                mock_optimizer.return_value.run.return_value = (
                    np.random.rand(15),
                    0.8,
                    np.random.randn(100, 15),
                    np.random.randn(100)
                )
                
                start_time = time.time()
                results = optimize_portfolio_sharpe(
                    time_period_days=1800,
                    include_eurostoxx=False,
                    risk_free_rate=0.025
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                print(f"Frontier optimization completed in {execution_time:.3f} seconds")
                print(f"  - Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.4f}")
                print(f"  - Expected Return: {results.get('expected_return', 'N/A'):.4f}")
                print(f"  - Volatility: {results.get('volatility', 'N/A'):.4f}")
                print(f"  - Number of Assets: {results.get('n_assets', 'N/A')}")
                
                return {
                    'method': 'HLOA',
                    'execution_time': execution_time,
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'expected_return': results.get('expected_return', 0),
                    'volatility': results.get('volatility', 0),
                    'success': True
                }
                
    except Exception as e:
        print(f"Frontier optimization failed: {e}")
        return {
            'method': 'HLOA',
            'execution_time': 0,
            'sharpe_ratio': 0,
            'expected_return': 0,
            'volatility': 0,
            'success': False,
            'error': str(e)
        }

def benchmark_qubo_optimization():
    print("\nBenchmarking QUBO Optimization...")
    
    try:
        from portfolio.qubo_optimization import optimize_portfolio_qubo
        
        with patch('portfolio.qubo_optimization.yf.download') as mock_download, \
             patch('portfolio.qubo_optimization.get_risk_free_rate_qubo') as mock_rf:
            
            mock_data = create_mock_data()
            mock_download.return_value = mock_data
            mock_rf.return_value = 0.025
            
            with patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean:
                
                mock_returns.return_value = mock_data.pct_change().dropna()
                mock_mean.return_value = pd.Series(np.random.randn(15) * 0.1 + 0.08)
                
                start_time = time.time()
                results = optimize_portfolio_qubo(
                    time_period_days=1800,
                    include_eurostoxx=False,
                    risk_free_rate=0.025,
                    max_assets=8,
                    method='simulated_annealing'
                )
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                print(f"QUBO optimization completed in {execution_time:.3f} seconds")
                print(f"  - Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.4f}")
                print(f"  - Expected Return: {results.get('expected_return', 'N/A'):.4f}")
                print(f"  - Volatility: {results.get('volatility', 'N/A'):.4f}")
                print(f"  - Number of Assets: {results.get('num_assets', 'N/A')}")
                print(f"  - Selected Assets: {results.get('selected_assets', [])}")
                
                return {
                    'method': 'QUBO',
                    'execution_time': execution_time,
                    'sharpe_ratio': results.get('sharpe_ratio', 0),
                    'expected_return': results.get('expected_return', 0),
                    'volatility': results.get('volatility', 0),
                    'num_assets': results.get('num_assets', 0),
                    'success': True
                }
                
    except Exception as e:
        print(f"QUBO optimization failed: {e}")
        return {
            'method': 'QUBO',
            'execution_time': 0,
            'sharpe_ratio': 0,
            'expected_return': 0,
            'volatility': 0,
            'num_assets': 0,
            'success': False,
            'error': str(e)
        }

def compare_results(frontier_results, qubo_results):
    print("\n" + "="*60)
    print("OPTIMIZATION COMPARISON")
    print("="*60)
    
    if not frontier_results['success'] or not qubo_results['success']:
        print("One or both optimizations failed. Cannot compare results.")
        return
    
    print(f"{'Metric':<20} {'HLOA':<15} {'QUBO':<15} {'Winner':<10}")
    print("-" * 60)
    
    metrics = [
        ('Execution Time (s)', 'execution_time', 'lower'),
        ('Sharpe Ratio', 'sharpe_ratio', 'higher'),
        ('Expected Return', 'expected_return', 'higher'),
        ('Volatility', 'volatility', 'lower')
    ]
    
    for metric_name, key, better in metrics:
        frontier_val = frontier_results[key]
        qubo_val = qubo_results[key]
        
        if better == 'higher':
            winner = 'HLOA' if frontier_val > qubo_val else 'QUBO'
        else:
            winner = 'HLOA' if frontier_val < qubo_val else 'QUBO'
        
        print(f"{metric_name:<20} {frontier_val:<15.4f} {qubo_val:<15.4f} {winner:<10}")
    
    print(f"\n{'Number of Assets':<20} {frontier_results.get('n_assets', 'N/A'):<15} {qubo_results.get('num_assets', 'N/A'):<15} {'N/A':<10}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if frontier_results['sharpe_ratio'] > qubo_results['sharpe_ratio']:
        print("HLOA achieved higher Sharpe ratio")
    elif qubo_results['sharpe_ratio'] > frontier_results['sharpe_ratio']:
        print("QUBO achieved higher Sharpe ratio")
    else:
        print("Both methods achieved similar Sharpe ratios")
    
    if frontier_results['execution_time'] < qubo_results['execution_time']:
        print("HLOA was faster")
    elif qubo_results['execution_time'] < frontier_results['execution_time']:
        print("QUBO was faster")
    else:
        print("Both methods had similar execution times")

def main():
    print("="*60)
    print("PORTFOLIO OPTIMIZATION BENCHMARK")
    print("="*60)
    
    print("Running benchmarks with mock data...")
    print("Note: This uses simulated data for consistent testing")
    
    frontier_results = benchmark_frontier_optimization()
    qubo_results = benchmark_qubo_optimization()
    
    compare_results(frontier_results, qubo_results)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    if frontier_results['success'] and qubo_results['success']:
        print("Both optimization methods completed successfully!")
        print("\nKey Differences:")
        print("- HLOA: Continuous weights, evolutionary optimization")
        print("- QUBO: Binary selection, equal weights, combinatorial optimization")
        print("- Both methods can be used depending on your specific requirements")
    else:
        print("Some optimizations failed. Check the error messages above.")

if __name__ == "__main__":
    main()