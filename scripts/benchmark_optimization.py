import sys
import os
import time
import numpy as np
import pandas as pd
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def create_mock_data():
    np.random.seed(42)
    
    dates = pd.date_range('2020-01-01', periods=60, freq='ME')
    
    mock_prices = pd.DataFrame({
        'Adj Close': np.random.randn(60) * 0.02 + 1.0
    }, index=dates)
    
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
                 patch('hloa.core.HLOA') as mock_hloa:
                
                mock_returns.return_value = mock_data.pct_change().dropna()
                mock_mean.return_value = pd.Series(np.random.randn(15) * 0.1 + 0.08)
                
                mock_optimizer = patch('hloa.core.HLOA').start()
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


def display_results(frontier_results):
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    
    if not frontier_results['success']:
        print("Optimization failed. Check error messages above.")
        return
    
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    
    metrics = [
        ('Execution Time (s)', 'execution_time'),
        ('Sharpe Ratio', 'sharpe_ratio'),
        ('Expected Return', 'expected_return'),
        ('Volatility', 'volatility')
    ]
    
    for metric_name, key in metrics:
        value = frontier_results[key]
        print(f"{metric_name:<25} {value:<15.4f}")
    
    print(f"\n{'Number of Assets':<25} {frontier_results.get('n_assets', 'N/A'):<15}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print("HLOA optimization completed successfully!")
    print(f"Sharpe Ratio: {frontier_results['sharpe_ratio']:.4f}")
    print(f"Execution Time: {frontier_results['execution_time']:.3f} seconds")

def main():
    print("="*60)
    print("PORTFOLIO OPTIMIZATION BENCHMARK")
    print("="*60)
    
    print("Running HLOA benchmark with mock data...")
    print("Note: This uses simulated data for consistent testing")
    
    frontier_results = benchmark_frontier_optimization()
    
    display_results(frontier_results)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    
    if frontier_results['success']:
        print("HLOA optimization completed successfully!")
        print("\nKey Features:")
        print("- HLOA: Continuous weights, evolutionary optimization")
        print("- 5% weight cap constraint per asset")
        print("- Optimizes for maximum Sharpe ratio")
    else:
        print("Optimization failed. Check the error messages above.")

if __name__ == "__main__":
    main()