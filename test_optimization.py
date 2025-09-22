import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_frontier_optimization():
    print("Testing Frontier Optimization...")
    
    try:
        from portfolio.frontier import get_portfolio_data, get_risk_free_rate, optimize_portfolio_sharpe
        
        with patch('portfolio.frontier.yf.download') as mock_download:
            mock_data = pd.DataFrame({
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='ME'))
            mock_download.return_value = mock_data
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean:
                
                mock_returns.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
                mock_mean.return_value = pd.Series([0.15], index=['Adj Close'])
                
                stock_df, benchmark_df, ret_and_vol, corr_matrix = get_portfolio_data(
                    time_period_days=100, 
                    include_eurostoxx=False
                )
                
                assert isinstance(stock_df, pd.DataFrame), "Stock DataFrame should be returned"
                assert isinstance(benchmark_df, pd.DataFrame), "Benchmark DataFrame should be returned"
                assert isinstance(ret_and_vol, pd.DataFrame), "Returns DataFrame should be returned"
                assert isinstance(corr_matrix, pd.DataFrame), "Correlation matrix should be returned"
                
                print("get_portfolio_data test passed")
        
        rate = get_risk_free_rate(currency='EUR', source='manual')
        assert isinstance(rate, float), "Risk-free rate should be float"
        assert rate == 0.025, "EUR manual rate should be 0.025"
        print("get_risk_free_rate test passed")
        
        print("All Frontier Optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"Frontier Optimization test failed: {e}")
        return False

def test_qubo_optimization():
    print("\nTesting QUBO Optimization...")
    
    try:
        from portfolio.qubo_optimization import (
            get_portfolio_data_qubo, 
            get_risk_free_rate_qubo,
            construct_qubo_matrix,
            solve_qubo_brute_force,
            solve_qubo_simulated_annealing,
            calculate_portfolio_metrics,
            optimize_portfolio_qubo
        )
        
        with patch('portfolio.qubo_optimization.yf.download') as mock_download:
            mock_data = pd.DataFrame({
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='ME'))
            mock_download.return_value = mock_data
            
            with patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean:
                
                mock_returns.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
                mock_mean.return_value = pd.Series([0.15], index=['Adj Close'])
                
                stock_df, ret_and_vol = get_portfolio_data_qubo(
                    time_period_days=100, 
                    include_eurostoxx=False
                )
                
                assert isinstance(stock_df, pd.DataFrame), "Stock DataFrame should be returned"
                assert isinstance(ret_and_vol, pd.DataFrame), "Returns DataFrame should be returned"
                
                print("get_portfolio_data_qubo test passed")
        
        rate = get_risk_free_rate_qubo(currency='USD', source='manual')
        assert isinstance(rate, float), "Risk-free rate should be float"
        assert rate == 0.045, "USD manual rate should be 0.045"
        print("get_risk_free_rate_qubo test passed")
        
        mu = pd.Series([0.1, 0.12, 0.08], index=['A', 'B', 'C'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.01, 0.02],
            'B': [0.01, 0.09, 0.01],
            'C': [0.02, 0.01, 0.16]
        }, index=['A', 'B', 'C'])
        
        Q = construct_qubo_matrix(mu, cov_matrix, 0.02, max_assets=2, penalty_weight=1.0)
        assert isinstance(Q, np.ndarray), "QUBO matrix should be numpy array"
        assert Q.shape == (3, 3), "QUBO matrix should be 3x3"
        assert np.all(np.isfinite(Q)), "QUBO matrix should contain finite values"
        print("construct_qubo_matrix test passed")
        
        Q = np.array([
            [-1.0, 0.5, 0.3],
            [0.5, -1.2, 0.4],
            [0.3, 0.4, -0.8]
        ])
        
        solution, objective = solve_qubo_brute_force(Q, max_assets=2)
        assert isinstance(solution, np.ndarray), "Solution should be numpy array"
        assert len(solution) == 3, "Solution should have 3 elements"
        assert np.all((solution == 0) | (solution == 1)), "Solution should be binary"
        assert np.sum(solution) <= 2, "Solution should respect max_assets constraint"
        assert isinstance(objective, float), "Objective should be float"
        print("solve_qubo_brute_force test passed")
        
        Q = np.array([
            [-1.0, 0.5, 0.3, 0.2],
            [0.5, -1.2, 0.4, 0.1],
            [0.3, 0.4, -0.8, 0.3],
            [0.2, 0.1, 0.3, -0.9]
        ])
        
        solution, objective = solve_qubo_simulated_annealing(Q, max_assets=3, n_iterations=100)
        assert isinstance(solution, np.ndarray), "Solution should be numpy array"
        assert len(solution) == 4, "Solution should have 4 elements"
        assert np.all((solution == 0) | (solution == 1)), "Solution should be binary"
        assert np.sum(solution) <= 3, "Solution should respect max_assets constraint"
        assert isinstance(objective, float), "Objective should be float"
        print("solve_qubo_simulated_annealing test passed")
        
        weights = np.array([1, 0, 1, 0])
        mu = pd.Series([0.1, 0.12, 0.08, 0.15], index=['A', 'B', 'C', 'D'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.01, 0.02, 0.01],
            'B': [0.01, 0.09, 0.01, 0.02],
            'C': [0.02, 0.01, 0.16, 0.01],
            'D': [0.01, 0.02, 0.01, 0.25]
        }, index=['A', 'B', 'C', 'D'])
        
        metrics = calculate_portfolio_metrics(weights, mu, cov_matrix, 0.02)
        assert isinstance(metrics, dict), "Metrics should be dictionary"
        assert 'sharpe_ratio' in metrics, "Metrics should contain sharpe_ratio"
        assert 'expected_return' in metrics, "Metrics should contain expected_return"
        assert 'volatility' in metrics, "Metrics should contain volatility"
        assert 'num_assets' in metrics, "Metrics should contain num_assets"
        assert 'selected_assets' in metrics, "Metrics should contain selected_assets"
        assert metrics['num_assets'] == 2, "Should select 2 assets"
        assert len(metrics['selected_assets']) == 2, "Should have 2 selected assets"
        print("calculate_portfolio_metrics test passed")
        
        print("All QUBO Optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"QUBO Optimization test failed: {e}")
        return False

def test_integration():
    print("\nTesting Integration...")
    
    try:
        from portfolio.frontier import get_portfolio_data, get_risk_free_rate
        from portfolio.qubo_optimization import get_portfolio_data_qubo, get_risk_free_rate_qubo
        
        with patch('portfolio.frontier.yf.download') as mock_download, \
             patch('portfolio.qubo_optimization.yf.download') as mock_download_qubo:
            
            mock_data = pd.DataFrame({
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='ME'))
            
            mock_download.return_value = mock_data
            mock_download_qubo.return_value = mock_data
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns_qubo, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean_qubo:
                
                mock_returns.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
                mock_mean.return_value = pd.Series([0.15], index=['Adj Close'])
                mock_returns_qubo.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
                mock_mean_qubo.return_value = pd.Series([0.15], index=['Adj Close'])
                
                stock_df, _, ret_and_vol, _ = get_portfolio_data(100, False)
                stock_df_qubo, ret_and_vol_qubo = get_portfolio_data_qubo(100, False)
                
                assert stock_df.shape == stock_df_qubo.shape, "Data shapes should match"
                assert ret_and_vol.shape == ret_and_vol_qubo.shape, "Returns shapes should match"
                print("Data consistency test passed")
        
        rate_frontier = get_risk_free_rate('EUR', 'manual')
        rate_qubo = get_risk_free_rate_qubo('EUR', 'manual')
        
        assert abs(rate_frontier - rate_qubo) < 1e-10, "Risk-free rates should be consistent"
        print("Risk-free rate consistency test passed")
        
        print("All Integration tests passed!")
        return True
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

def test_performance():
    print("\nTesting Performance...")
    
    try:
        import time
        from portfolio.frontier import get_portfolio_data
        from portfolio.qubo_optimization import get_portfolio_data_qubo
        
        with patch('portfolio.frontier.yf.download') as mock_download, \
             patch('portfolio.qubo_optimization.yf.download') as mock_download_qubo:
            
            mock_data = pd.DataFrame(
                np.random.randn(50, 10),
                index=pd.date_range('2020-01-01', periods=50, freq='ME'),
                columns=[f'ASSET_{i}' for i in range(10)]
            )
            
            mock_download.return_value = mock_data
            mock_download_qubo.return_value = mock_data
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns_qubo, \
                 patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean_qubo:
                
                mock_returns.return_value = pd.DataFrame(
                    np.random.randn(49, 10),
                    columns=[f'ASSET_{i}' for i in range(10)]
                )
                mock_mean.return_value = pd.Series(
                    np.random.randn(10),
                    index=[f'ASSET_{i}' for i in range(10)]
                )
                mock_returns_qubo.return_value = pd.DataFrame(
                    np.random.randn(49, 10),
                    columns=[f'ASSET_{i}' for i in range(10)]
                )
                mock_mean_qubo.return_value = pd.Series(
                    np.random.randn(10),
                    index=[f'ASSET_{i}' for i in range(10)]
                )
                
                start_time = time.time()
                get_portfolio_data(100, False)
                frontier_time = time.time() - start_time
                
                start_time = time.time()
                get_portfolio_data_qubo(100, False)
                qubo_time = time.time() - start_time
                
                assert frontier_time < 5.0, f"Frontier optimization too slow: {frontier_time:.2f}s"
                assert qubo_time < 5.0, f"QUBO optimization too slow: {qubo_time:.2f}s"
                
                print(f"Performance test passed - Frontier: {frontier_time:.3f}s, QUBO: {qubo_time:.3f}s")
        
        print("All Performance tests passed!")
        return True
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False

def main():
    print("="*60)
    print("PORTFOLIO OPTIMIZATION TEST SUITE")
    print("="*60)
    
    tests = [
        test_frontier_optimization,
        test_qubo_optimization,
        test_integration,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "="*60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("All tests passed! Both optimization methods are working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())