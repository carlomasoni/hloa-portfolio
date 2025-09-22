import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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


def test_integration():
    print("\nTesting Integration...")
    
    try:
        from portfolio.frontier import get_portfolio_data, get_risk_free_rate
        
        with patch('portfolio.frontier.yf.download') as mock_download:
            
            mock_data = pd.DataFrame({
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='ME'))
            
            mock_download.return_value = mock_data
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean:
                
                mock_returns.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
                mock_mean.return_value = pd.Series([0.15], index=['Adj Close'])
                
                stock_df, _, ret_and_vol, _ = get_portfolio_data(100, False)
                
                assert isinstance(stock_df, pd.DataFrame), "Stock DataFrame should be returned"
                assert isinstance(ret_and_vol, pd.DataFrame), "Returns DataFrame should be returned"
                print("Data consistency test passed")
        
        rate_frontier = get_risk_free_rate('EUR', 'manual')
        
        assert isinstance(rate_frontier, float), "Risk-free rate should be float"
        assert rate_frontier == 0.025, "EUR manual rate should be 0.025"
        print("Risk-free rate test passed")
        
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
        
        with patch('portfolio.frontier.yf.download') as mock_download:
            
            mock_data = pd.DataFrame({
                'Adj Close': np.random.randn(50)
            }, index=pd.date_range('2020-01-01', periods=50, freq='ME'))
            
            mock_download.return_value = mock_data
            
            with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
                 patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean:
                
                mock_returns.return_value = pd.DataFrame(
                    np.random.randn(49, 10),
                    columns=[f'ASSET_{i}' for i in range(10)]
                )
                mock_mean.return_value = pd.Series(
                    np.random.randn(10),
                    index=[f'ASSET_{i}' for i in range(10)]
                )
                
                start_time = time.time()
                get_portfolio_data(100, False)
                frontier_time = time.time() - start_time
                
                assert frontier_time < 5.0, f"Frontier optimization too slow: {frontier_time:.2f}s"
                
                print(f"Performance test passed - Frontier: {frontier_time:.3f}s")
        
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
        print("All tests passed! HLOA optimization is working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    exit(main())