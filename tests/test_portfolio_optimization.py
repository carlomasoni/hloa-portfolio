import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio.frontier import (
    get_portfolio_data, 
    get_risk_free_rate, 
    optimize_portfolio_sharpe
)

class TestFrontierOptimization:
    
    @patch('portfolio.frontier.yf.download')
    def test_get_portfolio_data(self, mock_download):
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
            
            assert isinstance(stock_df, pd.DataFrame)
            assert isinstance(benchmark_df, pd.DataFrame)
            assert isinstance(ret_and_vol, pd.DataFrame)
            assert isinstance(corr_matrix, pd.DataFrame)
            assert len(stock_df.columns) > 0
    
    @patch('portfolio.frontier.yf.download')
    def test_get_risk_free_rate_yfinance(self, mock_download):
        mock_data = pd.DataFrame({
            'Close': [4.5]
        }, index=pd.date_range('2023-01-01', periods=1))
        
        mock_download.return_value = mock_data
        
        rate = get_risk_free_rate(currency='USD', source='yfinance')
        
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
    
    def test_get_risk_free_rate_manual(self):
        rate = get_risk_free_rate(currency='EUR', source='manual')
        
        assert isinstance(rate, float)
        assert rate == 0.025
    
    @patch('portfolio.frontier.get_portfolio_data')
    @patch('portfolio.frontier.get_risk_free_rate')
    @patch('hloa.core.HLOA')
    def test_optimize_portfolio_sharpe(self, mock_hloa, mock_rf, mock_data):
        mock_data.return_value = (
            pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D']),
            pd.DataFrame(np.random.randn(10, 2), columns=['^GSPC', '^IXIC']),
            pd.DataFrame({
                'ret': [0.1, 0.12, 0.08, 0.15],
                'vol': [0.2, 0.18, 0.25, 0.22]
            }, index=['A', 'B', 'C', 'D']),
            pd.DataFrame(np.random.randn(4, 4), index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'])
        )
        
        mock_rf.return_value = 0.02
        
        mock_optimizer = MagicMock()
        mock_optimizer.run.return_value = (
            np.array([0.3, 0.4, 0.2, 0.1]),
            0.8,
            np.random.randn(100, 4),
            np.random.randn(100)
        )
        mock_hloa.return_value = mock_optimizer
        
        results = optimize_portfolio_sharpe(
            time_period_days=100,
            include_eurostoxx=False,
            risk_free_rate=0.02
        )
        
        assert isinstance(results, dict)
        assert 'optimal_weights' in results
        assert 'sharpe_ratio' in results
        assert 'expected_return' in results
        assert 'volatility' in results
        assert 'risk_free_rate' in results
        assert results['risk_free_rate'] == 0.02


class TestIntegration:
    
    def test_data_consistency(self):
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
                
                assert isinstance(stock_df, pd.DataFrame)
                assert isinstance(ret_and_vol, pd.DataFrame)
    
    def test_risk_free_rate_consistency(self):
        with patch('portfolio.frontier.yf.download') as mock_download:
            
            mock_data = pd.DataFrame({
                'Close': [3.5]
            }, index=pd.date_range('2023-01-01', periods=1))
            
            mock_download.return_value = mock_data
            
            rate_frontier = get_risk_free_rate('EUR', 'yfinance')
            
            assert isinstance(rate_frontier, float)
            assert 0.0 <= rate_frontier <= 1.0

def test_performance_benchmarks():
    with patch('portfolio.frontier.yf.download') as mock_download:
        
        mock_data = pd.DataFrame({
            'Adj Close': np.random.randn(50)
        }, index=pd.date_range('2020-01-01', periods=50, freq='ME'))
        
        mock_download.return_value = mock_data
        
        with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
             patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean:
            
            mock_returns.return_value = pd.DataFrame(np.random.randn(49, 10))
            mock_mean.return_value = pd.Series(np.random.randn(10))
            
            import time
            
            start_time = time.time()
            get_portfolio_data(100, False)
            frontier_time = time.time() - start_time
            
            assert frontier_time < 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
