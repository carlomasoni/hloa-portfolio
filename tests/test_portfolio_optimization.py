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
from portfolio.qubo_optimization import (
    get_portfolio_data_qubo,
    get_risk_free_rate_qubo,
    construct_qubo_matrix,
    solve_qubo_brute_force,
    solve_qubo_simulated_annealing,
    calculate_portfolio_metrics,
    optimize_portfolio_qubo
)

class TestFrontierOptimization:
    
    @patch('portfolio.frontier.yf.download')
    def test_get_portfolio_data(self, mock_download):
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5, freq='M'))
        
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
    @patch('portfolio.frontier.HLOA')
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
            np.array([0.3, 0.4, 0.2, 0.1]),  # weights
            0.8,  # fitness
            np.random.randn(100, 4),  # X_final
            np.random.randn(100)  # f_final
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

class TestQUBOOptimization:
    
    @patch('portfolio.qubo_optimization.yf.download')
    def test_get_portfolio_data_qubo(self, mock_download):
        mock_data = pd.DataFrame({
            'Adj Close': [100, 101, 102, 103, 104]
        }, index=pd.date_range('2023-01-01', periods=5, freq='M'))
        
        mock_download.return_value = mock_data
        
        with patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
             patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean:
            
            mock_returns.return_value = pd.DataFrame({'Adj Close': [0.01, 0.02, 0.01, 0.02]})
            mock_mean.return_value = pd.Series([0.15], index=['Adj Close'])
            
            stock_df, ret_and_vol = get_portfolio_data_qubo(
                time_period_days=100, 
                include_eurostoxx=False
            )
            
            assert isinstance(stock_df, pd.DataFrame)
            assert isinstance(ret_and_vol, pd.DataFrame)
            assert len(stock_df.columns) > 0
    
    @patch('portfolio.qubo_optimization.yf.download')
    def test_get_risk_free_rate_qubo_yfinance(self, mock_download):
        mock_data = pd.DataFrame({
            'Close': [3.2]
        }, index=pd.date_range('2023-01-01', periods=1))
        
        mock_download.return_value = mock_data
        
        rate = get_risk_free_rate_qubo(currency='EUR', source='yfinance')
        
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0
    
    def test_get_risk_free_rate_qubo_manual(self):
        rate = get_risk_free_rate_qubo(currency='USD', source='manual')
        
        assert isinstance(rate, float)
        assert rate == 0.045
    
    def test_construct_qubo_matrix(self):
        mu = pd.Series([0.1, 0.12, 0.08], index=['A', 'B', 'C'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.01, 0.02],
            'B': [0.01, 0.09, 0.01],
            'C': [0.02, 0.01, 0.16]
        }, index=['A', 'B', 'C'])
        
        Q = construct_qubo_matrix(mu, cov_matrix, 0.02, max_assets=2, penalty_weight=1.0)
        
        assert isinstance(Q, np.ndarray)
        assert Q.shape == (3, 3)
        assert np.all(np.isfinite(Q))
    
    def test_solve_qubo_brute_force(self):
        Q = np.array([
            [-1.0, 0.5, 0.3],
            [0.5, -1.2, 0.4],
            [0.3, 0.4, -0.8]
        ])
        
        solution, objective = solve_qubo_brute_force(Q, max_assets=2)
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 3
        assert np.all((solution == 0) | (solution == 1))
        assert np.sum(solution) <= 2
        assert isinstance(objective, float)
    
    def test_solve_qubo_simulated_annealing(self):
        Q = np.array([
            [-1.0, 0.5, 0.3, 0.2],
            [0.5, -1.2, 0.4, 0.1],
            [0.3, 0.4, -0.8, 0.3],
            [0.2, 0.1, 0.3, -0.9]
        ])
        
        solution, objective = solve_qubo_simulated_annealing(Q, max_assets=3, n_iterations=100)
        
        assert isinstance(solution, np.ndarray)
        assert len(solution) == 4
        assert np.all((solution == 0) | (solution == 1))
        assert np.sum(solution) <= 3
        assert isinstance(objective, float)
    
    def test_calculate_portfolio_metrics(self):
        weights = np.array([1, 0, 1, 0])
        mu = pd.Series([0.1, 0.12, 0.08, 0.15], index=['A', 'B', 'C', 'D'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.01, 0.02, 0.01],
            'B': [0.01, 0.09, 0.01, 0.02],
            'C': [0.02, 0.01, 0.16, 0.01],
            'D': [0.01, 0.02, 0.01, 0.25]
        }, index=['A', 'B', 'C', 'D'])
        
        metrics = calculate_portfolio_metrics(weights, mu, cov_matrix, 0.02)
        
        assert isinstance(metrics, dict)
        assert 'sharpe_ratio' in metrics
        assert 'expected_return' in metrics
        assert 'volatility' in metrics
        assert 'num_assets' in metrics
        assert 'selected_assets' in metrics
        assert metrics['num_assets'] == 2
        assert len(metrics['selected_assets']) == 2
    
    def test_calculate_portfolio_metrics_empty(self):
        weights = np.array([0, 0, 0, 0])
        mu = pd.Series([0.1, 0.12, 0.08, 0.15], index=['A', 'B', 'C', 'D'])
        cov_matrix = pd.DataFrame({
            'A': [0.04, 0.01, 0.02, 0.01],
            'B': [0.01, 0.09, 0.01, 0.02],
            'C': [0.02, 0.01, 0.16, 0.01],
            'D': [0.01, 0.02, 0.01, 0.25]
        }, index=['A', 'B', 'C', 'D'])
        
        metrics = calculate_portfolio_metrics(weights, mu, cov_matrix, 0.02)
        
        assert metrics['num_assets'] == 0
        assert metrics['sharpe_ratio'] == 0.0
        assert metrics['expected_return'] == 0.0
        assert metrics['volatility'] == 0.0
    
    @patch('portfolio.qubo_optimization.get_portfolio_data_qubo')
    @patch('portfolio.qubo_optimization.get_risk_free_rate_qubo')
    def test_optimize_portfolio_qubo(self, mock_rf, mock_data):
        mock_data.return_value = (
            pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D']),
            pd.DataFrame({
                'ret': [0.1, 0.12, 0.08, 0.15],
                'vol': [0.2, 0.18, 0.25, 0.22]
            }, index=['A', 'B', 'C', 'D'])
        )
        
        mock_rf.return_value = 0.025
        
        results = optimize_portfolio_qubo(
            time_period_days=100,
            include_eurostoxx=False,
            risk_free_rate=0.025,
            max_assets=3,
            method='brute_force'
        )
        
        assert isinstance(results, dict)
        assert 'binary_weights' in results
        assert 'qubo_objective' in results
        assert 'sharpe_ratio' in results
        assert 'expected_return' in results
        assert 'volatility' in results
        assert 'num_assets' in results
        assert 'selected_assets' in results
        assert 'method' in results
        assert results['method'] == 'brute_force'
        assert results['risk_free_rate'] == 0.025

class TestIntegration:
    
    def test_data_consistency(self):
        with patch('portfolio.frontier.yf.download') as mock_download, \
             patch('portfolio.qubo_optimization.yf.download') as mock_download_qubo:
            
            mock_data = pd.DataFrame({
                'Adj Close': [100, 101, 102, 103, 104]
            }, index=pd.date_range('2023-01-01', periods=5, freq='M'))
            
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
                
                assert stock_df.shape == stock_df_qubo.shape
                assert ret_and_vol.shape == ret_and_vol_qubo.shape
    
    def test_risk_free_rate_consistency(self):
        with patch('portfolio.frontier.yf.download') as mock_download, \
             patch('portfolio.qubo_optimization.yf.download') as mock_download_qubo:
            
            mock_data = pd.DataFrame({
                'Close': [3.5]
            }, index=pd.date_range('2023-01-01', periods=1))
            
            mock_download.return_value = mock_data
            mock_download_qubo.return_value = mock_data
            
            rate_frontier = get_risk_free_rate('EUR', 'yfinance')
            rate_qubo = get_risk_free_rate_qubo('EUR', 'yfinance')
            
            assert abs(rate_frontier - rate_qubo) < 1e-10

def test_performance_benchmarks():
    with patch('portfolio.frontier.yf.download') as mock_download, \
         patch('portfolio.qubo_optimization.yf.download') as mock_download_qubo:
        
        mock_data = pd.DataFrame({
            'Adj Close': np.random.randn(50, 10)
        }, index=pd.date_range('2020-01-01', periods=50, freq='M'))
        
        mock_download.return_value = mock_data
        mock_download_qubo.return_value = mock_data
        
        with patch('portfolio.frontier.pypfopt.expected_returns.returns_from_prices') as mock_returns, \
             patch('portfolio.frontier.pypfopt.expected_returns.mean_historical_return') as mock_mean, \
             patch('portfolio.qubo_optimization.pypfopt.expected_returns.returns_from_prices') as mock_returns_qubo, \
             patch('portfolio.qubo_optimization.pypfopt.expected_returns.mean_historical_return') as mock_mean_qubo:
            
            mock_returns.return_value = pd.DataFrame(np.random.randn(49, 10))
            mock_mean.return_value = pd.Series(np.random.randn(10))
            mock_returns_qubo.return_value = pd.DataFrame(np.random.randn(49, 10))
            mock_mean_qubo.return_value = pd.Series(np.random.randn(10))
            
            import time
            
            start_time = time.time()
            get_portfolio_data(100, False)
            frontier_time = time.time() - start_time
            
            start_time = time.time()
            get_portfolio_data_qubo(100, False)
            qubo_time = time.time() - start_time
            
            assert frontier_time < 5.0
            assert qubo_time < 5.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
