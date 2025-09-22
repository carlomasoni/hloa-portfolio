#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from portfolio.frontier import optimize_portfolio_sharpe

def main():
    print("="*80)
    print("HLOA EUROSTOXX 50 PORTFOLIO OPTIMIZATION (SIMPLIFIED)")
    print("="*80)
    
    print("\nRunning HLOA optimization on EuroStoxx 50 data...")
    print("Using a smaller subset to avoid rate limiting issues...\n")
    
    try:
        results = optimize_portfolio_sharpe(
            time_period_days=1000,  # Reduced time period
            include_eurostoxx=True,
            currency='EUR'
        )
        
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"PORTFOLIO PERFORMANCE:")
        print(f"   Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
        print(f"   Expected Return:  {results['expected_return']:.4f} ({results['expected_return']*100:.2f}%)")
        print(f"   Volatility:       {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"   Risk-free Rate:   {results['risk_free_rate']:.4f} ({results['risk_free_rate']*100:.2f}%)")
        print(f"   Number of Assets: {results['n_assets']}")
        
        print(f"OPTIMAL PORTFOLIO WEIGHTS:")
        print("-" * 80)
        
        sorted_weights = sorted(results['optimal_weights'].items(), 
                              key=lambda x: x[1], reverse=True)
        
        total_weight = 0
        for i, (asset, weight) in enumerate(sorted_weights, 1):
            percentage = weight * 100
            total_weight += weight
            print(f"   {i:2d}. {asset:<12} : {weight:.3f} ({percentage:5.1f}%)")
        
        print("-" * 80)
        print(f"   Total Weight:     {total_weight:.3f} ({total_weight*100:.1f}%)")
        
        print(f"TOP 10 ASSETS:")
        for i, (asset, weight) in enumerate(sorted_weights[:10], 1):
            percentage = weight * 100
            print(f"   {i:2d}. {asset:<12} : {percentage:5.1f}%")
        
        print(f"DIVERSIFICATION METRICS:")
        weights_list = list(results['optimal_weights'].values())
        max_weight = max(weights_list)
        min_weight = min(weights_list)
        avg_weight = sum(weights_list) / len(weights_list)
        
        print(f"   Assets with >0.1% weight: {len([w for w in weights_list if w > 0.001])}")
        print(f"   Maximum weight:           {max_weight:.3f} ({max_weight*100:.1f}%)")
        print(f"   Minimum weight:           {min_weight:.3f} ({min_weight*100:.1f}%)")
        print(f"   Average weight:           {avg_weight:.3f} ({avg_weight*100:.1f}%)")
        print(f"   Concentration ratio:      {max_weight:.3f}")
        
        print(f"OPTIMIZATION SUCCESSFUL!")
        print(f"The HLOA algorithm found an optimal portfolio with Sharpe ratio of {results['sharpe_ratio']:.4f}")
        
    except Exception as e:
        print(f"ERROR: {e}")
        print("Please check your internet connection and try again.")
        return 1
    
    print("\n" + "="*80)
    return 0

if __name__ == "__main__":
    exit(main())

