
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio.frontier import optimize_portfolio_sharpe

def main():
    print("="*70)
    print("HLOA PORTFOLIO OPTIMIZATION FOR EUROSTOXX 50")
    print("="*70)

    print("\nStarting HLOA optimization...")
    print("Data: EuroStoxx 50 stocks + European indices")
    print("Time period: ~5.5 years of historical data")
    print("Objective: Maximize Sharpe ratio")

    try:
        results = optimize_portfolio_sharpe(
            time_period_days=2000,
            include_eurostoxx=True,
            risk_free_rate=None,
            currency='EUR'
        )
        
        print("\n" + "="*70)
        print("OPTIMIZATION COMPLETE!")
        print("="*70)

        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"   Sharpe Ratio:     {results['sharpe_ratio']:.4f}")
        print(f"   Expected Return:  {results['expected_return']:.4f} ({results['expected_return']*100:.2f}%)")
        print(f"   Volatility:       {results['volatility']:.4f} ({results['volatility']*100:.2f}%)")
        print(f"   Risk-free Rate:   {results['risk_free_rate']:.4f} ({results['risk_free_rate']*100:.2f}%)")
        print(f"   Number of Assets: {results['n_assets']}")
        print(f"\nOPTIMAL PORTFOLIO WEIGHTS:")
        print("-" * 50)

        sorted_weights = sorted(
            results['optimal_weights'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )

        total_weight = 0
        for i, (asset, weight) in enumerate(sorted_weights):
            if weight > 0.001:
                total_weight += weight
                print(f"   {i+1:2d}. {asset:<15s}: {weight:.3f} ({weight*100:5.1f}%)")

        print("-" * 50)
        print(f"   Total Weight:     {total_weight:.3f} ({total_weight*100:.1f}%)")
        
        print(f"\nTOP 5 ASSETS:")
        for i, (asset, weight) in enumerate(sorted_weights[:5]):
            if weight > 0.001:
                print(f"   {i+1}. {asset:<15s}: {weight*100:5.1f}%")

        non_zero_weights = [w for w in results['optimal_weights'].values() if w > 0.001]
        if len(non_zero_weights) > 1:
            max_weight = max(non_zero_weights)
            min_weight = min(non_zero_weights)
            concentration = max_weight / sum(non_zero_weights)

            print(f"\nDIVERSIFICATION METRICS:")
            print(f"   Assets with >0.1% weight: {len(non_zero_weights)}")
            print(f"   Maximum weight:           {max_weight:.3f} ({max_weight*100:.1f}%)")
            print(f"   Minimum weight:           {min_weight:.3f} ({min_weight*100:.1f}%)")
            print(f"   Concentration ratio:      {concentration:.3f}")
        
        print(f"\nOptimization completed successfully!")
        print(f"The HLOA algorithm found an optimal portfolio with Sharpe ratio of {results['sharpe_ratio']:.4f}")
        
        return results
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("This might be due to:")
        print("   - Network connectivity issues")
        print("   - Missing data for some assets")
        print("   - API rate limits")
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Successfully optimized EuroStoxx 50 portfolio using HLOA")
        print(f"Found {results['n_assets']} assets with optimal Sharpe ratio of {results['sharpe_ratio']:.4f}")
        print(f"Expected annual return: {results['expected_return']*100:.2f}%")
        print(f"Expected annual volatility: {results['volatility']*100:.2f}%")
    else:
        print(f"\n" + "="*70)
        print("OPTIMIZATION FAILED")
        print("="*70)
        print("Please check the error messages above and try again.")