import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from portfolio.frontier import max_sharpe_capped
from portfolio.constraints import project_capped_simplex


def to_monthly_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert daily prices to monthly simple returns."""
    mclose = prices.resample("M").last()
    rets = mclose.pct_change().dropna(how="any")
    return rets


def slice_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    """Extract data for a specific month."""
    mask = (df.index.year == year) & (df.index.month == month)
    return df.loc[mask]


def estimate_monthly_mu_sigma(monthly_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Estimate mean returns and covariance matrix."""
    mu = monthly_returns.mean()
    cov = monthly_returns.cov()
    return mu, cov


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prices_csv", help="CSV with daily close prices, columns=tickers")
    parser.add_argument("--month", default="2025-08", help="Target month YYYY-MM")
    parser.add_argument("--cap", type=float, default=0.05, help="Weight cap per stock")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True).sort_index()
    monthly = to_monthly_simple_returns(prices)

    year, month = map(int, args.month.split("-"))
    month_rets = slice_month(monthly, year, month)
    if month_rets.empty:
        raise SystemExit("No returns found for the specified month.")

    mu, cov = estimate_monthly_mu_sigma(month_rets)
    w = max_sharpe_capped(mu, cov, cap=args.cap)

    exp_ret = float(w @ mu.values)
    vol = float(np.sqrt(w @ cov.values @ w))
    sharpe = exp_ret / (vol + 1e-12)

    print("Optimal weights (cap=%.2f):" % args.cap)
    for t, wi in zip(mu.index, w):
        if wi > 0:
            print(f"{t}: {wi:.4f}")
    print(f"Return: {exp_ret:.6f}, Vol: {vol:.6f}, Sharpe: {sharpe:.4f}")

    rng = np.random.default_rng(42)
    pts_r, pts_v = [], []
    for _ in range(2000):
        x = rng.random(mu.size)
        x = project_capped_simplex(x, total=1.0, cap=args.cap)
        pts_r.append(float(x @ mu.values))
        pts_v.append(float(np.sqrt(x @ cov.values @ x)))

    plt.figure(figsize=(7,5))
    plt.scatter(pts_v, pts_r, s=6, alpha=0.3, label="feasible")
    plt.scatter([vol], [exp_ret], c="red", s=60, label="max Sharpe")
    plt.xlabel("Monthly StdDev")
    plt.ylabel("Monthly Return")
    plt.legend()
    plt.tight_layout()
    out = Path("frontier_aug2025.png")
    plt.savefig(out)
    print(f"Saved plot to {out.resolve()}")


if __name__ == "__main__":
    main()


