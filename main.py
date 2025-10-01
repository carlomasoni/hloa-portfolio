# main.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

from hloa.core import HLOA, HLOA_Config

LOOKBACK_DAYS = 2000
INTERVAL = "1mo"         
CAP = 0.05               
SEED = 42

TICKERS = [
    # France (12)
    "OR.PA","MC.PA","RMS.PA","AI.PA","SU.PA","SAF.PA","DG.PA","BN.PA","EL.PA","AIR.PA","KER.PA","ORA.PA",
    # Germany (14)
    "SAP.DE","SIE.DE","ALV.DE","DTE.DE","BAYN.DE","BMW.DE","VOW3.DE","IFX.DE","DB1.DE","RWE.DE",
    "MUV2.DE","BAS.DE","DHL.DE","EOAN.DE",
    # Netherlands (6)
    "ASML.AS","AD.AS","PHIA.AS","HEIA.AS","PRX.AS","INGA.AS",
    # Spain (5)
    "SAN.MC","BBVA.MC","ITX.MC","IBE.MC","TEF.MC",
    # Italy (7)
    "ENEL.MI","ENI.MI","ISP.MI","UCG.MI","STM.MI","PRY.MI","MONC.MI",
    # Belgium (3)
    "ABI.BR","KBC.BR","SOLB.BR",
    # Finland (2)
    "NOKIA.HE","SAMPO.HE",
    # Austria (1)
    "OMV.VI",
    # Portugal (1)
    "EDP.LS",
]

def project_capped_simplex(w, cap=0.05, total=1.0, tol=1e-12, max_iter=100):
    w = np.asarray(w, float)
    N = w.size
    if N * cap + 1e-15 < total:
        cap = 1.0 / N + 1e-12
    lo = np.min(w) - cap
    hi = np.max(w)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        x = np.minimum(cap, np.maximum(0.0, w - mid))
        s = x.sum()
        if abs(s - total) <= tol:
            break
        if s > total:
            lo = mid
        else:
            hi = mid
    x_sum = x.sum()
    if x_sum > 0:
        x *= total / x_sum
    np.clip(x, 0.0, cap, out=x)
    x *= total / max(x.sum(), 1e-16)
    return x

def sharpe_ratio(w, mu, cov, rf=0.0):
    w = np.asarray(w, float)
    ex = float(w @ (mu - rf))
    vol = float(np.sqrt(w @ cov.values @ w))
    if vol <= 0 or not np.isfinite(vol):
        return float("-inf")
    return ex / vol

def ensure_feasible_cap(N: int, cap: float) -> float:
    return (1.0 / N + 1e-12) if N * cap < 1.0 else cap

def get_prices_yf(tickers, lookback_days=LOOKBACK_DAYS, interval=INTERVAL):
    interval = interval.lower()
    ann = {"1d": 252, "1wk": 52, "1mo": 12}.get(interval, 12)

    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    start_str, end_str = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    closes = []
    chunk = 50
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i+chunk]
        df = yf.download(
            batch, start=start_str, end=end_str, interval=interval,
            auto_adjust=True, actions=False, progress=False, group_by="ticker", threads=True,
        )
        if df.empty:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            for sym in batch:
                if (sym, "Adj Close") in df.columns:
                    closes.append(df[sym]["Adj Close"].rename(sym))
                elif (sym, "Close") in df.columns:
                    closes.append(df[sym]["Close"].rename(sym))
        else:
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            closes.append(df[col].rename(batch[0]))

    if not closes:
        raise ValueError("No Yahoo prices downloaded for any symbol.")

    prices = pd.concat(closes, axis=1).sort_index()
    prices = prices.dropna(axis=1, how="all").dropna(axis=0, how="all")

    min_obs = max(8, int(1.0 * ann))
    keep = prices.columns[prices.notna().sum() >= min_obs]
    prices = prices[keep]

    if prices.shape[1] == 0:
        raise ValueError("No assets remain after cleaning; try interval='1wk' or shorter lookback.")
    return prices  

def optimize_portfolio_with_HLOA(
    lookback_days=LOOKBACK_DAYS,
    interval=INTERVAL,
    cap=CAP,
    seed=SEED,
):
    prices = get_prices_yf(TICKERS, lookback_days=lookback_days, interval=interval)

  
    monthly_returns_matrix = prices.pct_change().dropna(how="any")
    monthly_return = monthly_returns_matrix.mean(axis=0)                 
    monthly_standard_deviation = monthly_returns_matrix.std(axis=0)     
    cov_monthly = monthly_returns_matrix.cov()                           

    mu = monthly_return
    Sigma = cov_monthly


    N = prices.shape[1]
    cap = ensure_feasible_cap(N, cap)

    def fitness(batch):
        scores = np.empty(batch.shape[0], dtype=float)
        for i, w in enumerate(batch):
            wp = project_capped_simplex(w, total=1.0, cap=cap)
            scores[i] = sharpe_ratio(wp, mu, Sigma, rf=0.0)
        return scores

    lb = np.zeros(N); ub = np.ones(N)
    cfg = HLOA_Config(pop_size=200, iters=1000, seed=seed)
    opt = HLOA(obj=fitness, bounds=(lb, ub), config=cfg)
    w_best, _, _, _ = opt.run()

    w_opt = project_capped_simplex(w_best, total=1.0, cap=cap)
    w_opt = np.maximum(w_opt, 0.0)
    w_opt = w_opt / w_opt.sum()  # exact sum = 1


    return dict(sorted(zip(mu.index.tolist(), w_opt), key=lambda kv: kv[1], reverse=True))

def main():
    weights = optimize_portfolio_with_HLOA()
    for ticker, w in weights.items():
        print(f"{ticker},{w:.6f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
