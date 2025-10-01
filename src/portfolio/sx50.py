import calendar, requests
from io import StringIO
import pandas as pd
from datetime import datetime, timezone   

def fetch_latest_sx5e_constituents(max_months_back: int = 6):
    base = "https://www.stoxx.com/document/Reports/STOXXSelectionList"
    today = datetime.now(timezone.utc).date()

    for m in range(max_months_back + 1):
        y = today.year
        mo = today.month - m
        while mo <= 0:
            mo += 12
            y -= 1
        month_name = calendar.month_name[mo]  

        for d in range(1, 8):
            fname = f"slpublic_sx5e_{y}{mo:02d}{d:02d}.csv"
            url = f"{base}/{y}/{month_name}/{fname}"
            try:
                r = requests.get(url, timeout=10)
                if r.status_code == 200 and r.content and len(r.content) > 200:
                    df = pd.read_csv(StringIO(r.content.decode("utf-8")), sep=";")
                    if "RIC" in df.columns:
                        tickers = (
                            df["RIC"]
                            .dropna()
                            .astype(str)
                            .str.strip()
                            .tolist()
                        )
                        seen = set(); uniq = []
                        for t in tickers:
                            if t and t not in seen:
                                seen.add(t); uniq.append(t)
                        return uniq, url
            except Exception:
                pass

    raise RuntimeError("Could not fetch an SX5E selection list from STOXX within the lookback window.")