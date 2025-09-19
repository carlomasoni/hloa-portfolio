from base64 import standard_b64decode
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy as sp
from datetime import datetime, timedelta  
import pypoft      

tickers = [ 'STOXX50E' ] # QQQ, VTI + other markets when needed
end = datetime.now()
start = end - timedelta(days=30)


adjusted_closed_dfs = pd.DataFrame([])
for ticker in tickers:
    adjusted_closed_dfs[ticker] = yf.download(ticker, start=start, end=end)['Adj Close']


# Returns

log_nreturns = np.log(adjusted_closed_dfs) / adjusted_closed_dfs.shift(1) 
log_nreturns = log_nreturns.dropna() 


cov_matrix = log_nreturns.cov()











