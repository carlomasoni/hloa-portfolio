# HLOA Portfolio Optimization

Portfolio optimization using the Horned Lizard Optimization Algorithm (HLOA) for EuroStoxx 50 stocks.



## ---------- To - Do: ----------
```
-> Remove the RFRR from the sharpe calculation [DONE]
-> compare to market weights []
-> ensure HLOA is finding optimal solutions on n10 problems [SORT OF DONE]
-> rework data collection to work offline for testing []
-> use fred rather than yfinance

```


## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install system-wide on Arch Linux
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Quick Start

```bash
# Run HLOA optimization for EuroStoxx 50
python run_eurostoxx_optimization.py

# Run HLOA benchmark
python scripts/benchmark_optimization.py


```

```


## Dependencies

- numpy >= 1.26.0
- pandas >= 0.19
- matplotlib >= 3.0
- yfinance >= 0.2.0
- PyPortfolioOpt >= 1.5.0
- scipy >= 1.3

## License

MIT License


