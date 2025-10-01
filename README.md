# HLOA Portfolio Optimization

Portfolio optimization using the Horned Lizard Optimization Algorithm (HLOA) for EuroStoxx 50 stocks.



## ---------- To - Do: ----------
```
-> Remove the RFRR from the sharpe calculation [DONE]
-> ensure HLOA is finding optimal solutions on n10 problems [DONE]
-> rework data collection to work offline for testing []

```


## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install system-wide on Linux
python3 -m pip install --user --break-system-packages -r requirements.txt
```

## Quick Start

```bash
# Run HLOA optimization for EuroStoxx 50
python main.py

# Run HLOA benchmark
python scripts/test_hloa_benchmark
```


## License

MIT License


