# HLOA Portfolio Optimization

Portfolio optimization using the Horned Lizard Optimization Algorithm (HLOA) for EuroStoxx 50 stocks.



## ---------- To - Do: ----------
```
-> Remove the RFRR from the sharpe calculation
-> compare to market weights
-> ensure HLOA is finding optimal solutions on n10 problems 
-> rework data collection to work offline for testing 

```







## Features

- **HLOA Optimization**: Continuous weight optimization using evolutionary algorithms
- **EuroStoxx 50 Support**: Optimized for European markets
- **Risk-Free Rate Calculation**: Automatic market data integration
- **Sharpe Ratio Maximization**: Risk-adjusted return optimization
- **5% Weight Cap**: Individual asset weight constraint


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



├── docs/                  # Documentation
│   └── BENCHMARK_TESTING.md
├── requirements.txt       # Dependencies
├── .gitignore            # Git ignore rules
└── README.md             # Updated project documentation
